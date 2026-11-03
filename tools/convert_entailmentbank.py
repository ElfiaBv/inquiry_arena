"""Turn EntailmentBank task 2 into Inquiry Arena scenarios (split knowledge + distractors).

For every EntailmentBank item:
  - the answer options come from ARC (joined on the item id, e.g. Mercury_SC_413544);
  - proof facts that only restate the question become public facts;
  - the other proof facts are split between the agents (see --mode);
  - EntailmentBank's own distractors are dealt so that every agent holds the same number of facts;
  - fact IDs are shuffled, so their order gives nothing away;
  - the gold tree, key facts and provenance are written to the file for evaluation (agents never see them).

Example:
  python convert_entailmentbank.py --eb-dir entailment_bank/data/public_dataset/entailment_trees_emnlp2021_data_v2/dataset/task_2 \
      --arc hf --split dev --agents 2 --mode step --seed 1 --distractors 3 --out scenarios/dev_n2_step_s1

--arc hf        downloads ARC from Hugging Face (needs `pip install datasets`)
--arc <folder>  reads ARC files you downloaded yourself (.jsonl in AI2's format, or .parquet from Hugging Face)

Modes for splitting the proof facts:
  random   facts dealt at random (every agent gets at least one; nobody gets all)
  step     the premises of each proof step go to different agents (tests whether agents see why a fact matters)
  subtree  each agent gets whole branches of the proof (tests whether partial conclusions get combined)
"""

import argparse
import csv
import glob
import json
import os
import random
import re
import sys

NAMES = ["Ana", "Bob", "Cara", "Dan", "Eve", "Finn"]
STOP = set("the a an of to in on for and or is are was were be by with as at from that this it its into than "
           "which what when where how why does do did can could would should will".split())


# ----------------------------------------------------------------- reading the data

def load_eb(eb_dir, split):
    path = os.path.join(eb_dir, f"{split}.jsonl")
    if not os.path.exists(path):
        sys.exit(f"Cannot find {path}. Point --eb-dir at the task_2 folder of EntailmentBank.")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _arc_row(obj):
    """Return (id, stem, [(label, text)], answerKey) for ARC rows in AI2 or Hugging Face format."""
    q = obj.get("question")
    if isinstance(q, dict):  # AI2 jsonl format
        stem = q.get("stem", "")
        choices = [(c.get("label"), c.get("text")) for c in q.get("choices", [])]
    else:  # Hugging Face format
        stem = q or ""
        ch = obj.get("choices") or {}
        choices = list(zip(ch.get("label", []), ch.get("text", [])))
    return obj.get("id"), stem, choices, obj.get("answerKey")


def load_arc(source):
    rows = []
    if source == "hf":
        try:
            from datasets import load_dataset
        except ImportError:
            sys.exit("--arc hf needs the datasets package: pip install datasets")
        for cfg in ("ARC-Easy", "ARC-Challenge"):
            ds = load_dataset("allenai/ai2_arc", cfg)
            for split in ds:
                rows += [dict(r) for r in ds[split]]
    else:
        files = glob.glob(os.path.join(source, "**", "*.jsonl"), recursive=True)
        files += glob.glob(os.path.join(source, "**", "*.parquet"), recursive=True)
        if not files:
            sys.exit(f"No .jsonl or .parquet ARC files found under {source}")
        for fn in files:
            if fn.endswith(".jsonl"):
                with open(fn, encoding="utf-8") as f:
                    rows += [json.loads(line) for line in f if line.strip()]
            else:
                import pandas as pd
                for r in pd.read_parquet(fn).to_dict("records"):
                    ch = r.get("choices") or {}
                    r["choices"] = {"text": list(ch.get("text", [])), "label": list(ch.get("label", []))}
                    rows.append(r)
    arc = {}
    for r in rows:
        rid, stem, choices, key = _arc_row(r)
        if rid and choices:
            arc[rid] = {"stem": stem, "choices": choices, "answerKey": key}
    return arc


# ----------------------------------------------------------------- helpers

def words(s):
    return {w for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 2 and w not in STOP}


def norm(s):
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def parse_proof(proof):
    """'sent1 & sent3 -> int1: text; int1 & sent2 -> hypothesis;' -> [(premises, conclusion_id, text)]"""
    steps = []
    for part in proof.split(";"):
        if "->" not in part:
            continue
        lhs, rhs = part.split("->", 1)
        prem = [p.strip() for p in lhs.split("&") if p.strip()]
        rhs = rhs.strip()
        if rhs.startswith("hypothesis"):
            concl, text = "hypothesis", ""
        else:
            m = re.match(r"(int\d+)\s*:?\s*(.*)", rhs)
            if not m:
                continue
            concl, text = m.group(1), m.group(2).strip()
        steps.append((prem, concl, text))
    return steps


def leaves_under(node, by_concl, leafset, seen=None):
    seen = seen or set()
    if node in leafset:
        return {node}
    if node in seen or node not in by_concl:
        return set()
    seen.add(node)
    out = set()
    for p in by_concl[node][0]:
        out |= leaves_under(p, by_concl, leafset, seen)
    return out


def options_from_arc(item, arc):
    a = arc.get(item["id"])
    if not a:
        return None, None, "no ARC entry for this id"
    letters = "ABCDEFGH"
    opts = {letters[i]: t for i, (_, t) in enumerate(a["choices"])}
    labels = [lab for lab, _ in a["choices"]]
    if a["answerKey"] not in labels:
        return None, None, "ARC answer key not among the choices"
    gold = letters[labels.index(a["answerKey"])]
    eb, arc_t = norm(item["answer"]), norm(opts[gold])
    overlap = len(words(item["answer"]) & words(opts[gold])) / max(1, len(words(item["answer"]) | words(opts[gold])))
    if eb != arc_t and eb.replace(" ", "") != arc_t.replace(" ", "") and overlap < 0.8:
        return None, None, f"EntailmentBank answer '{item['answer']}' does not match ARC's '{opts[gold]}'"
    return opts, gold, None


# ----------------------------------------------------------------- splitting the proof facts

def allocate(private, steps, by_concl, n, mode, rng):
    """Return {leaf: agent index}. Every agent gets at least one leaf; nobody gets all of them."""
    leafset = set(private)
    if mode == "random":
        for _ in range(2000):
            owner = {l: rng.randrange(n) for l in private}
            if len(set(owner.values())) == n:
                return owner, "random"
        mode = "step"
    if mode == "subtree":
        groups = [sorted(leaves_under(p, by_concl, leafset)) for p in by_concl["hypothesis"][0]] if "hypothesis" in by_concl else []
        groups = [g for g in groups if g]
        # split the biggest branch until there are enough branches for everyone
        while len(groups) < n:
            groups.sort(key=len, reverse=True)
            big = groups[0]
            sub = None
            for concl, (prem, _) in by_concl.items():
                if concl != "hypothesis" and set(leaves_under(concl, by_concl, leafset)) == set(big):
                    sub = [sorted(leaves_under(p, by_concl, leafset)) for p in prem]
                    sub = [g for g in sub if g]
                    break
            if not sub or len(sub) < 2:
                break
            groups = sub + groups[1:]
        if len(groups) >= n:
            groups.sort(key=len, reverse=True)
            sizes = [0] * n
            owner = {}
            for g in groups:  # biggest branch to the agent holding least so far
                k = sizes.index(min(sizes))
                for l in g:
                    owner.setdefault(l, k)
                sizes[k] += len(g)
            for l in private:
                owner.setdefault(l, rng.randrange(n))
            if len(set(owner.values())) == n:
                return owner, "subtree"
        mode = "step"  # a single chain cannot be split by branches
    # step: premises of the same step go to different agents
    owner, ptr = {}, rng.randrange(n)
    for prem, _, _ in steps:
        for p in prem:
            if p in leafset and p not in owner:
                owner[p] = ptr
                ptr = (ptr + 1) % n
    for l in private:
        if l not in owner:
            owner[l] = ptr
            ptr = (ptr + 1) % n
    counts = [list(owner.values()).count(k) for k in range(n)]
    for k in range(n):  # make sure nobody is empty
        if counts[k] == 0:
            donor = counts.index(max(counts))
            l = next(x for x in private if owner[x] == donor)
            owner[l] = k
            counts[k], counts[donor] = 1, counts[donor] - 1
    return owner, "step"


def cross_steps(steps, by_concl, leafset, owner):
    """Proof steps whose premises are held by more than one agent."""
    c = 0
    for prem, _, _ in steps:
        holders = set()
        for p in prem:
            holders |= {owner[l] for l in leaves_under(p, by_concl, leafset) if l in owner}
        c += len(holders) >= 2
    return c


# ----------------------------------------------------------------- one item -> one scenario

def convert(item, arc, n, mode, rng, n_distractors, min_private, split):
    m = item["meta"]
    triples = m["triples"]
    distractors = list(m.get("distractors", []))
    rel = m.get("distractors_relevance") or [0.0] * len(distractors)
    leaves = [k for k in triples if k not in set(distractors)]
    steps = parse_proof(item.get("proof") or m.get("step_proof", ""))
    if not steps:
        return None, "no proof steps"
    by_concl = {c: (p, t) for p, c, t in steps}
    opts, gold, why = options_from_arc(item, arc)
    if why:
        return None, why

    q = words(item["question"])
    public = [l for l in leaves if words(triples[l]) and len(words(triples[l]) & q) / len(words(triples[l])) >= 0.6]
    private = [l for l in leaves if l not in public]
    if len(private) < max(min_private, n):
        return None, f"only {len(private)} private proof facts for {n} agents"

    owner, mode_used = allocate(private, steps, by_concl, n, mode, rng)
    held = [[l for l in private if owner[l] == k] for k in range(n)]

    # distractors: highest relevance first, dealt so everyone ends with the same number of facts
    ranked = [d for _, d in sorted(zip(rel, distractors), key=lambda x: -x[0])]
    high = set(ranked[: max(1, len(ranked) // 3)])
    target = max(len(h) for h in held) + (len(ranked) if n_distractors == "all" else int(n_distractors))
    need = [target - len(h) for h in held]
    while sum(need) > len(ranked):
        target -= 1
        need = [max(0, target - len(h)) for h in held]
    dealt = [[] for _ in range(n)]
    queue = list(ranked)
    while any(len(dealt[k]) < need[k] for k in range(n)) and queue:
        for k in range(n):
            if len(dealt[k]) < need[k] and queue:
                dealt[k].append(queue.pop(0))

    # build the agents with shuffled, renumbered fact IDs
    newid, agents, fact_types, fact_sources = {}, [], {}, {}
    for k in range(n):
        name = NAMES[k]
        items = held[k] + dealt[k]
        rng.shuffle(items)
        facts = {}
        for i, sid in enumerate(items, 1):
            fid = f"{name[0]}{i}"
            newid[sid] = fid
            facts[fid] = triples[sid]
            fact_sources[fid] = f"EB {sid}"
            fact_types[fid] = ("key" if sid in owner else
                               "distractor (high relevance)" if sid in high else "distractor (low relevance)")
        agents.append({"name": name, "facts": facts})
    public_facts = {}
    for i, sid in enumerate(public, 1):
        newid[sid] = f"P{i}"
        public_facts[f"P{i}"] = triples[sid]
        fact_sources[f"P{i}"] = f"EB {sid}"
        fact_types[f"P{i}"] = "key (restates the question; public)"

    def name_of(x):
        if x == "hypothesis":
            return "hypothesis"
        return newid.get(x, x)

    gold_tree = [f"{' + '.join(name_of(p) for p in prem)} -> {name_of(c)}"
                 + (f": {t}" if t else f": {item['hypothesis']}") for prem, c, t in steps]
    explanation = " ".join(
        f"{'; '.join(triples.get(p, m.get('intermediate_conclusions', {}).get(p, by_concl.get(p, ('', p))[1])) for p in prem)}, "
        f"so {t or item['hypothesis']}." for prem, c, t in steps)
    xs = cross_steps(steps, by_concl, set(private), owner)
    scenario = {
        "title": f"EntailmentBank {item['id']} ({n} agents, {mode_used} split)",
        "question": item["question"],
        "options": opts,
        "answer": gold,
        "public_facts": public_facts,
        "agents": agents,
        "key_facts": sorted(newid[l] for l in private),
        "gold_explanation": explanation,
        "gold_tree": gold_tree,
        "fact_types": fact_types,
        "fact_sources": fact_sources,
        "notes": (f"Generated from EntailmentBank task 2 ({split}, {item['id']}). Proof facts split with mode "
                  f"'{mode_used}'; {xs} of {len(steps)} proof steps need facts from more than one agent. "
                  f"Distractors are EntailmentBank's own, highest relevance first."),
        "source": {"dataset": "EntailmentBank (Dalvi et al., EMNLP 2021)", "task": "task_2", "split": split,
                   "item": item["id"], "hypothesis": item["hypothesis"], "options": "ARC (joined on id)"},
        "allocation": {"mode": mode_used, "mode_requested": mode, "n_agents": n,
                       "distractors_per_agent": n_distractors},
        "difficulty": {"proof_facts": len(leaves), "private_proof_facts": len(private), "public_proof_facts": len(public),
                       "proof_steps": len(steps), "cross_agent_steps": xs, "depth": item.get("depth_of_proof")},
    }
    return scenario, None


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eb-dir", required=True, help="EntailmentBank task_2 folder (with train/dev/test.jsonl)")
    ap.add_argument("--arc", required=True, help="'hf' or a folder with ARC .jsonl/.parquet files")
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--agents", type=int, default=2)
    ap.add_argument("--mode", default="step", choices=["random", "step", "subtree"])
    ap.add_argument("--distractors", default="3", help="distractors per agent on top of its proof facts, or 'all'")
    ap.add_argument("--min-private", type=int, default=3, help="skip items with fewer private proof facts")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many scenarios (0 = all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.agents < 2 or a.agents > len(NAMES):
        sys.exit(f"--agents must be between 2 and {len(NAMES)}")

    items = load_eb(a.eb_dir, a.split)
    arc = load_arc(a.arc)
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    made, skipped, seen = [], [], set()
    for item in items:
        if a.limit and len(made) >= a.limit:
            break
        if item["id"] in seen:  # EntailmentBank repeats a few questions; keep the first usable one
            skipped.append({"item": item["id"], "reason": "item id already converted"})
            continue
        sc, why = convert(item, arc, a.agents, a.mode, rng, a.distractors, a.min_private, a.split)
        if sc is not None:
            seen.add(item["id"])
        if sc is None:
            skipped.append({"item": item["id"], "reason": why})
            continue
        fn = f"scenario_eb_{a.split}_{item['id']}_{sc['allocation']['mode']}_n{a.agents}_s{a.seed}.json"
        with open(os.path.join(a.out, fn), "w", encoding="utf-8") as f:
            json.dump(sc, f, indent=2, ensure_ascii=False)
        d = sc["difficulty"]
        made.append({"file": fn, "item": item["id"], "split": a.split, "agents": a.agents,
                     "mode": sc["allocation"]["mode"], "seed": a.seed, "answer": sc["answer"],
                     "options": len(sc["options"]), "proof_facts": d["proof_facts"],
                     "private_proof_facts": d["private_proof_facts"], "public_proof_facts": d["public_proof_facts"],
                     "proof_steps": d["proof_steps"], "cross_agent_steps": d["cross_agent_steps"],
                     "depth": d["depth"], "facts_per_agent": "/".join(str(len(x["facts"])) for x in sc["agents"])})
    for name, rows in (("index.csv", made), ("skipped.csv", skipped)):
        if rows:
            with open(os.path.join(a.out, name), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print(f"{len(made)} scenarios written to {a.out}; {len(skipped)} items skipped (see skipped.csv)")
    reasons = {}
    for s in skipped:
        key = re.sub(r"'.*", "...", s["reason"])
        reasons[key] = reasons.get(key, 0) + 1
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {c:5d}  {r}")


if __name__ == "__main__":
    main()
