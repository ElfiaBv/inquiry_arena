"""Inquiry Arena: LLM agents with split knowledge play an inquiry dialogue.

The referee (this program) is the only one that sees every fact. Each LLM
sees the public facts, its own private facts, and the conversation so far.
Moves: assert (share facts by ID + a claim), why (ask someone to justify), pass.
The dialogue ends when every agent passes in a row, or at the turn limit.
A judge model then reads ONLY the public record and picks an answer.
Optionally the judge also turns the record into an argument graph; the app
(not the LLM) labels it accepted / rejected / undecided with grounded semantics.
"""

import glob
import json
import os
import random
import textwrap
from datetime import datetime, timezone

import requests
import streamlit as st

st.set_page_config(page_title="Inquiry Arena", page_icon="🔎", layout="wide")

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
AVATARS = ["🟦", "🟧", "🟩", "🟪", "🟥", "🟨", "⬛", "⬜"]

# Defaults are picked from this list if OpenRouter still offers them.
PREFERRED_MODELS = [
    "openai/gpt-4o-mini", "openai/gpt-4.1-mini", "openai/gpt-5-mini",
    "anthropic/claude-3.5-haiku", "anthropic/claude-haiku-4.5",
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen-2.5-72b-instruct", "qwen/qwen3-32b",
    "google/gemini-2.0-flash-001", "google/gemini-2.5-flash",
    "mistralai/mistral-small-3.2-24b-instruct",
]
FALLBACK_MODEL = "openai/gpt-4o-mini"

RULES = """You are {name}{role}, one of {n} AI agents working together on an inquiry.
The group's goal is to find the correct answer to the question, with a short explanation.

How it works:
- Everyone sees the public facts and the conversation.
- Each agent also has PRIVATE facts that nobody else can see. Others only learn your facts if you share them.
- Be honest: only share facts from YOUR OWN list, citing their IDs. Never invent facts or IDs.
- You don't have to share everything. Share what helps the group answer correctly.
- You may also reason from facts already on the table, ask another agent "why", or pass.

On each turn reply with ONLY one JSON object, nothing else. One of:
{{"move": "assert", "facts": ["ID", ...], "claim": "what you conclude, in one or two sentences"}}
{{"move": "why", "to": "<agent name>", "question": "what you want them to justify"}}
{{"move": "pass"}}

The other agents are: {others}.
The dialogue ends when every agent passes in a row."""

JUDGE_SYSTEM = ("You are a neutral judge. You know only what is written in the message. "
                "Reply with ONLY one JSON object, nothing else.")

GRAPH_SYSTEM = ("You turn a discussion into an argument graph. You know only what is written in the message. "
                "Reply with ONLY one JSON object, nothing else.")

GRAPH_INSTRUCTIONS = """Build the argument graph of this discussion.
- Make one node for each distinct argument an agent actually put forward. Merge repeats. Do not invent arguments nobody made.
- Each node has: "id" ("a1", "a2", ...), "by" (agent name), "turn" (turn where it first appeared),
  "facts" (the fact IDs it relies on), "text" (one short sentence: premises, therefore conclusion).
- The answer options are already nodes, with the IDs listed above (opt_A, opt_B, ...).
- "attacks": pairs [attacker, target] where the attacker argues against the target: it undermines the target
  argument, or it rules out the target answer option.
- "supports": pairs [supporter, target] where the supporter argues for the target argument or answer option.
- Only include relations that are stated or clearly implied in the discussion.
Reply with ONLY:
{"arguments": [{"id": "a1", "by": "Alice", "turn": 1, "facts": ["A1"], "text": "..."}],
 "attacks": [["a2", "opt_A"]], "supports": [["a1", "opt_B"]]}"""

LABEL_STYLE = {  # fill, border
    "IN": ("#d3f2da", "#1e7b34"),
    "OUT": ("#f9d6d5", "#b42318"),
    "UNDEC": ("#e7e7e7", "#6b6b6b"),
}

FORMAT_HELP = """{
  "title": "Short name",
  "question": "The question the agents must answer",
  "options": {"A": "...", "B": "...", "C": "..."},
  "answer": "B",
  "public_facts": {"P1": "Everyone knows this"},
  "agents": [
    {"name": "Alice", "role": "a fisher", "facts": {"A1": "Only Alice knows this"}},
    {"name": "Bob",   "role": "a lab tech", "facts": {"B1": "Only Bob knows this"}}
  ],
  "key_facts": ["A1", "B1"],
  "gold_explanation": "What a good explanation says"
}
Fact IDs must be unique. key_facts and gold_explanation are optional."""


# ----------------------------------------------------------------- helpers

def secret(name):
    try:
        return st.secrets.get(name)
    except Exception:  # no secrets file at all
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_models():
    """Public list of OpenRouter models: {id: (price_in_per_token, price_out_per_token)}."""
    try:
        r = requests.get(f"{OPENROUTER_BASE}/models", timeout=15)
        r.raise_for_status()
        out = {}
        for m in r.json().get("data", []):
            p = m.get("pricing") or {}
            try:
                out[m["id"]] = (float(p.get("prompt", 0)), float(p.get("completion", 0)))
            except (TypeError, ValueError):
                out[m["id"]] = (0.0, 0.0)
        return dict(sorted(out.items()))
    except Exception:
        return {}


def price_label(model_id, models):
    pin, pout = models.get(model_id, (0.0, 0.0))
    if pin == 0 and pout == 0:
        return model_id + "  (free or unknown price)"
    return f"{model_id}  (${pin * 1e6:.2f} in / ${pout * 1e6:.2f} out per 1M tokens)"


def load_builtin_scenarios():
    out = {}
    for path in sorted(glob.glob(os.path.join(APP_DIR, "scenario_*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                sc = json.load(f)
            out[sc.get("title", os.path.basename(path))] = sc
        except Exception as e:
            st.sidebar.warning(f"Could not read {os.path.basename(path)}: {e}")
    return out


def validate(sc):
    if not isinstance(sc, dict):
        return ["the file must contain one JSON object"]
    errors = [f"missing '{k}'" for k in ("question", "options", "answer", "agents") if k not in sc]
    if errors:
        return errors
    ids = list(sc.get("public_facts", {}).keys())
    names = []
    for a in sc["agents"]:
        if not isinstance(a, dict) or "name" not in a or "facts" not in a:
            errors.append("each agent needs a 'name' and 'facts'")
            continue
        names.append(a["name"])
        ids += list(a["facts"].keys())
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        errors.append(f"duplicate fact IDs: {dup}")
    if len(set(names)) != len(names):
        errors.append("agent names must be unique")
    if sc["answer"] not in sc["options"]:
        errors.append("'answer' must be one of the option letters")
    if len(names) < 2:
        errors.append("you need at least two agents")
    return errors


def parse_json(text):
    """Return the first JSON object found in the text, or None."""
    text = text or ""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None


def all_facts(sc):
    out = dict(sc.get("public_facts", {}))
    for a in sc["agents"]:
        out.update(a["facts"])
    return out


def verdict(result):
    if result.get("error"):
        return "error"
    return "correct" if result["correct"] else "wrong"


def options_text(sc):
    return "\n".join(f"{k}. {v}" for k, v in sc["options"].items())


def facts_text(d):
    return "\n".join(f"{k}: {v}" for k, v in d.items()) or "(none)"


def table_text(table, fact_text):
    lines = []
    for t in table:
        who = t["agent"]
        if t["move"] == "assert":
            s = f"Turn {t['turn']} - {who} asserts."
            shared = "; ".join(f'{fid}: "{fact_text[fid]}"' for fid in t["valid_facts"])
            if shared:
                s += f" Cites {shared}."
            if t.get("claim"):
                s += f" Claim: {t['claim']}"
        elif t["move"] == "why":
            s = f"Turn {t['turn']} - {who} asks {t.get('to') or 'the group'} why: {t.get('question', '')}"
        else:
            s = f"Turn {t['turn']} - {who} passes."
        lines.append(s)
    return "\n".join(lines) if lines else "(nothing yet - you speak first)"


# ----------------------------------------------------------------- argument graph

def grounded_labels(node_ids, attacks):
    """Grounded labelling: IN if all attackers are OUT, OUT if some attacker is IN, repeat; the rest UNDEC."""
    attackers = {n: set() for n in node_ids}
    for a, b in attacks:
        attackers[b].add(a)
    label = {}
    changed = True
    while changed:
        changed = False
        for n in node_ids:
            if n in label:
                continue
            if all(label.get(x) == "OUT" for x in attackers[n]):
                label[n], changed = "IN", True
            elif any(label.get(x) == "IN" for x in attackers[n]):
                label[n], changed = "OUT", True
    return {n: label.get(n, "UNDEC") for n in node_ids}


def build_graph(obj, sc, support_rule):
    """Clean up the judge's graph and compute labels. Options exclude each other."""
    letters = list(sc["options"])
    nodes, warnings = [], []
    for k in letters:
        nodes.append({"id": f"opt_{k}", "kind": "option", "by": "", "turn": None, "facts": [],
                      "text": f"{k}. {sc['options'][k]}"})
    ids = {n["id"] for n in nodes}
    args = obj.get("arguments") if isinstance(obj.get("arguments"), list) else []
    for i, a in enumerate(args, 1):
        if not isinstance(a, dict):
            continue
        nid = str(a.get("id") or f"a{i}").strip()
        if nid in ids or nid.lower().startswith("opt"):
            warnings.append(f"renamed duplicate or reserved node id '{nid}'")
            nid = f"a{i}_{len(ids)}"
        facts = a.get("facts") or []
        facts = [str(f) for f in (facts if isinstance(facts, list) else [facts])]
        try:
            turn = int(a.get("turn"))
        except (TypeError, ValueError):
            turn = None
        nodes.append({"id": nid, "kind": "argument", "by": str(a.get("by", "")), "turn": turn,
                      "facts": facts, "text": str(a.get("text", ""))[:300]})
        ids.add(nid)

    def norm(x):
        x = str(x).strip()
        if x in ids:
            return x
        up = x.upper()
        for k in letters:  # accept "B", "opt_B", "Option B", "OPT_B"
            if up in (k.upper(), f"OPT_{k.upper()}", f"OPTION {k.upper()}", f"OPTION_{k.upper()}"):
                return f"opt_{k}"
        return None

    def edges(key):
        out = []
        raw_edges = obj.get(key) or []
        if not isinstance(raw_edges, list):
            warnings.append(f"skipped '{key}': expected a list of pairs")
            raw_edges = []
        for e in raw_edges:
            if isinstance(e, dict):
                e = [e.get("from") or e.get("source"), e.get("to") or e.get("target")]
            if not isinstance(e, (list, tuple)) or len(e) != 2:
                warnings.append(f"skipped malformed {key[:-1]} {e!r}")
                continue
            a, b = norm(e[0]), norm(e[1])
            if a is None or b is None:
                warnings.append(f"skipped {key[:-1]} {list(e)} (unknown node)")
                continue
            if a.startswith("opt_") and b.startswith("opt_"):
                continue  # options already exclude each other
            if [a, b] not in out:
                out.append([a, b])
        return out

    attacks, supports = edges("attacks"), edges("supports")
    exclusion = [[f"opt_{x}", f"opt_{y}"] for x in letters for y in letters if x != y]
    via_support = []
    if support_rule:
        for a, b in supports:
            if b.startswith("opt_") and not a.startswith("opt_"):
                via_support += [[a, f"opt_{y}"] for y in letters if f"opt_{y}" != b]
    all_attacks = attacks + exclusion + [e for e in via_support if e not in attacks]
    order = [n["id"] for n in nodes]
    labels = grounded_labels(order, all_attacks)
    winners = [k for k in letters if labels[f"opt_{k}"] == "IN"]
    return {"nodes": nodes, "attacks": attacks, "supports": supports, "exclusion": exclusion,
            "via_support": via_support, "support_rule": support_rule, "labels": labels,
            "verdict": winners[0] if len(winners) == 1 else "", "warnings": warnings}


def _dot_text(s, width=34):
    s = "\n".join(textwrap.wrap(str(s), width)) or " "
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def graph_dot(g):
    lines = ['digraph G {',
             '  graph [rankdir=BT, bgcolor="white", nodesep=0.35, ranksep=0.55];',
             '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, penwidth=1.6];',
             '  edge [fontname="Helvetica", fontsize=9];']
    opts = []
    for n in g["nodes"]:
        fill, border = LABEL_STYLE[g["labels"][n["id"]]]
        if n["kind"] == "option":
            label = _dot_text(n["text"], 28)
            lines.append(f'  "{n["id"]}" [label="{label}", fillcolor="{fill}", color="{border}", '
                         f'penwidth=3, fontsize=12];')
            opts.append(f'"{n["id"]}"')
        else:
            head = f'{n["id"]} · {n["by"]}' + (f' (turn {n["turn"]})' if n["turn"] else "")
            body = _dot_text(n["text"])
            facts = f'\\n[{", ".join(n["facts"])}]' if n["facts"] else ""
            lines.append(f'  "{n["id"]}" [label="{_dot_text(head, 60)}\\n{body}{facts}", '
                         f'fillcolor="{fill}", color="{border}"];')
    if opts:
        lines.append(f'  {{ rank=same; {"; ".join(opts)}; }}')
        if len(opts) > 1:  # keep options in A, B, C order
            lines.append(f'  {" -> ".join(opts)} [style=invis];')
    for x, y in g["attacks"]:
        lines.append(f'  "{x}" -> "{y}" [color="#b42318", penwidth=1.6];')
    for x, y in g["supports"]:
        lines.append(f'  "{x}" -> "{y}" [color="#1e7b34", style=dashed, arrowhead=empty];')
    seen = set()
    for x, y in g["exclusion"]:
        if (y, x) not in seen:
            seen.add((x, y))
            lines.append(f'  "{x}" -> "{y}" [color="#9a9a9a", style=dotted, dir=none, constraint=false];')
    lines.append("}")
    return "\n".join(lines)


# ----------------------------------------------------------------- the referee

class Referee:
    def __init__(self, sc, models, judge_model, settings, api_key, prices):
        self.sc = sc
        self.models = models
        self.judge_model = judge_model
        self.settings = settings
        self.prices = prices
        self.mock = settings["mock"]
        self.client = None
        if not self.mock:
            from openai import OpenAI
            self.client = OpenAI(base_url=OPENROUTER_BASE, api_key=api_key)
        self.agents = sc["agents"]
        self.names = [a["name"] for a in self.agents]
        self.public = set(sc.get("public_facts", {}))
        self.fact_text = dict(sc.get("public_facts", {}))
        self.owner = {}
        for a in self.agents:
            for fid, txt in a["facts"].items():
                self.fact_text[fid] = txt
                self.owner[fid] = a["name"]
        self.on_table = set(self.public)
        self.first_shared = {}
        self.table = []
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "est_cost_usd": 0.0, "calls": 0}

    # -- one LLM call
    def ask(self, model, system, user, mock_fn, max_tokens=1500):
        if self.mock:
            return mock_fn(), None
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=self.settings["temperature"],
                max_tokens=max_tokens,
                extra_headers={"X-OpenRouter-Title": "Inquiry Arena"},
            )
            u = getattr(resp, "usage", None)
            if u:
                pt, ct = (u.prompt_tokens or 0), (u.completion_tokens or 0)
                pin, pout = self.prices.get(model, (0.0, 0.0))
                self.usage["prompt_tokens"] += pt
                self.usage["completion_tokens"] += ct
                self.usage["est_cost_usd"] += pt * pin + ct * pout
            self.usage["calls"] += 1
            return (resp.choices[0].message.content or ""), None
        except Exception as e:
            return "", f"{type(e).__name__}: {str(e)[:300]}"

    # -- the dialogue
    def run(self, on_turn, on_status):
        n = len(self.agents)
        idx, passes_in_row, errors_in_row = 0, 0, 0
        for turn in range(1, self.settings["max_turns"] + 1):
            agent = self.agents[idx]
            name = agent["name"]
            on_status(f"Turn {turn}: {name} is thinking...")
            role = f", {agent['role']}" if agent.get("role") else ""
            others = ", ".join(x for x in self.names if x != name)
            system = RULES.format(name=name, role=role, n=n, others=others)
            user = (f"Question: {self.sc['question']}\nPossible answers:\n{options_text(self.sc)}\n\n"
                    f"Public facts (everyone sees these):\n{facts_text(self.sc.get('public_facts', {}))}\n\n"
                    f"Your private facts (only you see these):\n{facts_text(agent['facts'])}\n\n"
                    f"Conversation so far:\n{table_text(self.table, self.fact_text)}\n\n"
                    "Your move. Reply with one JSON object only.")
            raw, err = self.ask(self.models[name], system, user, lambda: self.mock_agent(agent))
            obj = parse_json(raw) if not err else None
            move = str(obj.get("move", "")).strip().lower() if obj else ""
            entry = {"turn": turn, "agent": name, "model": self.models[name], "raw": raw, "error": err}

            if move == "assert":
                cited = obj.get("facts") or []
                if isinstance(cited, str):
                    cited = [cited]
                cited = [str(x).strip() for x in cited]
                valid, new, invalid = [], [], []
                for fid in cited:
                    if fid in agent["facts"] or fid in self.on_table:
                        valid.append(fid)
                        if fid not in self.on_table:
                            new.append(fid)
                            self.on_table.add(fid)
                            self.first_shared[fid] = turn
                    else:
                        invalid.append(fid)
                entry.update(move="assert", facts=cited, valid_facts=valid, new_facts=new,
                             invalid_facts=invalid, claim=str(obj.get("claim", "")))
            elif move == "why":
                entry.update(move="why", to=str(obj.get("to", "")), question=str(obj.get("question", "")))
            else:
                entry.update(move="pass", unparsed=(not err and move != "pass"))

            self.table.append(entry)
            on_turn(entry)

            errors_in_row = errors_in_row + 1 if err else 0
            if errors_in_row >= n:
                return "stopped: every agent hit an API error in a row (check your key, credits or model names)"
            passes_in_row = passes_in_row + 1 if entry["move"] == "pass" else 0
            if passes_in_row >= n:
                return "everyone passed"

            lower = [x.lower() for x in self.names]
            target = entry.get("to", "").strip().lower()
            if entry["move"] == "why" and target in lower and target != name.lower():
                idx = lower.index(target)
            else:
                idx = (idx + 1) % n
        return "turn limit reached"

    # -- the judge
    def judge(self, pooled=False):
        if pooled:
            all_facts = dict(self.sc.get("public_facts", {}))
            for a in self.agents:
                all_facts.update(a["facts"])
            body = f"All known facts:\n{facts_text(all_facts)}"
        else:
            shared = {fid: self.fact_text[fid] for fid in sorted(self.on_table - self.public)}
            body = (f"Public facts:\n{facts_text(self.sc.get('public_facts', {}))}\n\n"
                    f"Facts shared during the discussion:\n{facts_text(shared)}\n\n"
                    f"The discussion:\n{table_text(self.table, self.fact_text)}")
        user = (f"Question: {self.sc['question']}\nPossible answers:\n{options_text(self.sc)}\n\n{body}\n\n"
                "Based only on the information above, which answer is best supported? Reply with ONLY: "
                '{"answer": "<letter>", "explanation": "<two or three sentences>", "facts_used": ["ID", ...]}')
        raw, err = self.ask(self.judge_model, JUDGE_SYSTEM, user, self.mock_judge)
        obj = parse_json(raw) or {}
        ans = str(obj.get("answer", "")).strip().upper()[:1]
        return {"model": self.judge_model, "answer": ans, "correct": ans == self.sc["answer"],
                "explanation": str(obj.get("explanation", "")), "facts_used": obj.get("facts_used", []),
                "raw": raw, "error": err}

    # -- the argument graph (judge extracts, the app labels)
    def extract_graph(self, support_rule):
        shared = {fid: self.fact_text[fid] for fid in sorted(self.on_table - self.public)}
        opts = "\n".join(f"opt_{k}: {v}" for k, v in self.sc["options"].items())
        user = (f"Question: {self.sc['question']}\nAnswer options (graph node IDs):\n{opts}\n\n"
                f"Public facts:\n{facts_text(self.sc.get('public_facts', {}))}\n\n"
                f"Facts shared during the discussion:\n{facts_text(shared)}\n\n"
                f"The discussion:\n{table_text(self.table, self.fact_text)}\n\n{GRAPH_INSTRUCTIONS}")
        raw, err = self.ask(self.judge_model, GRAPH_SYSTEM, user, self.mock_graph, max_tokens=4000)
        obj = parse_json(raw)
        g = build_graph(obj or {}, self.sc, support_rule)
        if not err and obj is None:
            g["warnings"].insert(0, "the judge's reply was not valid JSON, so the graph only has the options")
        g.update(model=self.judge_model, raw=raw, error=err)
        g["dot"] = graph_dot(g)
        return g

    # -- fake players for testing without spending money
    def mock_agent(self, agent):
        unshared = [f for f in agent["facts"] if f not in self.on_table]
        if unshared and random.random() < 0.7:
            fid = random.choice(unshared)
            return json.dumps({"move": "assert", "facts": [fid], "claim": "(mock) This might matter."})
        if random.random() < 0.15:
            other = random.choice([x for x in self.names if x != agent["name"]])
            return json.dumps({"move": "why", "to": other, "question": "(mock) Why do you think so?"})
        return json.dumps({"move": "pass"})

    def mock_judge(self):
        return json.dumps({"answer": random.choice(list(self.sc["options"])),
                           "explanation": "(mock judge, random answer)", "facts_used": []})

    def mock_graph(self):
        letters = list(self.sc["options"])
        args, attacks, supports = [], [], []
        for t in self.table:
            if t["move"] != "assert":
                continue
            aid = f"a{len(args) + 1}"
            args.append({"id": aid, "by": t["agent"], "turn": t["turn"], "facts": t["valid_facts"],
                         "text": f"(mock) {t['agent']} cites {', '.join(t['valid_facts']) or 'nothing'}"})
            r = random.random()
            if r < 0.5:
                supports.append([aid, f"opt_{random.choice(letters)}"])
            elif r < 0.8:
                attacks.append([aid, f"opt_{random.choice(letters)}"])
            elif len(args) > 1:
                attacks.append([aid, random.choice(args[:-1])["id"]])
        return json.dumps({"arguments": args, "attacks": attacks, "supports": supports})

    # -- numbers for the paper
    def metrics(self):
        private_shared = self.on_table - self.public
        key = set(self.sc.get("key_facts", [])) - self.public
        return {
            "turns": len(self.table),
            "private_facts_total": len(self.owner),
            "private_facts_shared": len(private_shared),
            "shared_by_agent": {a["name"]: [len(set(a["facts"]) & private_shared), len(a["facts"])]
                                for a in self.agents},
            "key_facts_total": len(key),
            "key_facts_shared": len(key & private_shared),
            "extra_facts_shared": len(private_shared - key) if key else None,
            "invalid_citations": sum(len(t.get("invalid_facts", [])) for t in self.table),
            "unparsed_replies": sum(1 for t in self.table if t.get("unparsed")),
            "api_errors": sum(1 for t in self.table if t.get("error")),
        }


# ----------------------------------------------------------------- display

def show_turn(entry, avatar, facts):
    with st.chat_message(entry["agent"], avatar=avatar):
        head = f"**{entry['agent']}** · `{entry['model']}` · turn {entry['turn']} · **{entry['move']}**"
        if entry["move"] == "why":
            head += f" → {entry.get('to') or 'group'}"
        st.markdown(head)
        if entry.get("error"):
            st.error(entry["error"])
        if entry["move"] == "assert":
            for fid in entry["valid_facts"]:
                tag = "new" if fid in entry["new_facts"] else "already on table"
                st.markdown(f"- `{fid}` ({tag}): {facts.get(fid, '')}")
            for fid in entry["invalid_facts"]:
                st.markdown(f"- :red[`{fid}`: invalid citation (not this agent's fact)]")
            if entry.get("claim"):
                st.markdown(f"> {entry['claim']}")
        elif entry["move"] == "why":
            st.markdown(f"> {entry.get('question', '')}")
        elif entry.get("unparsed"):
            st.caption("Reply was not valid JSON, counted as a pass.")
        with st.expander("raw model output"):
            st.code(entry.get("raw") or "(empty)", language=None)


def show_graph(g, sc, stamp):
    st.subheader("Argument graph")
    st.caption("The judge model drew the graph from the public record. The app, not the LLM, decides "
               "accepted / rejected with grounded semantics.")
    if g.get("error"):
        st.error(g["error"])
    n_args = sum(1 for n in g["nodes"] if n["kind"] == "argument")
    c1, c2, c3, c4 = st.columns(4)
    v = g["verdict"]
    c1.metric("Graph verdict (grounded)", v or "none",
              ("correct" if v == sc["answer"] else "wrong") if v else "no single answer accepted",
              delta_color="green" if v == sc["answer"] else ("red" if v else "gray"), delta_arrow="off")
    c2.metric("Arguments", n_args)
    c3.metric("Attacks", len(g["attacks"]))
    c4.metric("Supports", len(g["supports"]))
    st.graphviz_chart(g["dot"], width="stretch")
    rule = (" A support for one answer also counts as an attack on the other answers."
            if g["support_rule"] else " Supports are drawn but don't change the labels.")
    st.caption("Green = accepted (IN) · red = rejected (OUT) · grey = undecided. "
               "Solid red arrow = attack · dashed green arrow = support · "
               "dotted grey line = answers exclude each other." + rule)
    for w in g.get("warnings", []):
        st.warning(w)
    rows = [{"node": n["id"], "status": g["labels"][n["id"]], "by": n["by"], "turn": n["turn"],
             "facts": ", ".join(n["facts"]), "text": n["text"]} for n in g["nodes"]]
    st.dataframe(rows, width="stretch", hide_index=True)
    with st.expander("Raw graph output from the judge"):
        st.code(g.get("raw") or "(empty)", language=None)
    st.download_button("Download graph (.dot, opens in Graphviz)", g["dot"],
                       file_name=f"inquiry_graph_{stamp}.dot", mime="text/vnd.graphviz")


def show_results(run):
    sc = run["scenario"]
    avatar = {a["name"]: AVATARS[i % len(AVATARS)] for i, a in enumerate(sc["agents"])}
    facts = all_facts(sc)
    st.subheader("Transcript")
    st.caption(f"Ended because: {run['ended']}")
    for entry in run["transcript"]:
        show_turn(entry, avatar[entry["agent"]], facts)

    st.subheader("Verdict")
    m = run["metrics"]
    j = run["judge"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Judge (public record only)", j["answer"] or "?", verdict(j),
              delta_color="green" if j["correct"] else "red", delta_arrow="off")
    if run.get("pooled"):
        p = run["pooled"]
        c2.metric("Pooled baseline (all facts)", p["answer"] or "?", verdict(p),
                  delta_color="green" if p["correct"] else "red", delta_arrow="off")
    c3.metric("Gold answer", sc["answer"])
    c4.metric("Est. cost (USD)", f"{run['usage']['est_cost_usd']:.4f}")
    st.markdown(f"**Judge's explanation:** {j['explanation'] or '(none)'}")
    if j.get("error"):
        st.error(j["error"])
    if sc.get("gold_explanation"):
        st.markdown(f"**Gold explanation:** {sc['gold_explanation']}")

    if run.get("graph"):
        show_graph(run["graph"], sc, run["timestamp"].replace(":", "-"))

    st.subheader("Disclosure")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Turns", m["turns"])
    d2.metric("Private facts shared", f"{m['private_facts_shared']} / {m['private_facts_total']}")
    if m["key_facts_total"]:
        d3.metric("Key facts shared", f"{m['key_facts_shared']} / {m['key_facts_total']}")
    d4.metric("Invalid citations", m["invalid_citations"])
    rows = []
    for a in sc["agents"]:
        for fid, txt in a["facts"].items():
            rows.append({"fact": fid, "holder": a["name"], "key": fid in sc.get("key_facts", []),
                         "shared at turn": run["first_shared"].get(fid), "text": txt})
    st.dataframe(rows, width="stretch", hide_index=True)

    stamp = run["timestamp"].replace(":", "-")
    st.download_button("Download this run (JSON)", json.dumps(run, indent=2, ensure_ascii=False),
                       file_name=f"inquiry_run_{stamp}.json", mime="application/json")


# ----------------------------------------------------------------- page

password = secret("APP_PASSWORD")
if password and not st.session_state.get("authed"):
    st.title("🔎 Inquiry Arena")
    entered = st.text_input("Password", type="password")
    if entered == password:
        st.session_state.authed = True
        st.rerun()
    if entered:
        st.error("Wrong password.")
    st.stop()

with st.sidebar:
    st.header("Setup")
    api_key = secret("OPENROUTER_API_KEY")
    if api_key:
        st.success("Using the app's OpenRouter key.")
    else:
        api_key = st.text_input("Your OpenRouter API key", type="password",
                                help="Not stored. Get one at openrouter.ai.")
    mock = st.toggle("Mock mode (fake players, no API calls, free)", value=not api_key)

    builtin = load_builtin_scenarios()
    uploaded = st.file_uploader("Upload a scenario (.json)", type="json")
    scenarios = dict(builtin)
    if uploaded is not None:
        try:
            up = json.load(uploaded)
            scenarios = {f"(uploaded) {up.get('title', uploaded.name)}": up, **scenarios}
        except Exception as e:
            st.error(f"Could not read that file: {e}")
    if not scenarios:
        st.warning("No scenarios found. Upload one.")
        st.stop()
    chosen = st.selectbox("Scenario", list(scenarios))
    with st.expander("Scenario file format"):
        st.code(FORMAT_HELP, language="json")

    max_turns = st.slider("Max turns", 4, 60, 20)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    run_pooled = st.checkbox("Also run pooled baseline (judge sees all facts)", value=True)
    build_arg_graph = st.checkbox("Build argument graph at the end", value=True,
                                  help="One more call to the judge model.")
    support_rule = st.checkbox("Support for one answer counts as attack on the others", value=True,
                               disabled=not build_arg_graph,
                               help="Answers are mutually exclusive, so arguing for B argues against A and C. "
                                    "Turn off to label the graph with attacks only.")

sc = scenarios[chosen]
problems = validate(sc)
if problems:
    st.error("This scenario has problems: " + "; ".join(problems))
    st.stop()

st.title("🔎 Inquiry Arena")
st.caption("LLM agents with split knowledge try to answer a question together. "
           "Only the referee sees all facts.")

st.markdown(f"### {sc.get('title', 'Scenario')}")
st.markdown(f"**Question:** {sc['question']}")
st.markdown("\n".join(f"- **{k}.** {v}" for k, v in sc["options"].items()))
if sc.get("public_facts"):
    st.markdown("**Public facts:** " + " ".join(f"`{k}` {v}" for k, v in sc["public_facts"].items()))

models_available = fetch_models()
model_ids = list(models_available)
if not mock and not model_ids:
    st.warning("Could not load the model list from OpenRouter. Type model names by hand.")


def default_model(i):
    present = [m for m in PREFERRED_MODELS if m in models_available]
    if present:
        return present[i % len(present)]
    return model_ids[i % len(model_ids)] if model_ids else FALLBACK_MODEL


def model_picker(label, key, i):
    if model_ids:
        default = default_model(i)
        return st.selectbox(label, model_ids, index=model_ids.index(default), key=key,
                            format_func=lambda m: price_label(m, models_available))
    return st.text_input(label, value=FALLBACK_MODEL, key=key)


st.markdown("#### Players")
cols = st.columns(len(sc["agents"]))
models = {}
for i, (col, a) in enumerate(zip(cols, sc["agents"])):
    with col:
        st.markdown(f"{AVATARS[i % len(AVATARS)]} **{a['name']}**" + (f", {a['role']}" if a.get("role") else ""))
        models[a["name"]] = model_picker("Model", f"model_{chosen}_{a['name']}", i)
        with st.expander(f"Private facts ({len(a['facts'])})"):
            st.markdown("\n".join(f"- `{k}` {v}" for k, v in a["facts"].items()))
judge_model = model_picker("Judge model (reads only the public record)", f"judge_{chosen}", 0)
with st.expander("Show gold answer"):
    st.markdown(f"**{sc['answer']}**. {sc['options'][sc['answer']]}")
    if sc.get("gold_explanation"):
        st.markdown(sc["gold_explanation"])

if st.button("▶ Run dialogue", type="primary", width="stretch"):
    if not mock and not api_key:
        st.error("Add an OpenRouter key in the sidebar, or switch on mock mode.")
        st.stop()
    settings = {"max_turns": max_turns, "temperature": temperature, "mock": mock,
                "argument_graph": build_arg_graph, "support_rule": support_rule}
    ref = Referee(sc, models, judge_model, settings, api_key, models_available)
    avatar = {a["name"]: AVATARS[i % len(AVATARS)] for i, a in enumerate(sc["agents"])}
    status = st.empty()
    live = st.container()

    facts = all_facts(sc)

    def on_turn(entry):
        with live:
            show_turn(entry, avatar[entry["agent"]], facts)

    ended = ref.run(on_turn, lambda s: status.info(s))
    status.info("The judge is reading the public record...")
    judge = ref.judge(pooled=False)
    pooled = None
    if run_pooled:
        status.info("Running the pooled baseline...")
        pooled = ref.judge(pooled=True)
    graph = None
    if build_arg_graph:
        status.info("The judge is drawing the argument graph...")
        graph = ref.extract_graph(support_rule)
    status.empty()
    st.session_state.last_run = {
        "app": "inquiry-arena v0.2",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenario": sc,
        "models": models,
        "judge_model": judge_model,
        "settings": settings,
        "ended": ended,
        "transcript": ref.table,
        "first_shared": ref.first_shared,
        "judge": judge,
        "pooled": pooled,
        "graph": graph,
        "metrics": ref.metrics(),
        "usage": ref.usage,
    }
    st.rerun()

if "last_run" in st.session_state:
    st.divider()
    show_results(st.session_state.last_run)
