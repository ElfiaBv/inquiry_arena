"""Inquiry Arena: LLM agents with split knowledge play an inquiry dialogue.

The referee (this program) is the only one that sees every fact. Each LLM sees the
public facts, its own private facts, its own private notes, and the public record.

Moves
  assert   one fact, aimed at an earlier argument or at an answer, as attack or support,
           plus a short spoken line. The agent also writes a PRIVATE reconstruction:
           how it reads the argument it targets, which part it aims at, and its own
           full argument (premises, unstated assumption, conclusion). Nobody else sees
           it unless someone asks.
  show     reveal the recorded reconstruction behind an earlier argument, e.g. when an
           attack on you makes no sense to you. The record is shown exactly as it was
           written at the time, so it cannot be rewritten afterwards.
  retract  withdraw one of your own earlier arguments.
  ask      post an open goal: a statement the group still needs to establish.
  pass
An assert can also say which open goal it answers, and open new goals (the premises it
still needs). Goals live in a public list that every agent sees.
The dialogue ends when every agent passes in a row, or at the turn limit.

Strategies (v0.5): every agent gets the same moves, information, turn limit and stopping
rule. Only the strategy paragraph at the end of its instructions changes:
free inquiry, goal-directed explanation building, assumption checking and repair,
or contrastive testing.

Afterwards: a judge answers from the public record only; the app builds the argument
graph from the targets the agents declared and labels it with grounded semantics;
optionally the judge checks each reading against what the original speaker meant.
"""

import glob
import json
import os
import random
import re
import textwrap
from datetime import datetime, timezone

import requests
import streamlit as st

st.set_page_config(page_title="Inquiry Arena", page_icon="🔎", layout="wide")

APP_VERSION = "inquiry-arena v0.5"
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
The other agents are: {others}.

How it works:
- Everyone sees the public facts and the public conversation.
- Each agent also has PRIVATE facts. Others only learn your facts if you state them.
- People talk briefly: they state one fact and say what it is for, without spelling out every assumption.
  So each time you make an argument you ALSO write a private reconstruction. Nobody else sees it
  unless someone asks for it:
  * reading_of_target: how you understand the argument you are responding to: its premises,
    the unstated assumption it relies on, and its conclusion (null if you target an answer directly);
  * aim: which part of the target you attack or support: "premise", "assumption" or "conclusion";
  * mine: your own argument in full: premises, unstated assumption, conclusion.
- Open goals: a public list of statements the group still needs to establish (G1, G2, ...).
  Anyone can post one. An argument can say which open goal it answers, and can open new goals.
- Be honest: use only your own facts or facts already on the table. Never invent facts or IDs.

On each turn reply with ONLY one flat JSON object (do not nest it inside another object), one of these:

1. Make an argument with exactly ONE fact:
{{"move": "assert", "fact": "ID", "target": <turn number or answer letter>, "relation": "attack" or "support",
  "say": "what you say out loud, one or two sentences",
  "answers_goal": "G<number>" or null,
  "opens": ["a statement this argument still needs, not yet established", ...] or [],
  "reconstruction": {{"reading_of_target": {{"premises": "...", "assumption": "...", "conclusion": "..."}},
                      "aim": "premise" or "assumption" or "conclusion",
                      "mine": {{"premises": "...", "assumption": "...", "conclusion": "..."}}}}}}

2. Post an open goal, without stating a fact:
{{"move": "ask", "goal": "the statement the group needs to establish", "for": <answer letter, turn number or null>,
  "say": "what you say out loud"}}

3. Reveal the private reconstruction behind an earlier argument, yours or someone else's:
{{"move": "show", "turn": <turn number>, "reason": "..."}}

4. Withdraw one of your own earlier arguments:
{{"move": "retract", "turn": <turn number>, "reason": "..."}}

5. Nothing to add right now:
{{"move": "pass"}}

The dialogue ends when every agent passes in a row.

YOUR STRATEGY
{strategy}"""

# Every agent gets the same rules above; only this paragraph changes between strategies.
STRATEGIES = {
    "free": {
        "label": "1. Free inquiry",
        "question": "What would help?",
        "caption": "Any useful move, in any order. The baseline.",
        "text": (
            "Free inquiry. Choose whatever contribution seems most useful at this point. You may share\n"
            "evidence, post a goal, challenge, ask to see a reconstruction, withdraw, or pass. No kind of\n"
            "move is preferred over another."),
    },
    "goals": {
        "label": "2. Goal-directed explanation building",
        "question": "What do we still need to establish?",
        "caption": "Answer open goals first; otherwise build an explanation and open its missing premises.",
        "text": (
            "Goal-directed explanation building. Ask yourself: what do we still need to establish?\n"
            "- First, if one of your facts answers an open goal, state it and set answers_goal.\n"
            "- Otherwise, develop a candidate explanation: state one fact or rule towards an answer, say which\n"
            "  answer it serves, and list in \"opens\" the premises it still needs that are not yet established.\n"
            "  Only open goals that are premises of the fact you state, not things you merely suspect.\n"
            "- Do not state facts that neither answer an open goal nor advance a candidate explanation.\n"
            "- If nothing fits, pass.\n"
            "Example of the form: \"X happens whenever Y and Z. This could explain answer B. Still needed: Y; Z.\""),
    },
    "assumptions": {
        "label": "3. Assumption checking and repair",
        "question": "Does that argument actually follow?",
        "caption": "Check what earlier arguments take for granted; supply, challenge or ask.",
        "text": (
            "Assumption checking and repair. Ask yourself: does that argument actually follow?\n"
            "- Look at the arguments already made, starting with those the currently favoured answer rests on,\n"
            "  for an unstated premise or bridge they rely on.\n"
            "- If you hold a fact that establishes it, state it in support of that argument. If you hold a fact\n"
            "  against it, attack the argument. If you cannot tell what an argument means, use show on it.\n"
            "  If the premise is unknown to you, post it with ask instead of accepting it silently.\n"
            "- If one of your own arguments rested on a misreading, withdraw it.\n"
            "- If no argument needs checking, make a new argument backed by one of your facts. If nothing fits, pass."),
    },
    "contrast": {
        "label": "4. Contrastive testing",
        "question": "What distinguishes the candidate answers?",
        "caption": "Prefer moves that separate competing answers, including against the favourite.",
        "text": (
            "Contrastive testing. Ask yourself: what distinguishes the candidate answers?\n"
            "- Prefer a fact or a question that separates competing answers, including evidence against the\n"
            "  answer that currently looks best.\n"
            "- Choose such a test over another argument for an answer that is already well supported.\n"
            "- Say which answers your move separates. If no test is available, develop an answer that is still\n"
            "  unresolved.\n"
            "- Lack of support for an answer does not rule it out; rule an answer out only with a fact against it.\n"
            "- If nothing fits, pass."),
    },
}

FORMAT_REMINDER = """

IMPORTANT: your last reply could not be read. Reply with exactly one flat JSON object and nothing else,
for example: {"move": "pass"}, or {"move": "assert", "fact": "ID", "target": "A", "relation": "support",
"say": "...", "answers_goal": null, "opens": [], "reconstruction": {"reading_of_target": null, "aim": "conclusion",
"mine": {"premises": "...", "assumption": "...", "conclusion": "..."}}}"""

MOVE_ALIASES = {"assert": "assert", "show": "show", "show_reconstruction": "show", "reveal": "show",
                "retract": "retract", "withdraw": "retract", "ask": "ask", "open_goal": "ask",
                "post_goal": "ask", "goal": "ask", "pass": "pass"}

JUDGE_SYSTEM = ("You are a neutral judge. You know only what is written in the message. "
                "Reply with ONLY one JSON object, nothing else.")
CHECK_SYSTEM = ("You compare reconstructions of short arguments. You know only what is written in the message. "
                "Reply with ONLY one JSON object, nothing else.")

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
  "gold_explanation": "What a good explanation says",
  "notes": "What the scenario is designed to test"
}
Fact IDs must be unique. key_facts, gold_explanation and notes are optional."""


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


def read_reply(raw):
    """Parse an agent reply. Accepts the flat format and the common nested variant
    {"move": {"move": "assert", ...}} or {"move": [{...}]}. Returns (obj, move_name)."""
    obj = parse_json(raw)
    if not isinstance(obj, dict):
        return None, ""
    inner = obj.get("move")
    if isinstance(inner, list) and inner and isinstance(inner[0], dict):
        inner = inner[0]
    if isinstance(inner, dict):
        merged = {k: v for k, v in obj.items() if k != "move"}
        merged.update(inner)
        merged["move"] = inner.get("move") or inner.get("type") or inner.get("action") or ""
        obj = merged
    name = _s(obj.get("move") or obj.get("action") or obj.get("type")).lower()
    return obj, MOVE_ALIASES.get(name, name)


def _s(v):
    if isinstance(v, list):
        return "; ".join(str(i) for i in v)
    return "" if v is None else str(v).strip()


def parse_int(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return int(x)
    m = re.search(r"\d+", str(x or ""))
    return int(m.group()) if m else None


def parse_target(raw, letters):
    """Return ("turn", n), ("option", letter) or (None, None)."""
    if raw is None or isinstance(raw, bool):
        return None, None
    if isinstance(raw, (int, float)):
        return "turn", int(raw)
    s = str(raw).strip()
    m = re.fullmatch(r"(?:turn\s*)?#?(\d+)", s, re.I)
    if m:
        return "turn", int(m.group(1))
    up = s.upper()
    for k in letters:
        K = k.upper()
        if up in (K, f"OPT_{K}", f"OPTION {K}", f"OPTION_{K}", f"ANSWER {K}"):
            return "option", k
    return None, None


def parse_relation(raw):
    r = _s(raw).lower()
    if r.startswith("att") or r in ("rebut", "undercut", "undermine"):
        return "attack"
    if r.startswith("sup"):
        return "support"
    return ""


def norm_part(x):
    if isinstance(x, dict):
        return {k: _s(x.get(k)) for k in ("premises", "assumption", "conclusion")}
    if x is None or _s(x).lower() in ("", "null", "none", "n/a"):
        return None
    return {"premises": "", "assumption": "", "conclusion": _s(x)}


def norm_recon(r):
    r = r if isinstance(r, dict) else {}
    aim = _s(r.get("aim")).lower()
    aim = next((a for a in ("premise", "assumption", "conclusion") if aim.startswith(a[:5])), aim)
    return {"reading_of_target": norm_part(r.get("reading_of_target")), "aim": aim,
            "mine": norm_part(r.get("mine"))}


def part_text(p):
    if not p:
        return "(none)"
    bits = []
    if p.get("premises"):
        bits.append(f"premises: {p['premises']}")
    if p.get("assumption"):
        bits.append(f"unstated assumption: {p['assumption']}")
    if p.get("conclusion"):
        bits.append(f"conclusion: {p['conclusion']}")
    return "; ".join(bits) or "(empty)"


def recon_text(rec):
    if not rec:
        return "(no reconstruction recorded)"
    s = []
    if rec.get("reading_of_target"):
        s.append(f"reads the target as [{part_text(rec['reading_of_target'])}]")
    if rec.get("aim"):
        s.append(f"aims at its {rec['aim']}")
    s.append(f"own argument [{part_text(rec.get('mine'))}]")
    return "; ".join(s)


def describe_target(t):
    verb = {"attack": "attacks", "support": "supports"}.get(t.get("relation"), "responds to")
    if t.get("target_kind") == "option":
        return f"{verb} answer {t['target']}"
    if t.get("target_kind") == "turn":
        return f"{verb} turn {t['target']}"
    return "no clear target"


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
    """Clean up a graph and compute labels. Answer options exclude each other."""
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
        for k in letters:
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
                continue
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
            lines.append(f'  "{n["id"]}" [label="{_dot_text(n["text"], 28)}", fillcolor="{fill}", '
                         f'color="{border}", penwidth=3, fontsize=12];')
            opts.append(f'"{n["id"]}"')
        else:
            head = f'Turn {n["turn"]} · {n["by"]}' if n["turn"] else f'{n["id"]} · {n["by"]}'
            facts = f'\\n[{", ".join(n["facts"])}]' if n["facts"] else ""
            lines.append(f'  "{n["id"]}" [label="{_dot_text(head, 60)}\\n{_dot_text(n["text"])}{facts}", '
                         f'fillcolor="{fill}", color="{border}"];')
    if opts:
        lines.append(f'  {{ rank=same; {"; ".join(opts)}; }}')
        if len(opts) > 1:  # keep options in A, B, C order
            lines.append(f'  {" -> ".join(opts)} [style=invis];')
    misread = {tuple(e) for e in g.get("misread", [])}
    orange = '#c2410c'
    for x, y in g["attacks"]:
        if (x, y) in misread:
            lines.append(f'  "{x}" -> "{y}" [color="{orange}", fontcolor="{orange}", style=dashed, '
                         f'penwidth=1.8, label=" misread"];')
        else:
            lines.append(f'  "{x}" -> "{y}" [color="#b42318", penwidth=1.6];')
    for x, y in g["supports"]:
        if (x, y) in misread:
            lines.append(f'  "{x}" -> "{y}" [color="{orange}", fontcolor="{orange}", style=dashed, '
                         f'arrowhead=empty, label=" misread"];')
        else:
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
        self.letters = list(sc["options"])
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
        self.goals = []  # public list of open goals
        self.strategy = settings.get("strategy", "free")
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "est_cost_usd": 0.0, "calls": 0}

    # -- open goals
    def goal(self, gid):
        gid = _s(gid).upper().replace(" ", "")
        if gid and not gid.startswith("G"):
            gid = "G" + gid
        return next((g for g in self.goals if g["id"] == gid), None)

    def add_goal(self, text, by, turn, via, for_kind=None, for_val=None):
        text = _s(text)
        if not text:
            return None
        g = {"id": f"G{len(self.goals) + 1}", "text": text[:300], "by": by, "turn": turn, "via": via,
             "for_kind": for_kind, "for": for_val, "status": "open",
             "answered_at": None, "answered_by": None, "answer_fact": None}
        self.goals.append(g)
        return g["id"]

    def goal_line(self, g):
        s = f'{g["id"]} (posted by {g["by"]} at turn {g["turn"]}'
        if self.settings["show_targets"] and g.get("for") is not None:
            s += f", for {'answer' if g['for_kind'] == 'option' else 'turn'} {g['for']}"
        s += f'): "{g["text"]}"'
        if g["status"] == "answered":
            s += f" - answered by {g['answered_by']} at turn {g['answered_at']} with {g['answer_fact']}"
        else:
            s += " - open"
        return s

    def goals_text(self):
        return "\n".join(self.goal_line(g) for g in self.goals) or "(none yet)"

    def entry_at(self, t):
        if isinstance(t, int) and 1 <= t <= len(self.table):
            return self.table[t - 1]
        return None

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

    # -- what everyone can see
    def public_line(self, t):
        who = t["agent"]
        if t["move"] == "assert":
            s = f"Turn {t['turn']} - {who}"
            if self.settings["show_targets"]:
                s += f" ({describe_target(t)})"
            if t["fact_status"] in ("new", "on table"):
                s += f' states {t["fact"]}: "{self.fact_text[t["fact"]]}"'
            elif t["fact_status"] == "invalid":
                s += f' cites "{t["fact"]}", which is not a fact they hold'
            s += f'. Says: "{t["say"]}"' if t["say"] else "."
            if t.get("answers_goal"):
                s += f" Answers {t['answers_goal']}."
            if t.get("opened"):
                s += " Opens " + "; ".join(f'{gid}: "{self.goal(gid)["text"]}"' for gid in t["opened"]) + "."
            if t.get("retracted_at"):
                s += f" [withdrawn by {who} at turn {t['retracted_at']}]"
            return s
        if t["move"] == "ask":
            if t["problem"]:
                return f"Turn {t['turn']} - {who} tried to post a goal, but {t['problem']}."
            g = self.goal(t["goal_id"])
            s = f'Turn {t["turn"]} - {who} posts open goal {g["id"]}: "{g["text"]}"'
            if self.settings["show_targets"] and g.get("for") is not None:
                s += f" (for {'answer' if g['for_kind'] == 'option' else 'turn'} {g['for']})"
            s += f'. Says: "{t["say"]}"' if t["say"] else "."
            return s
        if t["move"] == "unreadable":
            return f"Turn {t['turn']} - {who}'s reply could not be read (no move)."
        if t["move"] == "show":
            if t["problem"]:
                return (f"Turn {t['turn']} - {who} asked to see the reconstruction behind turn "
                        f"{t['turn_requested']}, but {t['problem']}.")
            reason = f' ("{t["reason"]}")' if t["reason"] else ""
            return (f"Turn {t['turn']} - {who} asks to see the private reconstruction behind turn "
                    f"{t['turn_requested']}{reason}. Revealed record of {t['owner']}: {recon_text(t['revealed'])}.")
        if t["move"] == "retract":
            if t["problem"]:
                return f"Turn {t['turn']} - {who} tried to withdraw turn {t['turn_requested']}, but {t['problem']}."
            reason = f' ("{t["reason"]}")' if t["reason"] else ""
            return f"Turn {t['turn']} - {who} withdraws turn {t['turn_requested']}{reason}."
        return f"Turn {t['turn']} - {who} passes."

    def public_record(self):
        return "\n".join(self.public_line(t) for t in self.table) or "(nothing yet - you speak first)"

    def agent_prompt(self, agent):
        notes = [f"Turn {t['turn']} (you {describe_target(t).replace('attacks', 'attacked').replace('supports', 'supported')}"
                 f" with {t['fact'] or 'no fact'}): {recon_text(t['reconstruction'])}"
                 + (f" [you withdrew it at turn {t['retracted_at']}]" if t.get("retracted_at") else "")
                 for t in self.table if t["move"] == "assert" and t["agent"] == agent["name"]]
        return (f"Question: {self.sc['question']}\nPossible answers:\n{options_text(self.sc)}\n\n"
                f"Public facts (everyone sees these):\n{facts_text(self.sc.get('public_facts', {}))}\n\n"
                f"Your private facts (only you see these):\n{facts_text(agent['facts'])}\n\n"
                f"Your private notes (your earlier reconstructions; only you see these):\n"
                f"{chr(10).join(notes) or '(none yet)'}\n\n"
                f"Public conversation so far:\n{self.public_record()}\n\n"
                f"Open goals (public):\n{self.goals_text()}\n\n"
                "Your move. Reply with one JSON object only.")

    # -- the moves
    def do_assert(self, entry, obj, agent):
        fact = obj.get("fact", obj.get("facts"))
        extra = []
        if isinstance(fact, list):
            extra = [_s(x) for x in fact[1:]]
            fact = fact[0] if fact else ""
        fact = _s(fact)
        if not fact:
            status = "missing"
        elif fact in agent["facts"] and fact not in self.on_table:
            status = "new"
            self.on_table.add(fact)
            self.first_shared[fact] = entry["turn"]
        elif fact in self.on_table:
            status = "on table"
        else:
            status = "invalid"
        kind, val = parse_target(obj.get("target"), self.letters)
        problems = []
        if kind == "turn":
            tgt = self.entry_at(val)
            if tgt is None:
                problems.append(f"turn {val} doesn't exist yet")
                kind = val = None
            elif tgt["move"] != "assert":
                problems.append(f"turn {val} is not an argument")
                kind = val = None
            elif tgt.get("retracted_at"):
                problems.append(f"turn {val} was already withdrawn")
        elif kind is None:
            problems.append(f"unclear target {obj.get('target')!r}")
        rel = parse_relation(obj.get("relation"))
        if not rel:
            problems.append("no attack/support given")
        # open goals: which goal this answers, and which new goals it opens
        answers = None
        raw_goal = obj.get("answers_goal")
        if raw_goal not in (None, "", False) and _s(raw_goal).lower() not in ("null", "none"):
            g = self.goal(raw_goal)
            if g is None:
                problems.append(f"unknown goal {raw_goal!r}")
            elif g["status"] == "answered":
                problems.append(f"goal {g['id']} was already answered")
            elif status in ("new", "on table"):
                g.update(status="answered", answered_at=entry["turn"], answered_by=entry["agent"], answer_fact=fact)
                answers = g["id"]
        opens = obj.get("opens") or []
        if isinstance(opens, str):
            opens = [opens]
        opened = []
        if isinstance(opens, list):
            for o in opens[:3]:  # at most three new goals per argument
                gid = self.add_goal(o, entry["agent"], entry["turn"], "assert", kind, val)
                if gid:
                    opened.append(gid)
        entry.update(move="assert", fact=fact, fact_status=status, extra_facts=extra,
                     target_kind=kind, target=val, relation=rel,
                     say=_s(obj.get("say") or obj.get("claim")),
                     answers_goal=answers, opened=opened,
                     opened_texts=[self.goal(g)["text"] for g in opened],
                     reconstruction=norm_recon(obj.get("reconstruction")),
                     revealed_at=None, retracted_at=None, problem="; ".join(problems) or None)

    def do_ask(self, entry, obj):
        text = obj.get("goal") or obj.get("statement") or obj.get("question") or ""
        kind, val = parse_target(obj.get("for", obj.get("target")), self.letters)
        problem = None
        if kind == "turn" and self.entry_at(val) is None:
            kind = val = None
        gid = self.add_goal(text, entry["agent"], entry["turn"], "ask", kind, val)
        if gid is None:
            problem = "no goal was given"
        entry.update(move="ask", goal_id=gid, goal_text=_s(text)[:300], say=_s(obj.get("say")), problem=problem)

    def do_show(self, entry, obj):
        t = parse_int(obj.get("turn", obj.get("target")))
        tgt = self.entry_at(t)
        problem, owner, revealed = None, None, None
        if tgt is None:
            problem = f"turn {t} doesn't exist"
        elif tgt["move"] != "assert":
            problem = f"turn {t} is not an argument"
        else:
            owner = tgt["agent"]
            revealed = dict(tgt["reconstruction"])
            if tgt.get("revealed_at") is None:
                tgt["revealed_at"] = entry["turn"]
        entry.update(move="show", turn_requested=t, owner=owner, reason=_s(obj.get("reason")),
                     problem=problem, revealed=revealed)
        return owner

    def do_retract(self, entry, obj):
        t = parse_int(obj.get("turn", obj.get("target")))
        tgt = self.entry_at(t)
        problem = None
        if tgt is None:
            problem = f"turn {t} doesn't exist"
        elif tgt["move"] != "assert":
            problem = f"turn {t} is not an argument"
        elif tgt["agent"] != entry["agent"]:
            problem = f"turn {t} is not their own argument"
        elif tgt.get("retracted_at"):
            problem = f"turn {t} was already withdrawn"
        else:
            tgt["retracted_at"] = entry["turn"]
            if tgt.get("answers_goal"):  # a withdrawn answer re-opens its goal
                g = self.goal(tgt["answers_goal"])
                if g and g["answered_at"] == tgt["turn"]:
                    g.update(status="open", answered_at=None, answered_by=None, answer_fact=None)
        entry.update(move="retract", turn_requested=t, reason=_s(obj.get("reason")), problem=problem)

    # -- the dialogue
    def system_prompt(self, agent):
        n = len(self.agents)
        role = f", {agent['role']}" if agent.get("role") else ""
        others = ", ".join(x for x in self.names if x != agent["name"])
        return RULES.format(name=agent["name"], role=role, n=n, others=others,
                            strategy=STRATEGIES[self.strategy]["text"])

    def run(self, on_turn, on_status):
        n = len(self.agents)
        idx, passes_in_row, problems_in_row = 0, 0, 0
        for turn in range(1, self.settings["max_turns"] + 1):
            agent = self.agents[idx]
            name = agent["name"]
            on_status(f"Turn {turn}: {name} is thinking...")
            system = self.system_prompt(agent)
            prompt = self.agent_prompt(agent)
            raw, err = self.ask(self.models[name], system, prompt, lambda: self.mock_agent(agent))
            obj, move = read_reply(raw) if not err else (None, "")
            retried = False
            if not err and move not in MOVE_ALIASES.values():  # unreadable: ask once more
                retried = True
                raw2, err = self.ask(self.models[name], system, prompt + FORMAT_REMINDER,
                                     lambda: self.mock_agent(agent, retry=True))
                raw = raw + "\n\n--- retry ---\n" + raw2
                obj, move = read_reply(raw2) if not err else (None, "")
            entry = {"turn": turn, "agent": name, "model": self.models[name], "raw": raw,
                     "error": err, "retried": retried}
            next_idx = None
            if move == "assert":
                self.do_assert(entry, obj, agent)
            elif move == "show":
                owner = self.do_show(entry, obj)
                if owner and owner != name:  # the owner of the revealed record speaks next
                    next_idx = self.names.index(owner)
            elif move == "retract":
                self.do_retract(entry, obj)
            elif move == "ask":
                self.do_ask(entry, obj)
            elif move == "pass":
                entry.update(move="pass")
            else:  # an API error or a reply that could not be read: not counted as a pass
                entry.update(move="unreadable", problem=err or "the reply could not be read")
            self.table.append(entry)
            on_turn(entry)

            problems_in_row = problems_in_row + 1 if entry["move"] == "unreadable" else 0
            if problems_in_row >= 2 * n:
                return ("stopped: too many API errors or unreadable replies in a row "
                        "(check your key, credits or model names)")
            if entry["move"] == "pass":
                passes_in_row += 1
            elif entry["move"] != "unreadable":
                passes_in_row = 0
            if passes_in_row >= n:
                return "everyone passed"
            idx = next_idx if next_idx is not None else (idx + 1) % n
        return "turn limit reached"

    # -- the judge answers from the public record only
    def judge(self, pooled=False):
        if pooled:
            all_f = dict(self.sc.get("public_facts", {}))
            for a in self.agents:
                all_f.update(a["facts"])
            body = f"All known facts:\n{facts_text(all_f)}"
        else:
            shared = {fid: self.fact_text[fid] for fid in sorted(self.on_table - self.public)}
            body = (f"Public facts:\n{facts_text(self.sc.get('public_facts', {}))}\n\n"
                    f"Facts stated during the discussion:\n{facts_text(shared)}\n\n"
                    f"The discussion:\n{self.public_record()}\n\n"
                    f"Open goals at the end of the discussion:\n{self.goals_text()}")
        user = (f"Question: {self.sc['question']}\nPossible answers:\n{options_text(self.sc)}\n\n{body}\n\n"
                "Based only on the information above, which answer is best supported? Reply with ONLY: "
                '{"answer": "<letter>", "explanation": "<two or three sentences>", "facts_used": ["ID", ...]}')
        raw, err = self.ask(self.judge_model, JUDGE_SYSTEM, user, self.mock_judge)
        obj = parse_json(raw) or {}
        ans = str(obj.get("answer", "")).strip().upper()[:1]
        return {"model": self.judge_model, "answer": ans, "correct": ans == self.sc["answer"],
                "explanation": str(obj.get("explanation", "")), "facts_used": obj.get("facts_used", []),
                "raw": raw, "error": err}

    # -- did responders read arguments the way their speakers meant them?
    def reading_pairs(self):
        pairs = []
        for t in self.table:
            if (t["move"] == "assert" and t["target_kind"] == "turn"
                    and t["reconstruction"]["reading_of_target"]):
                o = self.entry_at(t["target"])
                if o and o["agent"] != t["agent"]:
                    pairs.append((t, o))
        return pairs

    def check_readings(self):
        pairs = self.reading_pairs()
        if not pairs:
            return {"pairs": 0, "error": None, "raw": ""}
        blocks = []
        for t, o in pairs:
            blocks.append(
                f"Pair {t['turn']}->{o['turn']}\n"
                f"  What {o['agent']} said at turn {o['turn']}: fact {o['fact']} "
                f"(\"{self.fact_text.get(o['fact'], '?')}\"), \"{o['say']}\"\n"
                f"  What {o['agent']} privately meant: {part_text(o['reconstruction']['mine'])}\n"
                f"  How {t['agent']} read it at turn {t['turn']}: "
                f"{part_text(t['reconstruction']['reading_of_target'])}")
        user = ("For each pair, compare how a responder read an argument with what the argument's speaker "
                "privately meant. Focus on the unstated assumption and the conclusion. Rate each pair:\n"
                '"match": the same argument;\n'
                '"partial": the same conclusion, but a different assumption or premises;\n'
                '"mismatch": a different argument.\n\n' + "\n\n".join(blocks) +
                '\n\nReply with ONLY: {"pairs": [{"pair": "4->3", "rating": "match", "note": "one short sentence"}]}')
        raw, err = self.ask(self.judge_model, CHECK_SYSTEM, user, lambda: self.mock_check(pairs),
                            max_tokens=3000)
        got = {}
        for p in (parse_json(raw) or {}).get("pairs") or []:
            if isinstance(p, dict):
                got[re.sub(r"\s+", "", _s(p.get("pair")))] = p
        for t, o in pairs:
            p = got.get(f"{t['turn']}->{o['turn']}")
            if p:
                r = _s(p.get("rating")).lower()
                t["reading_check"] = {"rating": r if r in ("match", "partial", "mismatch") else "unclear",
                                      "note": _s(p.get("note"))}
            else:
                t["reading_check"] = {"rating": "unrated", "note": ""}
        return {"pairs": len(pairs), "error": err, "raw": raw}

    # -- the argument graph, straight from the declared moves
    def declared_graph(self, support_rule):
        args, attacks, supports, misread, notes, live = [], [], [], [], [], {}
        for t in self.table:
            if t["move"] != "assert":
                continue
            if t.get("retracted_at"):
                notes.append(f"Turn {t['turn']} left out: withdrawn at turn {t['retracted_at']}.")
                continue
            if t["fact_status"] in ("invalid", "missing"):
                notes.append(f"Turn {t['turn']} left out: no valid fact.")
                continue
            live[t["turn"]] = f"t{t['turn']}"
            args.append({"id": f"t{t['turn']}", "by": t["agent"], "turn": t["turn"], "facts": [t["fact"]],
                         "text": t["say"] or self.fact_text.get(t["fact"], "")})
        for t in self.table:
            if t["move"] != "assert" or t["turn"] not in live or not t["relation"]:
                continue
            if t["target_kind"] == "option":
                dst = f"opt_{t['target']}"
            elif t["target_kind"] == "turn" and t["target"] in live:
                dst = live[t["target"]]
            else:
                continue
            edge = [live[t["turn"]], dst]
            (attacks if t["relation"] == "attack" else supports).append(edge)
            if (t.get("reading_check") or {}).get("rating") == "mismatch":
                misread.append(edge)
        g = build_graph({"arguments": args, "attacks": attacks, "supports": supports}, self.sc, support_rule)
        g["misread"], g["notes"] = misread, notes
        if misread:
            g2 = build_graph({"arguments": args,
                              "attacks": [e for e in attacks if e not in misread],
                              "supports": [e for e in supports if e not in misread]}, self.sc, support_rule)
            g["labels_without_misreads"], g["verdict_without_misreads"] = g2["labels"], g2["verdict"]
        else:
            g["labels_without_misreads"], g["verdict_without_misreads"] = None, g["verdict"]
        g["dot"] = graph_dot(g)
        return g

    # -- fake players for testing without spending money
    def mock_agent(self, agent, retry=False):
        name = agent["name"]
        others = [t for t in self.table if t["move"] == "assert" and t["agent"] != name]
        mine = [t for t in self.table if t["move"] == "assert" and t["agent"] == name and not t.get("retracted_at")]
        open_goals = [g for g in self.goals if g["status"] == "open"]
        r = random.random()
        if not retry and random.random() < 0.05:
            return "(mock) I think the answer is probably B."  # unreadable, to test the retry
        if r < 0.06:
            return json.dumps({"move": "ask", "goal": f"(mock) something {name} needs established",
                               "for": random.choice(self.letters), "say": "(mock) Does anyone know this?"})
        if others and r < 0.12:
            return json.dumps({"move": "show", "turn": random.choice(others)["turn"],
                               "reason": "(mock) I don't see how this connects."})
        if mine and r < 0.17:
            return json.dumps({"move": "retract", "turn": random.choice(mine)["turn"],
                               "reason": "(mock) That rested on a misreading."})
        unshared = [f for f in agent["facts"] if f not in self.on_table]
        if unshared and r < 0.8:
            live = [t for t in others if not t.get("retracted_at")]
            if live and random.random() < 0.6:
                target = random.choice(live)["turn"]
                reading = {"premises": "(mock)", "assumption": "(mock) what I think they assume",
                           "conclusion": "(mock)"}
            else:
                target, reading = random.choice(self.letters), None
            reply = {
                "move": "assert", "fact": random.choice(unshared), "target": target,
                "relation": random.choice(["attack", "support"]), "say": "(mock) This matters.",
                "answers_goal": random.choice(open_goals)["id"] if open_goals and random.random() < 0.5 else None,
                "opens": ["(mock) a premise this still needs"] if random.random() < 0.3 else [],
                "reconstruction": {"reading_of_target": reading,
                                   "aim": random.choice(["premise", "assumption", "conclusion"]),
                                   "mine": {"premises": "(mock)", "assumption": "(mock) what I assume",
                                            "conclusion": "(mock)"}}}
            if random.random() < 0.2:  # the nested format some models use
                reply = {"move": reply}
            return json.dumps(reply)
        return json.dumps({"move": "pass"})

    def mock_judge(self):
        return json.dumps({"answer": random.choice(self.letters),
                           "explanation": "(mock judge, random answer)", "facts_used": []})

    def mock_check(self, pairs):
        return json.dumps({"pairs": [{"pair": f"{t['turn']}->{o['turn']}",
                                      "rating": random.choice(["match", "partial", "mismatch"]),
                                      "note": "(mock)"} for t, o in pairs]})

    # -- numbers for the paper
    def metrics(self):
        private_shared = self.on_table - self.public
        key = set(self.sc.get("key_facts", [])) - self.public
        asserts = [t for t in self.table if t["move"] == "assert"]
        shows = [t for t in self.table if t["move"] == "show" and not t["problem"]]
        by_attacked = 0
        for s in shows:
            tgt = self.entry_at(s["turn_requested"])
            if tgt and tgt["relation"] == "attack" and tgt["target_kind"] == "turn":
                attacked = self.entry_at(tgt["target"])
                if attacked and attacked["agent"] == s["agent"]:
                    by_attacked += 1
        checked = [t for t in asserts if t.get("reading_check")]
        rating = lambda r: sum(1 for t in checked if t["reading_check"]["rating"] == r)
        mism = [t for t in checked if t["reading_check"]["rating"] == "mismatch"]
        return {
            "strategy": self.strategy,
            "turns": len(self.table),
            "arguments": len(asserts),
            "goals_posted": len(self.goals),
            "goals_posted_by_ask": sum(1 for g in self.goals if g["via"] == "ask"),
            "goals_opened_by_arguments": sum(1 for g in self.goals if g["via"] == "assert"),
            "goals_answered": sum(1 for g in self.goals if g["status"] == "answered"),
            "goals_open_at_end": sum(1 for g in self.goals if g["status"] == "open"),
            "arguments_answering_a_goal": sum(1 for t in asserts if t.get("answers_goal")),
            "private_facts_total": len(self.owner),
            "private_facts_shared": len(private_shared),
            "shared_by_agent": {a["name"]: [len(set(a["facts"]) & private_shared), len(a["facts"])]
                                for a in self.agents},
            "key_facts_total": len(key),
            "key_facts_shared": len(key & private_shared),
            "extra_facts_shared": len(private_shared - key) if key else None,
            "show_requests": len(shows),
            "shows_by_attacked_agent": by_attacked,
            "retractions": sum(1 for t in self.table if t["move"] == "retract" and not t["problem"]),
            "readings_checked": len(checked),
            "readings_match": rating("match"),
            "readings_partial": rating("partial"),
            "readings_mismatch": len(mism),
            "mismatches_revealed": sum(1 for t in mism if t.get("revealed_at")),
            "mismatches_retracted": sum(1 for t in mism if t.get("retracted_at")),
            "mismatches_unrepaired": sum(1 for t in mism if not t.get("revealed_at") and not t.get("retracted_at")),
            "invalid_citations": sum(1 for t in asserts if t["fact_status"] == "invalid"),
            "moves_with_problems": sum(1 for t in self.table if t.get("problem")),
            "unreadable_replies": sum(1 for t in self.table if t["move"] == "unreadable" and not t.get("error")),
            "replies_needing_retry": sum(1 for t in self.table if t.get("retried")),
            "api_errors": sum(1 for t in self.table if t.get("error")),
        }


# ----------------------------------------------------------------- display

BADGE = {"match": ":green[match]", "partial": ":orange[partial]", "mismatch": ":red[mismatch]"}


def show_recon(rec):
    if not rec:
        st.caption("No reconstruction recorded.")
        return
    if rec.get("reading_of_target"):
        st.markdown(f"**Reads the target as:** {part_text(rec['reading_of_target'])}")
    if rec.get("aim"):
        st.markdown(f"**Aims at:** {rec['aim']}")
    st.markdown(f"**Own argument:** {part_text(rec.get('mine'))}")


def show_turn(entry, avatar, facts):
    mv = entry["move"]
    with st.chat_message(entry["agent"], avatar=avatar):
        head = f"**{entry['agent']}** · `{entry['model']}` · turn {entry['turn']} · **{mv}**"
        if mv == "assert":
            head += f" · {describe_target(entry)}"
        elif mv in ("show", "retract"):
            head += f" → turn {entry.get('turn_requested')}"
        elif mv == "ask" and entry.get("goal_id"):
            head += f" · posts {entry['goal_id']}"
        st.markdown(head)
        if entry.get("error"):
            st.error(entry["error"])
        if mv == "assert":
            if entry["say"]:
                st.markdown(f"> {entry['say']}")
            fid, status = entry["fact"], entry["fact_status"]
            if status in ("new", "on table"):
                st.markdown(f"- `{fid}` ({'new' if status == 'new' else 'already on table'}): {facts.get(fid, '')}")
            elif status == "invalid":
                st.markdown(f"- :red[`{fid}`: invalid citation (not this agent's fact)]")
            else:
                st.markdown("- :red[no fact given]")
            if entry["extra_facts"]:
                st.caption(f"Extra facts ignored (one per move): {', '.join(entry['extra_facts'])}")
            if entry.get("answers_goal"):
                st.markdown(f"- :green[answers {entry['answers_goal']}]")
            for gid, gtxt in zip(entry.get("opened") or [], entry.get("opened_texts") or []):
                st.markdown(f"- :orange[opens {gid}:] {gtxt}")
            if entry.get("problem"):
                st.caption(f":orange[{entry['problem']}]")
            if entry.get("retracted_at"):
                st.caption(f"Withdrawn at turn {entry['retracted_at']}.")
            seen = (f"revealed at turn {entry['revealed_at']}" if entry.get("revealed_at")
                    else "not seen by other agents")
            with st.expander(f"🔒 Private reconstruction ({seen})"):
                show_recon(entry["reconstruction"])
                rc = entry.get("reading_check")
                if rc:
                    st.markdown(f"**Reading check:** {BADGE.get(rc['rating'], rc['rating'])} {rc['note']}")
        elif mv == "show":
            if entry["problem"]:
                st.caption(f":orange[Nothing shown: {entry['problem']}]")
            else:
                if entry["reason"]:
                    st.markdown(f"> {entry['reason']}")
                st.markdown(f"Revealed **{entry['owner']}**'s reconstruction behind turn {entry['turn_requested']}:")
                show_recon(entry["revealed"])
        elif mv == "retract":
            if entry["reason"]:
                st.markdown(f"> {entry['reason']}")
            if entry["problem"]:
                st.caption(f":orange[Not withdrawn: {entry['problem']}]")
        elif mv == "ask":
            if entry.get("problem"):
                st.caption(f":orange[No goal posted: {entry['problem']}]")
            else:
                st.markdown(f"- :orange[{entry['goal_id']}:] {entry['goal_text']}")
                if entry.get("say"):
                    st.markdown(f"> {entry['say']}")
        elif mv == "unreadable":
            st.caption(":orange[The reply could not be read, even after one retry. Not counted as a pass.]")
        if entry.get("retried") and mv != "unreadable":
            st.caption("The first reply could not be read; this is the retry.")
        with st.expander("raw model output"):
            st.code(entry.get("raw") or "(empty)", language=None)


def verdict_metric(col, label, v, gold):
    col.metric(label, v or "none",
               ("correct" if v == gold else "wrong") if v else "no single answer accepted",
               delta_color="green" if v == gold else ("red" if v else "gray"), delta_arrow="off")


def show_graph(g, sc, stamp):
    st.subheader("Argument graph")
    st.caption("Built from the targets the agents declared (which argument each move attacks or supports). "
               "The app labels it with grounded semantics. Withdrawn arguments and moves without a valid "
               "fact are left out.")
    c1, c2, c3, c4 = st.columns(4)
    verdict_metric(c1, "Graph verdict (as declared)", g["verdict"], sc["answer"])
    if g["labels_without_misreads"] is not None:
        verdict_metric(c2, "Graph verdict (misread links removed)", g["verdict_without_misreads"], sc["answer"])
    c3.metric("Arguments", sum(1 for n in g["nodes"] if n["kind"] == "argument"))
    c4.metric("Attacks / supports", f"{len(g['attacks'])} / {len(g['supports'])}")
    st.graphviz_chart(g["dot"], width="stretch")
    rule = (" A support for one answer also counts as an attack on the other answers."
            if g["support_rule"] else " Supports are drawn but don't change the labels.")
    st.caption("Green = accepted (IN) · red = rejected (OUT) · grey = undecided. "
               "Solid red arrow = attack · dashed green arrow = support · orange dashed = the responder misread "
               "its target (judge's check) · dotted grey line = answers exclude each other." + rule)
    for note in g.get("notes", []):
        st.caption(note)
    rows = []
    for n in g["nodes"]:
        row = {"node": n["id"], "status": g["labels"][n["id"]]}
        if g["labels_without_misreads"] is not None:
            row["status without misreads"] = g["labels_without_misreads"][n["id"]]
        row.update({"by": n["by"], "turn": n["turn"], "fact": ", ".join(n["facts"]), "text": n["text"]})
        rows.append(row)
    st.dataframe(rows, width="stretch", hide_index=True)
    st.download_button("Download graph (.dot, opens in Graphviz)", g["dot"],
                       file_name=f"inquiry_graph_{stamp}.dot", mime="text/vnd.graphviz")


def show_reconstructions(run):
    m = run["metrics"]
    st.subheader("Reconstructions and repair")
    st.caption("Every argument comes with a private reconstruction. 'Show' reveals one. The reading check "
               "compares how a responder read an argument with what its speaker meant.")
    c = st.columns(6)
    c[0].metric("Show requests", m["show_requests"])
    c[1].metric("…by the agent attacked", m["shows_by_attacked_agent"])
    c[2].metric("Retractions", m["retractions"])
    c[3].metric("Readings checked", m["readings_checked"])
    c[4].metric("Mismatched readings", m["readings_mismatch"])
    c[5].metric("Mismatches left unrepaired", m["mismatches_unrepaired"])
    rows = []
    for t in run["transcript"]:
        if t["move"] != "assert":
            continue
        rc = t.get("reading_check") or {}
        rows.append({"turn": t["turn"], "agent": t["agent"], "fact": t["fact"], "move": describe_target(t),
                     "aim": t["reconstruction"].get("aim", ""),
                     "assumes (own)": (t["reconstruction"].get("mine") or {}).get("assumption", ""),
                     "reading check": rc.get("rating", ""), "check note": rc.get("note", ""),
                     "revealed at": t.get("revealed_at"), "withdrawn at": t.get("retracted_at")})
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    rcheck = run.get("reading_check")
    if rcheck and rcheck.get("error"):
        st.error(rcheck["error"])
    if rcheck and rcheck.get("raw"):
        with st.expander("Raw reading-check output from the judge"):
            st.code(rcheck["raw"], language=None)


def show_goals(run):
    m = run["metrics"]
    st.subheader("Open goals")
    st.caption("Statements the agents said the group still needed to establish. A goal counts as answered when "
               "an argument claims to answer it; the app does not check that the fact really settles it.")
    c = st.columns(5)
    c[0].metric("Goals posted", m["goals_posted"])
    c[1].metric("…with ask", m["goals_posted_by_ask"])
    c[2].metric("…opened by arguments", m["goals_opened_by_arguments"])
    c[3].metric("Answered", m["goals_answered"])
    c[4].metric("Still open at the end", m["goals_open_at_end"])
    rows = [{"goal": g["id"], "posted by": g["by"], "turn": g["turn"], "how": g["via"],
             "for": (f"{'answer' if g['for_kind'] == 'option' else 'turn'} {g['for']}" if g.get("for") is not None else ""),
             "status": g["status"], "answered at": g["answered_at"], "answered by": g["answered_by"],
             "with fact": g["answer_fact"], "text": g["text"]} for g in run.get("goals", [])]
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)


def show_results(run):
    sc = run["scenario"]
    avatar = {a["name"]: AVATARS[i % len(AVATARS)] for i, a in enumerate(sc["agents"])}
    facts = all_facts(sc)
    stamp = run["timestamp"].replace(":", "-")
    strat = run.get("strategy") or {}
    st.subheader("Transcript")
    st.caption(f"Strategy: {strat.get('label', '?')} · ended because: {run['ended']}")
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

    show_graph(run["graph"], sc, stamp)
    show_reconstructions(run)
    show_goals(run)

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

    strategy = st.radio("Strategy", list(STRATEGIES), format_func=lambda k: STRATEGIES[k]["label"],
                        captions=[STRATEGIES[k]["caption"] for k in STRATEGIES],
                        help="Every agent gets the same moves, information, turn limit and stopping rule. "
                             "Only the strategy paragraph in their instructions changes.")
    max_turns = st.slider("Max turns", 4, 80, 30, help="One fact per move, so dialogues run longer.")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    show_targets = st.checkbox("Agents see what each move targets", value=True,
                               help="Off: others hear only the fact and the spoken line, not which "
                                    "argument it attacks or supports.")
    check_readings = st.checkbox("Check readings at the end (one judge call)", value=True,
                                 help="The judge compares each responder's reading with what the "
                                      "speaker privately meant.")
    run_pooled = st.checkbox("Also run pooled baseline (judge sees all facts)", value=True)
    support_rule = st.checkbox("Support for one answer counts as attack on the others", value=True,
                               help="Answers are mutually exclusive, so arguing for B argues against A and C. "
                                    "Turn off to label the graph with attacks only.")

sc = scenarios[chosen]
problems = validate(sc)
if problems:
    st.error("This scenario has problems: " + "; ".join(problems))
    st.stop()

st.title("🔎 Inquiry Arena")
st.caption("LLM agents with split knowledge try to answer a question together. Each argument states one fact; "
           "the reasoning behind it stays private unless someone asks to see it.")

st.markdown(f"**Strategy:** {STRATEGIES[strategy]['label']}: *{STRATEGIES[strategy]['question']}*")
with st.expander("What the agents are told"):
    st.caption("The same rules for every strategy (shown here for the first agent), ending with the strategy paragraph.")
    st.code(RULES.format(name="<agent>", role="", n="<n>", others="<the other agents>",
                         strategy=STRATEGIES[strategy]["text"]), language=None)

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
with st.expander("Show gold answer and scenario notes"):
    st.markdown(f"**{sc['answer']}**. {sc['options'][sc['answer']]}")
    if sc.get("gold_explanation"):
        st.markdown(sc["gold_explanation"])
    if sc.get("notes"):
        st.caption(sc["notes"])

if st.button(f"▶ Run dialogue ({STRATEGIES[strategy]['label']})", type="primary", width="stretch"):
    if not mock and not api_key:
        st.error("Add an OpenRouter key in the sidebar, or switch on mock mode.")
        st.stop()
    settings = {"max_turns": max_turns, "temperature": temperature, "mock": mock,
                "show_targets": show_targets, "check_readings": check_readings, "support_rule": support_rule,
                "strategy": strategy}
    ref = Referee(sc, models, judge_model, settings, api_key, models_available)
    avatar = {a["name"]: AVATARS[i % len(AVATARS)] for i, a in enumerate(sc["agents"])}
    facts = all_facts(sc)
    status = st.empty()
    live = st.container()

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
    reading = None
    if check_readings:
        status.info("The judge is checking how each argument was read...")
        reading = ref.check_readings()
    graph = ref.declared_graph(support_rule)
    status.empty()
    st.session_state.last_run = {
        "app": APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenario": sc,
        "models": models,
        "judge_model": judge_model,
        "settings": settings,
        "ended": ended,
        "strategy": {"key": strategy, "label": STRATEGIES[strategy]["label"], "text": STRATEGIES[strategy]["text"]},
        "transcript": ref.table,
        "goals": ref.goals,
        "first_shared": ref.first_shared,
        "judge": judge,
        "pooled": pooled,
        "reading_check": reading,
        "graph": graph,
        "metrics": ref.metrics(),
        "usage": ref.usage,
    }
    st.rerun()

if st.session_state.get("last_run", {}).get("app") not in (None, APP_VERSION):
    del st.session_state["last_run"]  # a run from an older version of the app
if "last_run" in st.session_state:
    st.divider()
    show_results(st.session_state.last_run)
