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

Orchestrator (v0.5.2): instead of one strategy for the whole dialogue, an orchestrator can assign a
strategy to each agent at the start of every round (one turn per agent). It never takes part and never
sees the gold answer. How it decides is a setting: fixed rules over the public record, an LLM that reads
the public record, a fixed schedule, or random assignment (a control).

v0.6: reconstructions are structured (premises are fact IDs and turns; the assumption is an
"If ..., then ..." bridge) and written before the move; flawed moves are sent back once to be fixed.
After the dialogue each agent privately writes its own answer as a proof tree plus a plain-English
explanation (the closing round). The judge does the same from the public record, the pooled baseline
from all facts, and the gold tree is put into plain English. One more judge call checks every step and
every explanation. All trees are compared with the gold tree.

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

APP_VERSION = "inquiry-arena v0.6"
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
  unless someone asks for it. Write it in this order, BEFORE you decide your own move:
  * reading_of_target (only when you respond to a turn; null when you aim at an answer): that argument
    as its speaker most plausibly meant it, from their fact and their words only. Do not credit them with
    anything they did not say or need, and do not shape your reading to suit your reply;
  * then your move, and aim: the part of the target you attack or support ("premise", "assumption" or "conclusion");
  * mine: your own argument in full.
  A reading and your own argument have the same three parts:
  * premises: a list of IDs only: facts (yours or already public) and earlier turns, e.g. ["B3", "turn 2"];
  * assumption: the unstated general rule that, added to the premises, makes the conclusion follow,
    written as "If ..., then ...". It must not repeat a premise;
  * conclusion: what the argument claims.
  Every assumption and conclusion is a full sentence that could be true or false, never a topic.
  If no public fact establishes your assumption, also put it in "opens".
  Example (fact K2 "the bakery opens at 7 am", supporting turn 1, which said Maria reached the bakery at 6:30 am):
    good: {{"premises": ["K2", "turn 1"], "assumption": "If someone reaches a shop before it opens, then they have to wait",
            "conclusion": "Maria had to wait outside the bakery."}}
    bad:  {{"premises": ["bakery hours"], "assumption": "The bakery opens at 7 am", "conclusion": "Waiting"}}
          (a topic as a premise, an assumption that repeats a premise, a conclusion that is not a sentence)
- Open goals: a public list of statements the group still needs to establish (G1, G2, ...).
  Anyone can post one. An argument can say which open goal it answers, and can open new goals.
- Be honest: use only your own facts or facts already on the table. Never invent facts or IDs.

On each turn reply with ONLY one flat JSON object (do not nest it inside another object), one of these:

1. Make an argument with exactly ONE fact (keep the keys in this order):
{{"move": "assert", "target": <turn number or answer letter>,
  "reading_of_target": {{"premises": ["ID or turn N", ...], "assumption": "If ..., then ...", "conclusion": "..."}} or null,
  "fact": "ID", "relation": "attack" or "support", "aim": "premise" or "assumption" or "conclusion",
  "mine": {{"premises": ["ID or turn N", ...], "assumption": "If ..., then ...", "conclusion": "..."}},
  "say": "what you say out loud, one or two sentences",
  "answers_goal": "G<number>" or null,
  "opens": ["a statement this argument still needs, not yet established", ...] or []}}

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

# ---- orchestrator: assigns a strategy to each agent at the start of every round
ORCHESTRATED = "orchestrated"
POLICIES = {
    "rules": {"label": "Rules",
              "caption": "Fixed if-then rules over the public record. No model call; the same state always "
                         "gets the same assignment."},
    "llm": {"label": "LLM orchestrator",
            "caption": "A model reads the public record and picks a strategy for each agent. One call per round."},
    "schedule": {"label": "Fixed schedule",
                 "caption": "Round 1 free, then goals, assumptions, contrast, and round again through those three. "
                            "Ignores what happens."},
    "random": {"label": "Random (control)",
               "caption": "A random strategy for each agent every round. Tells us whether switching itself helps."},
}
SCHEDULE = ["free", "goals", "assumptions", "contrast"]
# The rules policy, in order: the first rule whose condition holds decides the round (same strategy for everyone).
RULE_TABLE = [
    ("first round", "free", "Nothing is on the table yet, so everyone brings what seems useful."),
    ("everyone just passed (last call)", "assumptions",
     "Look again at what the favoured answer takes for granted; someone may hold the missing premise."),
    ("an open goal is waiting", "goals", "Someone may hold the fact that answers it."),
    ("two or more answers have support", "contrast", "Separate the candidates before building further."),
    ("the last round added no new fact", "assumptions",
     "Repeats and passes suggest a gap that nobody has named."),
    ("otherwise", "goals", "Keep building the explanation and name what is still missing."),
]
ORCH_LEAD = ("For this round the orchestrator has given you the strategy below. "
             "It may give you a different one in a later round.\n")
ORCH_SYSTEM = ("You are the orchestrator of an inquiry dialogue between AI agents. You do not take part, you do not "
               "know the answer, and you cannot add facts. At the start of each round you choose which strategy each "
               "agent follows during that round. Reply with ONLY one JSON object, nothing else.")
LAST_CALL_NOTE = "Everyone passed. Last call: one more round before the dialogue ends."


def strategy_name(k):
    return STRATEGIES[k]["label"].split(". ", 1)[-1] if k in STRATEGIES else str(k)


def condition_name(strategy, policy=None):
    """One name per experimental condition: a strategy key, or orch-<policy>."""
    return f"orch-{policy}" if strategy == ORCHESTRATED else strategy


def parse_condition(name):
    """Inverse of condition_name: 'orch-rules' -> ('orchestrated', 'rules'); 'goals' -> ('goals', None)."""
    if name.startswith("orch-") and name[5:] in POLICIES:
        return ORCHESTRATED, name[5:]
    return name, None


def condition_label(strategy, policy=None):
    if strategy == ORCHESTRATED:
        return f"5. Orchestrated: {POLICIES.get(policy, {}).get('label', policy)}"
    return STRATEGIES[strategy]["label"]


def match_strategy(x):
    """Map a model's free-form strategy name onto a key."""
    v = _s(x).lower()
    if v in STRATEGIES:
        return v
    for k, words in {"free": ("free",), "goals": ("goal",), "assumptions": ("assum", "repair", "check"),
                     "contrast": ("contrast", "test", "distinguish")}.items():
        if any(w in v for w in words):
            return k
    return None


FORMAT_REMINDER = """

IMPORTANT: your last reply could not be read. Reply with exactly one flat JSON object and nothing else,
for example: {"move": "pass"}, or {"move": "assert", "target": "A", "reading_of_target": null, "fact": "ID",
"relation": "support", "aim": "conclusion", "mine": {"premises": ["ID"], "assumption": "If ..., then ...",
"conclusion": "..."}, "say": "...", "answers_goal": null, "opens": []}"""

FIX_REMINDER = """

Your last reply had these problems:
{problems}
Reply again with one corrected JSON object (or choose a different move)."""

# ---- after the dialogue: trees and plain-English explanations
TREE_FORMAT = ('[{"id": "S1", "premises": ["ID", "ID"], "conclusion": "one full sentence"}, '
               '{"id": "S2", "premises": ["S1", "ID"], "conclusion": "..."}]')
EXPLAIN_RULE = ("two to four sentences of connected, natural English that someone who has not seen the facts could "
                "follow. Say why, with words like because, so and therefore. No fact IDs, no lists, and nothing "
                "that is not in your tree")
CLOSING_SYSTEM = ("You are {name}, one of the agents in an inquiry that has just ended. "
                  "Reply with ONLY one JSON object, nothing else.")
CLOSING_TASK = """The discussion is over. Now, privately (no other agent will see this), give your own answer
and explanation. Build the explanation as a small proof tree, then write it out in plain English.
- tree: steps S1, S2, ... Each step combines its premises into one conclusion. Premises are IDs of facts you can
  see (your own facts, public facts, facts stated in the discussion) or earlier steps. You may use your own facts
  even if you never said them. Use only what supports your answer. The last step concludes the answer.
- explanation: """ + EXPLAIN_RULE + """.
Reply with ONLY: {"answer": "<letter>", "tree": """ + TREE_FORMAT + """, "explanation": "..."}"""
JUDGE_TASK = """Based only on the information above, which answer is best supported? Build the explanation as a
small proof tree whose premises are only IDs of the facts listed above (or earlier steps), then write it out in
plain English.
- explanation: """ + EXPLAIN_RULE + """.
Reply with ONLY: {"answer": "<letter>", "tree": """ + TREE_FORMAT + """, "explanation": "...", "facts_used": ["ID", ...]}"""
GOLD_SYSTEM = "You rewrite proofs as short explanations. Reply with ONLY one JSON object, nothing else."
CHECK_TREES_SYSTEM = ("You check proof steps and explanations. You know only what is written in the message. "
                      "Reply with ONLY one JSON object, nothing else.")
HOLDER_FILL = ["#d6ebe8", "#f6e7c8", "#e5dcf3", "#dbe7f6", "#f3dcdc", "#e2efd7"]
PUBLIC_FILL = "#eeeeee"

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


@st.cache_resource(show_spinner=False)
def _read_json(path, mtime):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_collections():
    """{collection name: {scenario label: scenario}}.
    Built-in: scenario_*.json next to app.py. Collections: files in the scenarios/ folder, each either one
    scenario or a pack {"collection": "name", "scenarios": [...]} (as made by tools/convert_entailmentbank.py)."""
    cols = {}
    builtin = {}
    for path in sorted(glob.glob(os.path.join(APP_DIR, "scenario_*.json"))):
        try:
            sc = _read_json(path, os.path.getmtime(path))
            builtin[sc.get("title", os.path.basename(path))] = sc
        except Exception as e:
            st.sidebar.warning(f"Could not read {os.path.basename(path)}: {e}")
    if builtin:
        cols["Built-in scenarios"] = builtin
    for path in sorted(glob.glob(os.path.join(APP_DIR, "scenarios", "*.json"))):
        try:
            data = _read_json(path, os.path.getmtime(path))
        except Exception as e:
            st.sidebar.warning(f"Could not read scenarios/{os.path.basename(path)}: {e}")
            continue
        if isinstance(data, dict) and isinstance(data.get("scenarios"), list):
            name = data.get("collection") or os.path.basename(path)
            cols[name] = {(x.get("label") or x.get("title") or f"#{i + 1}"): x
                          for i, x in enumerate(data["scenarios"]) if isinstance(x, dict)}
        elif isinstance(data, dict) and "question" in data:
            cols.setdefault("Other scenarios", {})[data.get("title", os.path.basename(path))] = data
    return cols


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


def norm_premises(x):
    """A list of premise tokens: fact IDs as written, or 'turn N'."""
    if x is None:
        return []
    items = list(x) if isinstance(x, (list, tuple)) else re.split(r"[;,]", _s(x))
    out = []
    for it in items:
        v = _s(it).strip().strip('"\'').strip()
        if not v or v.lower() in ("none", "null"):
            continue
        m = re.fullmatch(r"(?:turn\s*|#)(\d+)", v, re.I) or re.fullmatch(r"(\d+)", v)
        out.append(f"turn {int(m.group(1))}" if m else v)
    return out


def norm_part(x):
    if isinstance(x, dict):
        return {"premises": norm_premises(x.get("premises")), "assumption": _s(x.get("assumption")),
                "conclusion": _s(x.get("conclusion"))}
    if x is None or _s(x).lower() in ("", "null", "none", "n/a"):
        return None
    return {"premises": [], "assumption": "", "conclusion": _s(x)}


def recon_source(obj):
    """v0.6 puts the reconstruction fields at the top of the reply; older replies nest them."""
    return obj.get("reconstruction") if isinstance(obj.get("reconstruction"), dict) else obj


STOP = set("a an the of to in on is are was were be it its that this and or if then for with as by at from".split())


def content_words(t):
    return {w for w in re.findall(r"[a-z]+", str(t).lower()) if w not in STOP}


def overlap(a, b):
    A, B = content_words(a), content_words(b)
    return len(A & B) / len(A | B) if A and B else 0.0


def gold_steps(sc):
    """The gold tree as steps: [{"id", "premises", "conclusion"}]."""
    out = []
    for line in sc.get("gold_tree") or []:
        if "->" not in line:
            continue
        lhs, rhs = line.split("->", 1)
        cid, _, text = rhs.partition(":")
        out.append({"id": cid.strip(), "premises": [x.strip() for x in lhs.split("+") if x.strip()],
                    "conclusion": text.strip()})
    return out


def step_leafsets(steps):
    sets = {}
    for st_ in steps:
        L = set()
        for p in st_["premises"]:
            L |= sets.get(p, {p})
        sets[st_["id"]] = L
    return sets


def tree_leaves(steps):
    ids = {st_["id"] for st_ in steps}
    out = []
    for st_ in steps:
        for p in st_["premises"]:
            if p not in ids and p not in out:
                out.append(p)
    return out


def compare_tree(steps, sc):
    """Compare a tree with the gold tree: leaves (by fact ID) and steps (same set of leaves underneath)."""
    gold = gold_steps(sc)
    if not gold:
        return None
    gl, leaves = set(tree_leaves(gold)), set(tree_leaves(steps))
    mine = list(step_leafsets(steps).values())
    matched = [g["id"] for g, L in zip(gold, step_leafsets(gold).values()) if L in mine]
    return {"gold_leaves": len(gl), "leaves": len(leaves), "leaves_in_gold": len(leaves & gl),
            "leaf_recall": round(len(leaves & gl) / len(gl), 2) if gl else None,
            "leaf_precision": round(len(leaves & gl) / len(leaves), 2) if leaves else None,
            "gold_steps": len(gold), "gold_steps_matched": len(matched), "matched": matched,
            "not_in_gold": sorted(leaves - gl), "missing_from_gold": sorted(gl - leaves)}


def steps_done(steps, view):
    have, ok, changed = set(view), [], True
    while changed:
        changed = False
        for g in steps:
            if g["id"] not in ok and all(x in have for x in g["premises"]):
                have.add(g["id"])
                ok.append(g["id"])
                changed = True
    return ok


def norm_tree(raw, allowed):
    """Clean a tree a model wrote. allowed: the fact IDs this writer may use. Returns (steps, problems)."""
    if not isinstance(raw, list) or not raw:
        return [], ["no tree given"]
    canon = {k.upper(): k for k in allowed}
    steps, seen, problems = [], set(), []
    for i, x in enumerate(raw, 1):
        if not isinstance(x, dict):
            continue
        sid = (_s(x.get("id")) or f"S{i}").upper().replace(" ", "")
        good = []
        for p in norm_premises(x.get("premises")):
            P = p.upper().replace(" ", "")
            if P in seen:
                good.append(P)
            elif P in canon:
                good.append(canon[P])
            else:
                problems.append(f"{sid}: '{p}' is not a fact this writer could use, or an earlier step")
        steps.append({"id": sid, "premises": good, "conclusion": _s(x.get("conclusion"))})
        seen.add(sid)
    if not steps:
        problems.append("no tree given")
    return steps, problems


def norm_recon(r):
    r = r if isinstance(r, dict) else {}
    aim = _s(r.get("aim")).lower()
    aim = next((a for a in ("premise", "assumption", "conclusion") if aim.startswith(a[:5])), aim)
    return {"reading_of_target": norm_part(r.get("reading_of_target")), "aim": aim,
            "mine": norm_part(r.get("mine"))}


def premise_text(prem, facts=None):
    if isinstance(prem, str):  # older runs: free text
        return prem
    out = []
    for x in prem or []:
        out.append(f'{x} ("{facts[x]}")' if facts and x in facts else x)
    return ", ".join(out)


def part_text(p, facts=None):
    if not p:
        return "(none)"
    bits = []
    if p.get("premises"):
        bits.append(f"premises: {premise_text(p['premises'], facts)}")
    if p.get("assumption"):
        bits.append(f"unstated assumption: {p['assumption']}")
    if p.get("conclusion"):
        bits.append(f"conclusion: {p['conclusion']}")
    return "; ".join(bits) or "(empty)"


def recon_text(rec, facts=None):
    if not rec:
        return "(no reconstruction recorded)"
    s = []
    if rec.get("reading_of_target"):
        s.append(f"reads the target as [{part_text(rec['reading_of_target'], facts)}]")
    if rec.get("aim"):
        s.append(f"aims at its {rec['aim']}")
    s.append(f"own argument [{part_text(rec.get('mine'), facts)}]")
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
        self.orchestrated = self.strategy == ORCHESTRATED
        self.policy = settings.get("policy") or "rules" if self.orchestrated else None
        self.orchestrator_model = settings.get("orchestrator_model") or judge_model
        start = "free" if self.orchestrated else self.strategy
        self.current = {a["name"]: start for a in self.agents}  # strategy each agent follows right now
        self.rounds = []  # one record per orchestrator decision
        self.notes = []  # public notes from the referee, e.g. the last call
        self.last_call_at = None
        self.orch_calls = 0
        self.rng = random.Random(settings.get("seed"))
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
                    f"{t['turn_requested']}{reason}. Revealed record of {t['owner']}: {recon_text(t['revealed'], self.fact_text)}.")
        if t["move"] == "retract":
            if t["problem"]:
                return f"Turn {t['turn']} - {who} tried to withdraw turn {t['turn_requested']}, but {t['problem']}."
            reason = f' ("{t["reason"]}")' if t["reason"] else ""
            return f"Turn {t['turn']} - {who} withdraws turn {t['turn_requested']}{reason}."
        return f"Turn {t['turn']} - {who} passes."

    def public_record(self):
        lines = []
        for t in self.table:
            lines.append(self.public_line(t))
            lines += [f"Note: {n['text']}" for n in self.notes if n["after_turn"] == t["turn"]]
        return "\n".join(lines) or "(nothing yet - you speak first)"

    def agent_prompt(self, agent):
        notes = [f"Turn {t['turn']} (you {describe_target(t).replace('attacks', 'attacked').replace('supports', 'supported')}"
                 f" with {t['fact'] or 'no fact'}): {recon_text(t['reconstruction'], self.fact_text)}"
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

    # -- checks on an argument
    def visible_ids(self, agent):
        return set(agent["facts"]) | self.on_table | self.public

    def resolve_premises(self, prem, agent, turn_now):
        canon = {k.upper(): k for k in self.fact_text}
        vis = self.visible_ids(agent)
        good, bad = [], []
        for x in prem:
            if x.startswith("turn "):
                k = int(x.split()[1])
                e = self.entry_at(k)
                (good if 1 <= k < turn_now and e and e["move"] == "assert" else bad).append(x)
            elif x.upper() in canon and canon[x.upper()] in vis:
                good.append(canon[x.upper()])
            else:
                bad.append(x)
        return good, bad

    def recon_problems(self, rec, kind, agent, fact, turn_now):
        """Problems with a reconstruction. Also puts premises into canonical form, and adds the stated fact
        to the speaker's own premises if they left it out."""
        probs = []
        parts = [("your own argument (mine)", rec.get("mine"))]
        if kind == "turn":
            if rec.get("reading_of_target"):
                parts.insert(0, ("your reading of the target", rec["reading_of_target"]))
            else:
                probs.append("you respond to a turn but gave no reading_of_target")
        for label, prt in parts:
            if not prt:
                probs.append(f"{label} is missing")
                continue
            good, bad = self.resolve_premises(prt["premises"], agent, turn_now)
            prt["premises"] = good + bad
            if bad:
                probs.append(f"{label}: premises must be fact IDs you can see or earlier turns, not {bad}")
            elif not good and label.startswith("your reading"):
                probs.append(f"{label} lists no premises")
            a = prt.get("assumption", "")
            if len(a.split()) < 4 or not re.search(r"\b(if|whenever|when|unless)\b", a, re.I):
                probs.append(f"{label}: the assumption must be an 'If ..., then ...' rule")
            else:
                same = [x for x in good if not x.startswith("turn ") and overlap(a, self.fact_text.get(x, "")) >= 0.7]
                if same:
                    probs.append(f"{label}: the assumption repeats premise {same[0]}; give the unstated rule instead")
            if len(prt.get("conclusion", "").split()) < 4:
                probs.append(f"{label}: the conclusion must be a full sentence")
        mine = rec.get("mine")
        if mine is not None and fact in self.fact_text and fact not in mine["premises"]:
            mine["premises"].insert(0, fact)
        return probs

    def assert_problems(self, obj, agent, turn_now):
        """Everything wrong with an assert, without applying it."""
        fact = obj.get("fact", obj.get("facts"))
        if isinstance(fact, list):
            fact = fact[0] if fact else ""
        fact = _s(fact)
        probs = []
        if not fact:
            probs.append("no fact given")
        elif fact not in agent["facts"] and fact not in self.on_table:
            probs.append(f"{fact} is not one of your facts or a fact already on the table")
        kind, val = parse_target(obj.get("target"), self.letters)
        if kind == "turn":
            tgt = self.entry_at(val)
            if tgt is None or tgt["move"] != "assert" or tgt.get("retracted_at"):
                probs.append(f"turn {val} is not a live argument you can respond to")
                kind = None
        elif kind is None:
            probs.append("no target: give an earlier turn number or an answer letter")
        if not parse_relation(obj.get("relation")):
            probs.append("no relation: say attack or support")
        probs += self.recon_problems(norm_recon(recon_source(obj)), kind, agent, fact, turn_now)
        return probs

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
                     reconstruction=norm_recon(recon_source(obj)),
                     revealed_at=None, retracted_at=None, problem="; ".join(problems) or None)
        entry["recon_problems"] = self.recon_problems(entry["reconstruction"], kind, agent, fact, entry["turn"])

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
        text = STRATEGIES[self.current[agent["name"]]]["text"]
        if self.orchestrated:
            text = ORCH_LEAD + text
        return RULES.format(name=agent["name"], role=role, n=n, others=others, strategy=text)

    # -- the orchestrator: what it may look at (public record only, unless the oracle setting is on)
    def round_summary(self, first_turn, last_turn):
        seg = [t for t in self.table if first_turn <= t["turn"] <= last_turn]
        return {"turns": len(seg),
                "new_facts": sum(1 for t in seg if t["move"] == "assert" and t["fact_status"] == "new"),
                "repeats": sum(1 for t in seg if t["move"] == "assert" and t["fact_status"] == "on table"),
                "passes": sum(1 for t in seg if t["move"] == "pass"),
                "asks": sum(1 for t in seg if t["move"] == "ask"),
                "shows": sum(1 for t in seg if t["move"] == "show"),
                "retractions": sum(1 for t in seg if t["move"] == "retract")}

    def anchor(self, t, depth=0):
        """Follow an argument's chain of targets to an answer: returns (letter, +1 for / -1 against) or (None, 0)."""
        if depth > 50 or t.get("move") != "assert" or not t.get("relation"):
            return None, 0
        sign = 1 if t["relation"] == "support" else -1
        if t["target_kind"] == "option":
            return t["target"], sign
        if t["target_kind"] == "turn":
            tgt = self.entry_at(t["target"])
            if tgt and not tgt.get("retracted_at"):
                letter, s2 = self.anchor(tgt, depth + 1)
                return letter, sign * s2
        return None, 0

    def signals(self, trigger):
        """The state the orchestrator decides on. Computed from the public record only."""
        live = [t for t in self.table if t["move"] == "assert" and not t.get("retracted_at")
                and t["fact_status"] in ("new", "on table")]
        support = {k: 0 for k in self.letters}
        against = {k: 0 for k in self.letters}
        unanchored = 0
        for t in live:
            letter, sign = self.anchor(t)
            if letter is None:
                unanchored += 1
            elif sign > 0:
                support[letter] += 1
            else:
                against[letter] += 1
        net = {k: support[k] - against[k] for k in self.letters}
        best = max(net.values()) if net else 0
        leaders = [k for k in self.letters if net[k] == best]
        prev = self.rounds[-1]["start_turn"] if self.rounds else 1
        last = self.round_summary(prev, len(self.table)) if self.table else None
        return {
            "round": len(self.rounds) + 1, "trigger": trigger, "turns_so_far": len(self.table),
            "facts_public": len(self.on_table - self.public),
            "last_round": last,
            "open_goals": [g["id"] for g in self.goals if g["status"] == "open"],
            "support": support, "against": against, "unanchored_arguments": unanchored,
            "answers_with_support": [k for k in self.letters if support[k] > 0],
            "favourite": leaders[0] if best > 0 and len(leaders) == 1 else None,
        }

    def policy_rules(self, sig):
        last = sig["last_round"]
        checks = [
            sig["round"] == 1,
            sig["trigger"] == "last_call",
            bool(sig["open_goals"]),
            len(sig["answers_with_support"]) >= 2,
            bool(last) and last["turns"] > 0 and last["new_facts"] == 0,
            True,
        ]
        i = checks.index(True)
        cond, strat, why = RULE_TABLE[i]
        detail = {2: f" ({', '.join(sig['open_goals'])})", 3: f" ({', '.join(sig['answers_with_support'])})"}.get(i, "")
        return {n: strat for n in self.names}, f"Rule {i + 1}: {cond}{detail}. {why}"

    def policy_schedule(self, sig):
        r = sig["round"]
        strat = "free" if r == 1 else SCHEDULE[1 + (r - 2) % 3]
        return {n: strat for n in self.names}, f"Schedule, round {r}: {strategy_name(strat)}."

    def policy_random(self, sig):
        return {n: self.rng.choice(list(STRATEGIES)) for n in self.names}, "Random assignment (control)."

    def mock_orchestrator(self):
        return json.dumps({"assignments": {n: random.choice(list(STRATEGIES)) for n in self.names},
                           "reason": "(mock orchestrator, random choice)"})

    def policy_llm(self, sig):
        menu = "\n".join(f'- "{k}": {strategy_name(k)}. {v["question"]} {v["caption"]}' for k, v in STRATEGIES.items())
        hist = []
        for i, rec in enumerate(self.rounds):
            end = self.rounds[i + 1]["start_turn"] - 1 if i + 1 < len(self.rounds) else len(self.table)
            if end < rec["start_turn"]:
                continue
            o = self.round_summary(rec["start_turn"], end)
            hist.append(f"Round {rec['round']} (turns {rec['start_turn']}-{end}): "
                        + ", ".join(f"{a} {s}" for a, s in rec["assignments"].items())
                        + f". Result: {o['new_facts']} new facts, {o['repeats']} repeated facts, {o['passes']} passes, "
                          f"{o['asks']} goals posted with ask, {o['shows']} shows, {o['retractions']} withdrawals.")
        state = (f"Facts made public so far: {sig['facts_public']}. Open goals: {', '.join(sig['open_goals']) or 'none'}. "
                 f"Arguments for each answer (following chains of targets): "
                 + ", ".join(f"{k} {sig['support'][k]} for / {sig['against'][k]} against" for k in self.letters)
                 + f"; {sig['unanchored_arguments']} arguments not linked to any answer.")
        private = ""
        if self.settings.get("orchestrator_sees_private"):
            private = "\n\nPrivate facts (the agents cannot see each other's):\n" + "\n".join(
                f"{a['name']}:\n{facts_text(a['facts'])}" for a in self.agents)
        situation = ("Everyone passed in a row. This is the last round before the dialogue ends."
                     if sig["trigger"] == "last_call" else f"Round {sig['round']} is about to start.")
        user = (f"Question: {self.sc['question']}\nPossible answers:\n{options_text(self.sc)}\n\n"
                f"Public facts:\n{facts_text(self.sc.get('public_facts', {}))}\n\n"
                f"Public conversation so far:\n{self.public_record()}\n\n"
                f"Open goals:\n{self.goals_text()}{private}\n\n"
                f"Earlier rounds:\n{chr(10).join(hist) or '(none yet)'}\n\nState: {state}\n\n"
                f"Strategies you can assign:\n{menu}\n\n{situation} Each agent ({', '.join(self.names)}) takes one "
                "turn per round. Choose a strategy for each agent so that the group reaches an answer that the public "
                "record fully justifies, with as few turns and as few unneeded facts as possible. Agents may get "
                "different strategies. Decide from the state of the dialogue, not from which answer you think is right.\n"
                'Reply with ONLY: {"assignments": {"<agent>": "<strategy key>", ...}, "reason": "<one or two sentences>"}')
        self.orch_calls += 1
        raw, err = self.ask(self.orchestrator_model, ORCH_SYSTEM, user, self.mock_orchestrator, max_tokens=400)
        obj = parse_json(raw) or {}
        got = obj.get("assignments") if isinstance(obj.get("assignments"), dict) else {}
        lower = {n.lower(): n for n in self.names}
        assign = {}
        for k, v in got.items():
            name, strat = lower.get(_s(k).lower()), match_strategy(v)
            if name and strat:
                assign[name] = strat
        if not assign:
            return None, "", raw, err or "the orchestrator's reply could not be read"
        missing = [n for n in self.names if n not in assign]
        for n in missing:  # keep what they had
            assign[n] = self.current[n]
        reason = _s(obj.get("reason"))
        if missing:
            reason += f" (no strategy given for {', '.join(missing)}: kept the previous one)"
        return assign, reason, raw, err

    def assign_round(self, trigger):
        sig = self.signals(trigger)
        raw = err = None
        if self.policy == "llm":
            assign, reason, raw, err = self.policy_llm(sig)
            if assign is None:  # unreadable reply: fall back to the rules, and record that we did
                assign, reason = self.policy_rules(sig)
                reason = f"(orchestrator reply unreadable; rules used) {reason}"
        elif self.policy == "schedule":
            assign, reason = self.policy_schedule(sig)
        elif self.policy == "random":
            assign, reason = self.policy_random(sig)
        else:
            assign, reason = self.policy_rules(sig)
        self.current.update(assign)
        rec = {"round": sig["round"], "start_turn": len(self.table) + 1, "trigger": trigger, "policy": self.policy,
               "assignments": dict(self.current), "reason": reason, "signals": sig, "raw": raw, "error": err}
        self.rounds.append(rec)
        return rec

    def run(self, on_turn, on_status, on_round=None):
        n = len(self.agents)
        idx, passes_in_row, problems_in_row = 0, 0, 0
        turns_in_round, trigger = n, "start"  # a round is n turns; the first one starts at turn 1
        for turn in range(1, self.settings["max_turns"] + 1):
            if self.orchestrated and turns_in_round >= n:
                on_status(f"Round {len(self.rounds) + 1}: the orchestrator is assigning strategies...")
                rec = self.assign_round(trigger)
                if on_round:
                    on_round(rec)
                turns_in_round, trigger = 0, "round"
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
            sent_back = None
            if move == "assert" and self.settings.get("fix_moves", True):
                probs = self.assert_problems(obj, agent, turn)
                if probs:  # one chance to fix it, with the problems spelled out
                    note = FIX_REMINDER.format(problems="\n".join(f"- {x}" for x in probs))
                    raw3, err3 = self.ask(self.models[name], system, prompt + note,
                                          lambda: self.mock_agent(agent, retry=True))
                    obj3, move3 = read_reply(raw3) if not err3 else (None, "")
                    sent_back = {"problems": probs, "replaced": bool(not err3 and move3 in MOVE_ALIASES.values())}
                    raw = raw + "\n\n--- sent back to fix ---\n" + (raw3 or err3 or "")
                    if sent_back["replaced"]:
                        obj, move = obj3, move3
            entry = {"turn": turn, "agent": name, "model": self.models[name], "raw": raw,
                     "error": err, "retried": retried, "strategy": self.current[name],
                     "round": len(self.rounds) if self.orchestrated else None, "sent_back": sent_back}
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
            turns_in_round += 1

            problems_in_row = problems_in_row + 1 if entry["move"] == "unreadable" else 0
            if problems_in_row >= 2 * n:
                return ("stopped: too many API errors or unreadable replies in a row "
                        "(check your key, credits or model names)")
            if entry["move"] == "pass":
                passes_in_row += 1
            elif entry["move"] != "unreadable":
                passes_in_row = 0
            if passes_in_row >= n:
                if self.settings.get("last_call") and self.last_call_at is None:
                    # one more round before ending; under the orchestrator it starts a fresh round
                    self.last_call_at = turn
                    self.notes.append({"after_turn": turn, "text": LAST_CALL_NOTE})
                    passes_in_row = 0
                    if self.orchestrated:
                        turns_in_round, trigger = n, "last_call"
                    on_status(LAST_CALL_NOTE)
                else:
                    return "everyone passed" + (" after the last call" if self.last_call_at else "")
            idx = next_idx if next_idx is not None else (idx + 1) % n
        return "turn limit reached"

    # -- the judge answers from the public record only
    def judge(self, pooled=False):
        if pooled:
            all_f = dict(self.sc.get("public_facts", {}))
            for a in self.agents:
                all_f.update(a["facts"])
            body = f"All known facts:\n{facts_text(all_f)}"
            allowed = set(all_f)
        else:
            shared = {fid: self.fact_text[fid] for fid in sorted(self.on_table - self.public)}
            body = (f"Public facts:\n{facts_text(self.sc.get('public_facts', {}))}\n\n"
                    f"Facts stated during the discussion:\n{facts_text(shared)}\n\n"
                    f"The discussion:\n{self.public_record()}\n\n"
                    f"Open goals at the end of the discussion:\n{self.goals_text()}")
            allowed = set(self.public) | set(self.on_table)
        user = f"Question: {self.sc['question']}\nPossible answers:\n{options_text(self.sc)}\n\n{body}\n\n{JUDGE_TASK}"
        raw, err = self.ask(self.judge_model, JUDGE_SYSTEM, user, lambda: self.mock_tree_reply(allowed),
                            max_tokens=2000)
        obj = parse_json(raw) or {}
        ans = str(obj.get("answer", "")).strip().upper()[:1]
        tree, tprobs = norm_tree(obj.get("tree"), allowed)
        used = obj.get("facts_used") if isinstance(obj.get("facts_used"), list) else tree_leaves(tree)
        used = [_s(x) for x in used]
        return {"model": self.judge_model, "answer": ans, "correct": ans == self.sc["answer"],
                "explanation": _s(obj.get("explanation")), "facts_used": used,
                "facts_used_not_allowed": [x for x in used if x not in allowed],
                "tree": tree, "tree_problems": tprobs, "raw": raw, "error": err}

    # -- after the dialogue: each agent's own answer, as a tree and in plain English (private)
    def closing(self):
        out = {}
        for a in self.agents:
            name = a["name"]
            allowed = self.visible_ids(a)
            user = (f"Question: {self.sc['question']}\nPossible answers:\n{options_text(self.sc)}\n\n"
                    f"Public facts:\n{facts_text(self.sc.get('public_facts', {}))}\n\n"
                    f"Your private facts (only you see these):\n{facts_text(a['facts'])}\n\n"
                    f"The discussion:\n{self.public_record()}\n\nOpen goals:\n{self.goals_text()}\n\n{CLOSING_TASK}")
            raw, err = self.ask(self.models[name], CLOSING_SYSTEM.format(name=name), user,
                                lambda allowed=allowed: self.mock_tree_reply(allowed), max_tokens=2000)
            obj = parse_json(raw) or {}
            ans = _s(obj.get("answer")).upper()[:1]
            tree, tprobs = norm_tree(obj.get("tree"), allowed)
            out[name] = {"answer": ans, "correct": ans == self.sc["answer"], "tree": tree, "tree_problems": tprobs,
                         "explanation": _s(obj.get("explanation")),
                         "unsaid_own_facts_used": [x for x in tree_leaves(tree)
                                                   if x in a["facts"] and x not in self.on_table],
                         "raw": raw, "error": err}
        return out

    # -- the gold tree, in plain English (one call; the gold tree itself is never shown to any agent)
    def verbalise_gold(self):
        gold = gold_steps(self.sc)
        if not gold:
            return None
        facts = all_facts(self.sc)
        concl = {g["id"]: g["conclusion"] for g in gold}
        lines = [f"- From: {' AND '.join(facts.get(x) or concl.get(x) or x for x in g['premises'])}\n"
                 f"  it follows: {g['conclusion']}" for g in gold]
        user = (f"Question: {self.sc['question']}\nCorrect answer: {self.sc['answer']}. "
                f"{self.sc['options'][self.sc['answer']]}\n\nProof, step by step:\n" + "\n".join(lines) +
                "\n\nRewrite this proof as an explanation of the answer in two to four sentences of connected, natural "
                "English for a student. Keep the reasoning; use only what the proof says and add nothing; no lists, "
                'no IDs.\nReply with ONLY: {"explanation": "..."}')
        raw, err = self.ask(self.judge_model, GOLD_SYSTEM, user,
                            lambda: json.dumps({"explanation": "(mock) " + " So ".join(g["conclusion"] for g in gold)}),
                            max_tokens=600)
        return {"explanation": _s((parse_json(raw) or {}).get("explanation")), "model": self.judge_model,
                "raw": raw, "error": err}

    def collect_trees(self, judge, pooled, closing, gold_text):
        trees = {}
        gold = gold_steps(self.sc)
        if gold:
            trees["Gold"] = {"kind": "gold", "answer": self.sc["answer"], "correct": True, "tree": gold,
                             "tree_problems": [], "explanation": (gold_text or {}).get("explanation", ""),
                             "compare": compare_tree(gold, self.sc)}
        for name, c in (closing or {}).items():
            trees[name] = {"kind": "agent", "answer": c["answer"], "correct": c["correct"], "tree": c["tree"],
                           "tree_problems": c["tree_problems"], "explanation": c["explanation"],
                           "unsaid_own_facts_used": c["unsaid_own_facts_used"],
                           "compare": compare_tree(c["tree"], self.sc)}
        for key, j in (("Judge", judge), ("Pooled", pooled)):
            if j:
                trees[key] = {"kind": key.lower(), "answer": j["answer"], "correct": j["correct"], "tree": j["tree"],
                              "tree_problems": j["tree_problems"], "explanation": j["explanation"],
                              "facts_used_not_allowed": j.get("facts_used_not_allowed", []),
                              "compare": compare_tree(j["tree"], self.sc)}
        return trees

    # -- one call that checks every step of every tree, and every explanation against its tree
    def check_trees(self, trees):
        facts = all_facts(self.sc)
        blocks = []
        for name, t in trees.items():
            concl = {x["id"]: x["conclusion"] for x in t["tree"]}
            lines = [f'Tree "{name}"' + (" (reference proof: rate its explanation only)" if t["kind"] == "gold" else "")]
            for x in t["tree"]:
                prem = "; ".join(f'[{p}: "{facts.get(p) or concl.get(p, p)}"]' for p in x["premises"]) or "(none)"
                lines.append(f'  {x["id"]}: from {prem} it concludes: "{x["conclusion"]}"')
            lines.append(f'  Explanation: "{t["explanation"] or "(none)"}"')
            blocks.append("\n".join(lines))
        user = ("Below are proof trees for the question, each followed by a plain-English explanation.\n"
                f"Question: {self.sc['question']}\nPossible answers:\n{options_text(self.sc)}\n\n" + "\n\n".join(blocks) +
                "\n\nFor every step (except in reference proofs): valid = the conclusion follows from the premises, "
                "allowing only obvious common-sense links; anything more is not valid.\n"
                "For every explanation: faithful = it says nothing its tree does not support; readable = connected, "
                "natural English that someone who has not seen the facts could follow.\n"
                'Reply with ONLY: {"steps": [{"tree": "<name>", "step": "S1", "valid": true, "note": "a few words"}], '
                '"explanations": [{"tree": "<name>", "faithful": true, "readable": true, "note": "a few words"}]}')
        raw, err = self.ask(self.judge_model, CHECK_TREES_SYSTEM, user, lambda: self.mock_check_trees(trees),
                            max_tokens=3000)
        obj = parse_json(raw) or {}
        lower = {k.lower(): k for k in trees}
        for x in obj.get("steps") or []:
            if not isinstance(x, dict):
                continue
            t = trees.get(lower.get(_s(x.get("tree")).lower(), ""))
            if t and t["kind"] != "gold":
                for st_ in t["tree"]:
                    if st_["id"] == _s(x.get("step")).upper().replace(" ", ""):
                        st_["valid"] = bool(x.get("valid")) if isinstance(x.get("valid"), bool) else None
                        st_["check_note"] = _s(x.get("note"))
        for x in obj.get("explanations") or []:
            if not isinstance(x, dict):
                continue
            t = trees.get(lower.get(_s(x.get("tree")).lower(), ""))
            if t:
                t["explanation_check"] = {"faithful": x.get("faithful") if isinstance(x.get("faithful"), bool) else None,
                                          "readable": x.get("readable") if isinstance(x.get("readable"), bool) else None,
                                          "note": _s(x.get("note"))}
        return {"raw": raw, "error": err}

    # -- what could be proved, turn by turn: the public record and each agent's own view (no model call)
    def proof_progress(self):
        gold = gold_steps(self.sc)
        if not gold:
            return None

        def done(view):
            have, ok, changed = set(view), [], True
            while changed:
                changed = False
                for g in gold:
                    if g["id"] not in ok and all(x in have for x in g["premises"]):
                        have.add(g["id"])
                        ok.append(g["id"])
                        changed = True
            return ok

        public, rows = set(self.public), []
        pub_full, view_full = None, {nm: None for nm in self.names}
        for t in [None] + self.table:
            if t and t["move"] == "assert" and t.get("fact_status") in ("new", "on table"):
                public.add(t["fact"])
            turn = t["turn"] if t else 0
            row = {"turn": turn, "public": len(done(public))}
            for a in self.agents:
                row[a["name"]] = len(done(public | set(a["facts"])))
                if row[a["name"]] == len(gold) and view_full[a["name"]] is None:
                    view_full[a["name"]] = turn
            if row["public"] == len(gold) and pub_full is None:
                pub_full = turn
            rows.append(row)
        T = len(self.table)
        held = {nm: (max(0, (pub_full if pub_full is not None else T) - v) if v is not None else 0)
                for nm, v in view_full.items()}
        return {"gold_steps": len(gold), "rows": rows, "public_complete_at": pub_full,
                "full_proof_in_own_view_from": view_full, "turns_held_privately": held,
                "public_steps_at_end": rows[-1]["public"]}

    def finish(self, run_pooled=True, on_status=lambda s: None):
        """Everything after the dialogue. The app and the batch runner both call this."""
        on_status("The judge is reading the public record...")
        judge = self.judge(pooled=False)
        pooled = None
        if run_pooled:
            on_status("Running the pooled baseline...")
            pooled = self.judge(pooled=True)
        reading = None
        if self.settings["check_readings"]:
            on_status("The judge is checking how each argument was read...")
            reading = self.check_readings()
        closing = {}
        if self.settings.get("closing_round", True):
            on_status("Closing round: each agent writes its own tree and explanation...")
            closing = self.closing()
        gold_text = None
        if self.settings.get("explain_gold", True):
            on_status("Putting the gold tree into plain English...")
            gold_text = self.verbalise_gold()
        self.trees = self.collect_trees(judge, pooled, closing, gold_text)
        tree_check = None
        if self.settings.get("check_trees", True) and self.trees:
            on_status("Checking every step and every explanation...")
            tree_check = self.check_trees(self.trees)
        self.progress = self.proof_progress()
        return {"judge": judge, "pooled": pooled, "reading_check": reading, "closing": closing,
                "gold_plain": gold_text, "trees": self.trees, "tree_check": tree_check,
                "proof_progress": self.progress, "graph": self.declared_graph(self.settings["support_rule"])}

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
                f"  What {o['agent']} privately meant: {part_text(o['reconstruction']['mine'], self.fact_text)}\n"
                f"  How {t['agent']} read it at turn {t['turn']}: "
                f"{part_text(t['reconstruction']['reading_of_target'], self.fact_text)}")
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
            fact = random.choice(unshared)
            live = [t for t in others if not t.get("retracted_at")]
            rule = "If (mock) this holds for the case, then the claim follows"
            if live and random.random() < 0.6:
                tgt = random.choice(live)
                target = tgt["turn"]
                reading = {"premises": [tgt["fact"]], "assumption": rule,
                           "conclusion": "(mock) the speaker's conclusion as I read it"}
            else:
                target, reading = random.choice(self.letters), None
            mine_part = {"premises": [fact] + ([f"turn {target}"] if reading else []), "assumption": rule,
                         "conclusion": "(mock) my conclusion in a full sentence"}
            if not retry and random.random() < 0.25:  # a typical flaw: a topic as premise, an assumption that is a fact
                mine_part = {"premises": ["(mock topic)"], "assumption": self.fact_text[fact], "conclusion": "Topic"}
            reply = {"move": "assert", "target": target, "reading_of_target": reading, "fact": fact,
                     "relation": random.choice(["attack", "support"]),
                     "aim": random.choice(["premise", "assumption", "conclusion"]), "mine": mine_part,
                     "say": "(mock) This matters.",
                     "answers_goal": random.choice(open_goals)["id"] if open_goals and random.random() < 0.5 else None,
                     "opens": ["(mock) a premise this still needs"] if random.random() < 0.3 else []}
            if random.random() < 0.2:  # the nested format some models use
                reply = {"move": reply}
            return json.dumps(reply)
        return json.dumps({"move": "pass"})

    def mock_tree_reply(self, allowed):
        ids = sorted(allowed)
        random.shuffle(ids)
        steps = []
        if ids:
            steps.append({"id": "S1", "premises": ids[:2], "conclusion": "(mock) a first conclusion in full."})
            if len(ids) > 2:
                steps.append({"id": "S2", "premises": ["S1", ids[2]], "conclusion": "(mock) so the answer follows."})
        return json.dumps({"answer": random.choice(self.letters), "tree": steps,
                           "explanation": "(mock) Because of the first point, the second follows, so the answer holds.",
                           "facts_used": [x for st_ in steps for x in st_["premises"] if not x.startswith("S")]})

    def mock_check_trees(self, trees):
        return json.dumps({
            "steps": [{"tree": nm, "step": x["id"], "valid": random.random() < 0.7, "note": "(mock)"}
                      for nm, t in trees.items() if t["kind"] != "gold" for x in t["tree"]],
            "explanations": [{"tree": nm, "faithful": random.random() < 0.7, "readable": random.random() < 0.8,
                              "note": "(mock)"} for nm in trees]})

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
        by_strategy = {k: sum(1 for t in self.table if t.get("strategy") == k) for k in STRATEGIES}
        switches = sum(1 for a, b in zip(self.rounds, self.rounds[1:])
                       for nme in self.names if a["assignments"][nme] != b["assignments"][nme])
        after_lc = ([t for t in self.table if t["turn"] > self.last_call_at] if self.last_call_at else [])
        return {
            "strategy": condition_name(self.strategy, self.policy),
            "rounds": len(self.rounds),
            "strategy_switches": switches,
            "turns_by_strategy": by_strategy,
            "orchestrator_calls": self.orch_calls,
            "last_call_at": self.last_call_at,
            "new_facts_after_last_call": sum(1 for t in after_lc if t["move"] == "assert" and t["fact_status"] == "new"),
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
            **self.tree_metrics(asserts),
        }

    def tree_metrics(self, asserts):
        tr = getattr(self, "trees", None) or {}
        pg = getattr(self, "progress", None) or {}
        ag = [v for v in tr.values() if v["kind"] == "agent"]
        steps = lambda vs: [x for v in vs for x in v["tree"]]
        cmp_ = lambda key, f: (tr.get(key) or {}).get("compare", {}) and tr[key]["compare"].get(f)
        checked = [v["explanation_check"] for v in tr.values() if v.get("explanation_check")]
        j = [tr["Judge"]] if "Judge" in tr else []
        return {
            "recon_flawed": sum(1 for t in asserts if t.get("recon_problems")),
            "moves_sent_back": sum(1 for t in self.table if t.get("sent_back")),
            "agents_correct": sum(1 for v in ag if v["correct"]) if ag else None,
            "agents_with_all_gold_leaves": sum(1 for v in ag if v["compare"] and not v["compare"]["missing_from_gold"])
            if ag else None,
            "agents_unsaid_facts_used": sum(len(v["unsaid_own_facts_used"]) for v in ag) if ag else None,
            "agents_steps_valid": sum(1 for x in steps(ag) if x.get("valid") is True) if ag else None,
            "agents_steps_total": len(steps(ag)) if ag else None,
            "judge_tree_leaf_recall": cmp_("Judge", "leaf_recall"),
            "judge_tree_gold_steps_matched": cmp_("Judge", "gold_steps_matched"),
            "judge_tree_problems": len(j[0]["tree_problems"]) if j else None,
            "judge_steps_valid": sum(1 for x in steps(j) if x.get("valid") is True) if j else None,
            "judge_steps_total": len(steps(j)) if j else None,
            "explanations_checked": len(checked),
            "explanations_faithful": sum(1 for c in checked if c.get("faithful") is True),
            "explanations_readable": sum(1 for c in checked if c.get("readable") is True),
            "gold_steps_total": pg.get("gold_steps"),
            "gold_steps_public_at_end": pg.get("public_steps_at_end"),
            "public_proof_complete_at": pg.get("public_complete_at"),
            "max_turns_proof_held_privately": max((pg.get("turns_held_privately") or {}).values(), default=None),
        }


# ----------------------------------------------------------------- display

BADGE = {"match": ":green[match]", "partial": ":orange[partial]", "mismatch": ":red[mismatch]"}


def show_recon(rec, facts=None):
    if not rec:
        st.caption("No reconstruction recorded.")
        return
    if rec.get("reading_of_target"):
        st.markdown(f"**Reads the target as:** {part_text(rec['reading_of_target'], facts)}")
    if rec.get("aim"):
        st.markdown(f"**Aims at:** {rec['aim']}")
    st.markdown(f"**Own argument:** {part_text(rec.get('mine'), facts)}")


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
        if entry.get("round"):
            head += f" · *{strategy_name(entry.get('strategy'))}*"
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
            if entry.get("recon_problems"):
                st.caption(":orange[Reconstruction: " + "; ".join(entry["recon_problems"]) + "]")
            with st.expander(f"🔒 Private reconstruction ({seen})"):
                show_recon(entry["reconstruction"], facts)
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
                show_recon(entry["revealed"], facts)
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
        sb = entry.get("sent_back")
        if sb:
            st.caption(("Sent back once to fix: " if sb["replaced"] else "Sent back once, but the fix was unreadable: ")
                       + "; ".join(sb["problems"]))
        with st.expander("raw model output"):
            st.code(entry.get("raw") or "(empty)", language=None)


def tree_dot(t, run):
    sc = run["scenario"]
    names = [a["name"] for a in sc["agents"]]
    holder = {f: i for i, a in enumerate(sc["agents"]) for f in a["facts"]}
    facts = all_facts(sc)
    said = run.get("first_shared") or {}
    public_ids = set(sc.get("public_facts", {}))
    steps = t["tree"]
    leaves = tree_leaves(steps)
    lines = ['digraph G {', '  graph [rankdir=BT, bgcolor="white", nodesep=0.3, ranksep=0.45];',
             '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, penwidth=1.5];',
             '  edge [color="#52606d"];']
    for f in leaves:
        if f in public_ids:
            who, fill, tag = "public", PUBLIC_FILL, "public"
        elif f in holder:
            i = holder[f]
            who, fill = names[i], HOLDER_FILL[i % len(HOLDER_FILL)]
            tag = f"said, turn {said[f]}" if f in said else "never said"
        else:
            who, fill, tag = "?", PUBLIC_FILL, "unknown"
        out_loud = f in public_ids or f in said
        lines.append(f'  "{f}" [label="{_dot_text(f"{f} ({who}, {tag})", 60)}\\n{_dot_text(facts.get(f, ""), 30)}", '
                     f'fillcolor="{fill}", color="{"#1f2933" if out_loud else "#8a96a3"}", '
                     f'style="rounded,filled{"" if out_loud else ",dashed"}"];')
    avail = set(steps_done(steps, public_ids | set(said))) if t["kind"] == "gold" else set()
    root = steps[-1]["id"] if steps else None
    for x in steps:
        if t["kind"] == "gold":
            ok = x["id"] in avail
            fill, border, dash = ("#d3f2da", "#1e7b34", "") if ok else ("white", "#8a96a3", ",dashed")
        else:
            v = x.get("valid")
            fill, border, dash = (("#d3f2da", "#1e7b34", "") if v is True else
                                  ("#f9d6d5", "#b42318", "") if v is False else ("white", "#6b6b6b", ""))
        pw = 2.6 if x["id"] == root else 1.5
        lines.append(f'  "{x["id"]}" [label="{_dot_text(x["id"] + ": " + (x["conclusion"] or "(no conclusion)"), 36)}", '
                     f'fillcolor="{fill}", color="{border}", penwidth={pw}, style="rounded,filled{dash}"];')
        for prem in x["premises"]:
            lines.append(f'  "{prem}" -> "{x["id"]}";')
    lines.append("}")
    return "\n".join(lines)


def yes_no(v):
    return {True: ":green[yes]", False: ":red[no]"}.get(v, "not checked")


def show_trees(run):
    trees = run.get("trees") or {}
    if not trees:
        return
    st.subheader("Trees and explanations")
    st.caption("After the dialogue each agent privately writes its own answer as a proof tree plus a plain-English "
               "explanation, from what it could see (its own facts and everything public). The judge does the same "
               "from the public record only, the pooled baseline from all facts, and the gold tree is put into plain "
               "English. Leaves are coloured by who held the fact; a dashed border means it was never said. Steps: "
               "green = checked valid, red = not valid, white = not checked. In the gold tree, green = every premise "
               "was public by the end.")
    rows = []
    for nm, t in trees.items():
        c = t.get("compare") or {}
        ec = t.get("explanation_check") or {}
        rows.append({
            "tree": nm, "answer": (t["answer"] + (" (right)" if t["correct"] else " (wrong)")) if t["answer"] else "-",
            "gold facts used": f"{c['leaves_in_gold']} / {c['gold_leaves']}" if c else "",
            "other facts used": ", ".join(c.get("not_in_gold", [])) if c else "",
            "gold steps matched": f"{c['gold_steps_matched']} / {c['gold_steps']}" if c else "",
            "steps valid": "" if t["kind"] == "gold" else
            f"{sum(1 for x in t['tree'] if x.get('valid') is True)} / {len(t['tree'])}",
            "own facts used but never said": ", ".join(t.get("unsaid_own_facts_used") or []),
            "explanation faithful": {True: "yes", False: "no"}.get(ec.get("faithful"), ""),
            "explanation readable": {True: "yes", False: "no"}.get(ec.get("readable"), ""),
            "tree problems": len(t["tree_problems"])})
    st.dataframe(rows, hide_index=True, width="stretch")
    for tab, (nm, t) in zip(st.tabs(list(trees)), trees.items()):
        with tab:
            st.markdown(f"> {t['explanation']}" if t["explanation"] else "*No explanation.*")
            ec = t.get("explanation_check")
            if ec:
                st.caption(f"Faithful to its tree: {yes_no(ec['faithful'])} · readable: {yes_no(ec['readable'])}"
                           + (f" · {ec['note']}" if ec.get("note") else ""))
            if t["tree"]:
                st.graphviz_chart(tree_dot(t, run), width="stretch")
            for x in t["tree"]:
                if x.get("valid") is False and x.get("check_note"):
                    st.caption(f":red[{x['id']} not valid:] {x['check_note']}")
            for pr in t["tree_problems"]:
                st.caption(f":orange[{pr}]")
            if t.get("facts_used_not_allowed"):
                st.caption(":red[Cited facts it could not have seen: " + ", ".join(t["facts_used_not_allowed"]) + "]")
    pg = run.get("proof_progress")
    if pg:
        st.markdown("**What could be proved, turn by turn.** Gold steps complete in the public record, and in each "
                    "agent's own view (its own facts plus everything public). No model involved.")
        for nm, v in pg["full_proof_in_own_view_from"].items():
            if v is not None:
                held = pg["turns_held_privately"][nm]
                st.caption(f"{nm} could complete the whole gold proof in its own view from turn {v}"
                           + (f"; it stayed incomplete in public for {held} more turn(s)." if held else "."))
        pub = pg["public_complete_at"]
        st.caption(f"Public record: {pg['public_steps_at_end']} of {pg['gold_steps']} gold steps complete at the end"
                   + (f" (complete from turn {pub})." if pub is not None else "."))
        with st.expander("Turn by turn"):
            st.dataframe(pg["rows"], hide_index=True, width="stretch")


def show_round(rec):
    who = ", ".join(f"{a} → **{strategy_name(s)}**" for a, s in rec["assignments"].items())
    tag = " · last call" if rec["trigger"] == "last_call" else ""
    st.info(f"**Round {rec['round']}** ({POLICIES.get(rec['policy'], {}).get('label', rec['policy'])}{tag}): "
            f"{who}  \n*{rec['reason'] or 'no reason given'}*", icon="🧭")


def show_orchestrator(run):
    rounds = run.get("rounds") or []
    if not rounds:
        return
    m = run["metrics"]
    st.subheader("Orchestrator")
    st.caption("Who got which strategy in each round, and why. The orchestrator reads only the public record"
               + (" and, in this run, the private facts (oracle setting)." if run["settings"].get("orchestrator_sees_private")
                  else ".") + " It never sees the gold answer, and the agents never see its reasons.")
    c = st.columns(4)
    c[0].metric("Rounds", m["rounds"])
    c[1].metric("Strategy switches", m["strategy_switches"])
    c[2].metric("Orchestrator calls", m["orchestrator_calls"])
    c[3].metric("Turns per strategy", " · ".join(f"{k[:4]} {v}" for k, v in m["turns_by_strategy"].items() if v) or "-")
    rows = []
    for r in rounds:
        sg = r["signals"]
        last = sg.get("last_round") or {}
        rows.append({"round": r["round"], "from turn": r["start_turn"], "trigger": r["trigger"],
                     "assignments": ", ".join(f"{a}: {s}" for a, s in r["assignments"].items()),
                     "why": r["reason"], "facts public": sg["facts_public"],
                     "new facts last round": last.get("new_facts"), "open goals": ", ".join(sg["open_goals"]),
                     "answers with support": ", ".join(sg["answers_with_support"]),
                     "favourite": sg["favourite"] or ""})
    st.dataframe(rows, width="stretch", hide_index=True)
    raws = [r for r in rounds if r.get("raw") or r.get("error")]
    if raws:
        with st.expander("Raw orchestrator replies"):
            for r in raws:
                st.markdown(f"**Round {r['round']}**" + (f" · :red[{r['error']}]" if r.get("error") else ""))
                st.code(r.get("raw") or "(empty)", language=None)


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
    starts = {r["start_turn"]: r for r in run.get("rounds") or []}
    for entry in run["transcript"]:
        if entry["turn"] in starts:
            show_round(starts[entry["turn"]])
        show_turn(entry, avatar[entry["agent"]], facts)
        for note in run.get("notes") or []:
            if note["after_turn"] == entry["turn"]:
                st.warning(note["text"], icon="📣")

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
    gp = (run.get("gold_plain") or {}).get("explanation")
    if gp:
        st.markdown(f"**Gold explanation (the gold tree in plain English):** {gp}")
    if sc.get("gold_explanation"):
        with st.expander("Gold explanation as written out by the converter (long form)"):
            st.markdown(sc["gold_explanation"])

    show_trees(run)
    show_orchestrator(run)
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

    collections = load_collections()
    uploaded = st.file_uploader("Upload a scenario (.json)", type="json")
    if uploaded is not None:
        try:
            up = json.load(uploaded)
            collections = {"Uploaded": {f"(uploaded) {up.get('title', uploaded.name)}": up}, **collections}
        except Exception as e:
            st.error(f"Could not read that file: {e}")
    if not collections:
        st.warning("No scenarios found. Upload one.")
        st.stop()
    collection = st.selectbox("Collection", list(collections),
                              format_func=lambda c: f"{c} ({len(collections[c])})")
    scenarios = collections[collection]
    chosen = st.selectbox("Scenario", list(scenarios), help="Click and type to search, e.g. part of the question.")
    with st.expander("Scenario file format"):
        st.code(FORMAT_HELP, language="json")

    strategy = st.radio("Strategy", list(STRATEGIES) + [ORCHESTRATED],
                        format_func=lambda k: "5. Assigned each round by an orchestrator" if k == ORCHESTRATED
                        else STRATEGIES[k]["label"],
                        captions=[STRATEGIES[k]["caption"] for k in STRATEGIES]
                        + ["An orchestrator picks a strategy for each agent at the start of every round."],
                        help="Every agent gets the same moves, information, turn limit and stopping rule. "
                             "Only the strategy paragraph in their instructions changes.")
    policy, oracle = None, False
    if strategy == ORCHESTRATED:
        policy = st.selectbox("How the orchestrator decides", list(POLICIES), format_func=lambda p: POLICIES[p]["label"])
        st.caption(POLICIES[policy]["caption"])
        if policy == "llm":
            oracle = st.checkbox("Orchestrator also sees the private facts (oracle, not realistic)", value=False,
                                 help="An upper bound: how good can strategy assignment get with full information? "
                                      "It still never sees the gold answer.")
    last_call = st.checkbox("Last call: when everyone passes, one more round before ending", value=False,
                            help="Applies to every strategy. Under the orchestrator the extra round gets its own "
                                 "assignment. Off = the v0.5 stopping rule.")
    max_turns = st.slider("Max turns", 4, 80, 30, help="One fact per move, so dialogues run longer.")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    show_targets = st.checkbox("Agents see what each move targets", value=True,
                               help="Off: others hear only the fact and the spoken line, not which "
                                    "argument it attacks or supports.")
    check_readings = st.checkbox("Check readings at the end (one judge call)", value=True,
                                 help="The judge compares each responder's reading with what the "
                                      "speaker privately meant.")
    run_pooled = st.checkbox("Also run pooled baseline (judge sees all facts)", value=True)
    fix_moves = st.checkbox("Send a flawed argument back once to be fixed", value=True,
                            help="Missing target, a fact the agent doesn't hold, or a reconstruction that breaks the "
                                 "format (premises that aren't IDs, an assumption that isn't an If-then rule or just "
                                 "repeats a premise, a conclusion that isn't a sentence).")
    with st.expander("After the dialogue"):
        closing_round = st.checkbox("Closing round: each agent writes its own tree and explanation (private)",
                                    value=True, help="One call per agent.")
        explain_gold = st.checkbox("Put the gold tree into plain English (one judge call)", value=True)
        check_trees = st.checkbox("Check every step and explanation (one judge call)", value=True)
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

if strategy == ORCHESTRATED:
    st.markdown(f"**Strategy:** assigned each round by the orchestrator · *{POLICIES[policy]['label']}*: "
                f"{POLICIES[policy]['caption']}")
    if policy == "rules":
        with st.expander("The rules, in order (the first that applies decides the round)"):
            st.dataframe([{"#": i + 1, "when": c, "everyone gets": strategy_name(k), "why": w}
                          for i, (c, k, w) in enumerate(RULE_TABLE)], hide_index=True, width="stretch")
    shown = ORCH_LEAD + "<the paragraph of the strategy assigned for this round>"
else:
    st.markdown(f"**Strategy:** {STRATEGIES[strategy]['label']}: *{STRATEGIES[strategy]['question']}*")
    shown = STRATEGIES[strategy]["text"]
with st.expander("What the agents are told"):
    st.caption("The same rules for every strategy (shown here for the first agent), ending with the strategy paragraph.")
    st.code(RULES.format(name="<agent>", role="", n="<n>", others="<the other agents>", strategy=shown), language=None)

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
orchestrator_model = None
if strategy == ORCHESTRATED and policy == "llm":
    orchestrator_model = model_picker("Orchestrator model", f"orch_{chosen}", 0)
with st.expander("Show gold answer and scenario notes"):
    st.markdown(f"**{sc['answer']}**. {sc['options'][sc['answer']]}")
    if sc.get("gold_explanation"):
        st.markdown(sc["gold_explanation"])
    if sc.get("notes"):
        st.caption(sc["notes"])

if st.button(f"▶ Run dialogue ({condition_label(strategy, policy)})", type="primary", width="stretch"):
    if not mock and not api_key:
        st.error("Add an OpenRouter key in the sidebar, or switch on mock mode.")
        st.stop()
    settings = {"max_turns": max_turns, "temperature": temperature, "mock": mock,
                "show_targets": show_targets, "check_readings": check_readings, "support_rule": support_rule,
                "strategy": strategy, "policy": policy, "orchestrator_model": orchestrator_model,
                "orchestrator_sees_private": oracle, "last_call": last_call, "seed": None,
                "fix_moves": fix_moves, "closing_round": closing_round, "explain_gold": explain_gold,
                "check_trees": check_trees}
    ref = Referee(sc, models, judge_model, settings, api_key, models_available)
    avatar = {a["name"]: AVATARS[i % len(AVATARS)] for i, a in enumerate(sc["agents"])}
    facts = all_facts(sc)
    status = st.empty()
    live = st.container()

    def on_turn(entry):
        with live:
            show_turn(entry, avatar[entry["agent"]], facts)
            for note in ref.notes:
                if note["after_turn"] == entry["turn"]:
                    st.warning(note["text"], icon="📣")

    def on_round(rec):
        with live:
            show_round(rec)

    ended = ref.run(on_turn, lambda s: status.info(s), on_round)
    res = ref.finish(run_pooled, lambda s: status.info(s))
    status.empty()
    st.session_state.last_run = {
        "app": APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenario": sc,
        "models": models,
        "judge_model": judge_model,
        "settings": settings,
        "ended": ended,
        "strategy": {"key": condition_name(strategy, policy), "label": condition_label(strategy, policy),
                     "text": "" if strategy == ORCHESTRATED else STRATEGIES[strategy]["text"]},
        "rounds": ref.rounds,
        "notes": ref.notes,
        "transcript": ref.table,
        "goals": ref.goals,
        "first_shared": ref.first_shared,
        "judge": res["judge"],
        "pooled": res["pooled"],
        "reading_check": res["reading_check"],
        "closing": res["closing"],
        "gold_plain": res["gold_plain"],
        "trees": res["trees"],
        "tree_check": res["tree_check"],
        "proof_progress": res["proof_progress"],
        "graph": res["graph"],
        "metrics": ref.metrics(),
        "usage": ref.usage,
    }
    st.rerun()

if st.session_state.get("last_run", {}).get("app") not in (None, APP_VERSION):
    del st.session_state["last_run"]  # a run from an older version of the app
if "last_run" in st.session_state:
    st.divider()
    show_results(st.session_state.last_run)
