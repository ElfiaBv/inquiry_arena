"""Inquiry Arena: LLM agents with split knowledge play an inquiry dialogue.

The referee (this program) only deals the facts, keeps turn order and keeps the
record. It never reveals anyone's private notes: if you want to know what a move
meant, you ask its author, and the author answers in their own words.

Moves
  assert        state ONE fact with a stance: attack or support a target (an earlier
                argument or an answer), or offer it as a premise (a building block that
                doesn't argue for or against anything yet). Plus a short spoken line.
  ask_stance    ask the author of an earlier argument what it was aimed at.
  ask_argument  ask the author to spell out the argument behind it.
  explain       explain one of your own arguments (the asked agent must do this next).
  retract       withdraw one of your own arguments.
  pass

Private state, updated by each agent on each of its turns
  - for its own arguments: what it meant (stance, target, full argument)
  - for everyone else's arguments: its READING: a reconstruction (premises, unstated
    assumption, conclusion) and what it thinks the move does: attack, support, premise,
    irrelevant or unclear, and against/for what
  - its leaning: the answer it currently favours
From these the app builds each agent's internal argument graph at every turn and
labels it with grounded semantics, next to the graph the speakers intended.
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

APP_VERSION = "inquiry-arena v0.4.1"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
AVATARS = ["🟦", "🟧", "🟩", "🟪", "🟥", "🟨", "⬛", "⬜"]
INTENDED = "What the speakers meant"

PREFERRED_MODELS = [
    "openai/gpt-4o-mini", "openai/gpt-4.1-mini", "openai/gpt-5-mini",
    "anthropic/claude-3.5-haiku", "anthropic/claude-haiku-4.5",
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen-2.5-72b-instruct", "qwen/qwen3-32b",
    "google/gemini-2.0-flash-001", "google/gemini-2.5-flash",
    "mistralai/mistral-small-3.2-24b-instruct",
]
FALLBACK_MODEL = "openai/gpt-4o-mini"

SPEAKER_STANCES = ("attack", "support", "premise")
READING_STANCES = ("attack", "support", "premise", "irrelevant", "unclear")

RULES = """You are <<NAME>><<ROLE>>, one of <<N>> AI agents working together on an inquiry.
The group's goal is to find the correct answer to the question, with a short explanation.
The other agents are: <<OTHERS>>.

How it works:
- Everyone sees the public facts and the public conversation. Each agent also has PRIVATE facts;
  others only learn them if you state them.
- People talk briefly: each argument states ONE fact and says a little about it, without spelling
  out every assumption. Others have to work out what you mean, and you have to work out what they mean.
- You keep PRIVATE notes that nobody else sees:
  * for each of your own arguments: what you meant (its stance, its target and your full argument);
  * for each argument by someone else: your READING of it: your reconstruction (premises, unstated
    assumption, conclusion) and what you think it is doing.
- What a move can be doing (its stance):
  "attack": it argues against a target (an earlier argument, by turn number, or an answer letter);
  "support": it argues for a target;
  "premise": it doesn't argue for or against anything yet; it is offered as a building block that
             may matter together with other facts;
  "irrelevant": it has no bearing on the question (only when reading others' moves);
  "unclear": you can't tell what it is doing (only when reading others' moves; consider asking).
- If you can't tell how a move connects, or someone seems to attack your argument in a way you don't
  recognise, ask its author. They must answer on the next turn, in their own words.
- Be honest: use only your own facts or facts already on the table. Never invent facts or IDs.
  Don't repeat an argument you have already made.

Reply with ONLY one flat JSON object. Every reply contains "readings", "leaning" and the fields of one
move, all at the top level (do not nest the move inside another object). A complete example reply:
{"readings": [{"turn": 2, "stance": "attack", "target": 1, "premises": "...", "assumption": "...",
  "conclusion": "..."}], "leaning": "undecided", "move": "assert", "fact": "X1", "stance": "support",
 "target": "A", "aim": "conclusion", "say": "...", "mine": {"premises": "...", "assumption": "...",
 "conclusion": "..."}}

The stance must match your conclusion: if your argument concludes that an answer (or an earlier argument)
is wrong, its stance is "attack" on it, not "support".

"readings": add a reading for every argument by others that you have not read yet, and revise any
reading that has changed (for example after an explanation). List only new or changed readings:
  [{"turn": <n>, "stance": "attack" | "support" | "premise" | "irrelevant" | "unclear",
    "target": <turn number or answer letter, or null>,
    "premises": "...", "assumption": "...", "conclusion": "..."}]
"leaning": the answer letter you currently favour, or "undecided".

The move is one of:
1. Make an argument with exactly ONE fact:
   {"move": "assert", "fact": "ID", "stance": "attack" | "support" | "premise",
    "target": <turn number or answer letter; null for a premise>,
    "aim": "premise" | "assumption" | "conclusion"  (which part of the target you attack or support),
    "say": "what you say out loud, one or two sentences",
    "mine": {"premises": "...", "assumption": "...", "conclusion": "..."}}
2. Ask the author of an earlier argument what it was aimed at:
   {"move": "ask_stance", "turn": <n>, "say": "your question"}
3. Ask the author of an earlier argument to spell out the argument behind it:
   {"move": "ask_argument", "turn": <n>, "say": "your question"}
4. Explain one of your own arguments (you must do this when asked; you may also do it unasked):
   {"move": "explain", "turn": <n>, "stance": "attack" | "support" | "premise", "target": <...>,
    "argument": {"premises": "...", "assumption": "...", "conclusion": "..."},
    "say": "your explanation, in your own words"}
5. Withdraw one of your own arguments: {"move": "retract", "turn": <n>, "say": "why"}
6. Nothing useful to add right now: {"move": "pass"}

The dialogue ends when every agent passes in a row."""

JUDGE_SYSTEM = ("You are a neutral judge. You know only what is written in the message. "
                "Reply with ONLY one JSON object, nothing else.")
CHECK_SYSTEM = ("You compare reconstructions of short arguments. You know only what is written in the message. "
                "Reply with ONLY one JSON object, nothing else.")

LABEL_STYLE = {"IN": ("#d3f2da", "#1e7b34"), "OUT": ("#f9d6d5", "#b42318"), "UNDEC": ("#e7e7e7", "#6b6b6b")}

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
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_models():
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


MOVE_NAMES = {"assert", "ask_stance", "stance?", "ask_argument", "show", "explain", "retract", "withdraw", "pass"}

FORMAT_REMINDER = """

YOUR PREVIOUS REPLY COULD NOT BE READ. Reply with ONE flat JSON object: the move's fields sit at the top
level, next to "readings" and "leaning" (do not nest them inside "move"). For example:
{"readings": [], "leaning": "undecided", "move": "assert", "fact": "X1", "stance": "support", "target": "A",
 "aim": "conclusion", "say": "...", "mine": {"premises": "...", "assumption": "...", "conclusion": "..."}}"""


def read_reply(raw):
    """Parse an agent's reply. Accepts the flat format and the common nested variant
    {"readings": ..., "leaning": ..., "move": {"move": "assert", ...}}. Returns (obj, move_name)."""
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
    name = obj.get("move") or obj.get("action") or obj.get("type") or ""
    return obj, _s(name).lower()


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


def parse_stance(raw):
    r = _s(raw).lower()
    if r.startswith("att") or r in ("rebut", "undercut", "undermine"):
        return "attack"
    if r.startswith("sup"):
        return "support"
    if r.startswith("prem") or r in ("neutral", "background", "information", "info"):
        return "premise"
    if r.startswith("irr"):
        return "irrelevant"
    if r.startswith("uncl") or r in ("unknown", "unsure", "?"):
        return "unclear"
    return ""


def norm_part(x):
    if isinstance(x, dict):
        return {k: _s(x.get(k)) for k in ("premises", "assumption", "conclusion")}
    if x is None or _s(x).lower() in ("", "null", "none", "n/a"):
        return None
    return {"premises": "", "assumption": "", "conclusion": _s(x)}


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


def stance_text(stance, kind, target):
    if stance in ("attack", "support"):
        verb = "attacks" if stance == "attack" else "supports"
        if kind == "option":
            return f"{verb} answer {target}"
        if kind == "turn":
            return f"{verb} turn {target}"
        return f"{verb} (no clear target)"
    if stance == "premise":
        return "offers a premise"
    if stance == "irrelevant":
        return "irrelevant"
    if stance == "unclear":
        return "unclear"
    return "no stance given"


def same_stance(a, b):
    """Do two (stance, kind, target) triples say the same thing?"""
    if a[0] != b[0]:
        return False
    if a[0] in ("attack", "support"):
        return a[1] == b[1] and a[2] == b[2]
    return True


def reading_at(readings, agent, arg_turn, t):
    """The reading `agent` held of the argument made at `arg_turn`, as of turn t."""
    hist = (readings.get(agent) or {}).get(arg_turn) or (readings.get(agent) or {}).get(str(arg_turn)) or []
    cur = None
    for r in hist:
        if r["as_of"] <= t:
            cur = r
    return cur


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


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ----------------------------------------------------------------- argument graphs

def grounded_labels(node_ids, attacks):
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


def build_graph(args, attacks, supports, sc, support_rule):
    """args: list of node dicts; attacks/supports: [src, dst] with dst a node id or opt_X."""
    letters = list(sc["options"])
    nodes = [{"id": f"opt_{k}", "kind": "option", "by": "", "turn": None, "facts": [],
              "text": f"{k}. {sc['options'][k]}"} for k in letters] + list(args)
    exclusion = [[f"opt_{x}", f"opt_{y}"] for x in letters for y in letters if x != y]
    via_support = []
    if support_rule:
        for a, b in supports:
            if b.startswith("opt_"):
                via_support += [[a, f"opt_{y}"] for y in letters if f"opt_{y}" != b]
    all_attacks = attacks + exclusion + [e for e in via_support if e not in attacks]
    labels = grounded_labels([n["id"] for n in nodes], all_attacks)
    winners = [k for k in letters if labels[f"opt_{k}"] == "IN"]
    return {"nodes": nodes, "attacks": attacks, "supports": supports, "exclusion": exclusion,
            "support_rule": support_rule, "labels": labels,
            "verdict": winners[0] if len(winners) == 1 else ""}


def _dot_text(s, width=34):
    s = "\n".join(textwrap.wrap(str(s), width)) or " "
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def graph_dot(g, highlight=()):
    """highlight: edges [src, dst] drawn orange (readings that differ from what the speaker meant)."""
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
            head = f'Turn {n["turn"]} · {n["by"]}'
            facts = f'\\n[{", ".join(n["facts"])}]' if n["facts"] else ""
            style = ', style="rounded,filled,dashed"' if n.get("unread") else ""
            lines.append(f'  "{n["id"]}" [label="{_dot_text(head, 60)}\\n{_dot_text(n["text"])}{facts}", '
                         f'fillcolor="{fill}", color="{border}"{style}];')
    if opts:
        lines.append(f'  {{ rank=same; {"; ".join(opts)}; }}')
        if len(opts) > 1:
            lines.append(f'  {" -> ".join(opts)} [style=invis];')
    hl = {tuple(e) for e in highlight}
    orange = "#c2410c"
    for x, y in g["attacks"]:
        if (x, y) in hl:
            lines.append(f'  "{x}" -> "{y}" [color="{orange}", penwidth=2.2];')
        else:
            lines.append(f'  "{x}" -> "{y}" [color="#b42318", penwidth=1.6];')
    for x, y in g["supports"]:
        color = orange if (x, y) in hl else "#1e7b34"
        lines.append(f'  "{x}" -> "{y}" [color="{color}", style=dashed, arrowhead=empty];')
    seen = set()
    for x, y in g["exclusion"]:
        if (y, x) not in seen:
            seen.add((x, y))
            lines.append(f'  "{x}" -> "{y}" [color="#9a9a9a", style=dotted, dir=none, constraint=false];')
    lines.append("}")
    return "\n".join(lines)


def compute_view(sc, table, readings, who, t, support_rule):
    """The argument graph as `who` sees it after turn t. who=None: what the speakers meant."""
    live = {}
    for e in table[:t]:
        if e["move"] != "assert" or e["fact_status"] in ("invalid", "missing"):
            continue
        if e.get("retracted_at") and e["retracted_at"] <= t:
            continue
        live[e["turn"]] = f"t{e['turn']}"
    args, attacks, supports, unread = [], [], [], []
    for e in table[:t]:
        if e["turn"] not in live:
            continue
        node = {"id": live[e["turn"]], "kind": "argument", "by": e["agent"], "turn": e["turn"],
                "facts": [e["fact"]], "text": e["say"] or ""}
        if who is None or e["agent"] == who:
            stance, kind, target = e["stance"], e["target_kind"], e["target"]
        else:
            r = reading_at(readings, who, e["turn"], t)
            if r is None:
                node["unread"] = True
                unread.append(e["turn"])
                args.append(node)
                continue
            stance, kind, target = r["stance"], r["target_kind"], r["target"]
        node["stance"] = stance_text(stance, kind, target)
        args.append(node)
        dst = f"opt_{target}" if kind == "option" else (live.get(target) if kind == "turn" else None)
        if stance in ("attack", "support") and dst and dst != live[e["turn"]]:
            (attacks if stance == "attack" else supports).append([live[e["turn"]], dst])
    g = build_graph(args, attacks, supports, sc, support_rule)
    g["unread"] = unread
    return g


def edge_set(g):
    return {("att", a, b) for a, b in g["attacks"]} | {("sup", a, b) for a, b in g["supports"]}


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
        self.readings = {n: {} for n in self.names}   # agent -> argument turn -> [readings over time]
        self.leanings = {n: [] for n in self.names}   # agent -> [(turn, letter)]
        self.questions = []                           # ask/explain bookkeeping
        self.pending = None
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "est_cost_usd": 0.0, "calls": 0}

    def entry_at(self, t):
        if isinstance(t, int) and 1 <= t <= len(self.table):
            return self.table[t - 1]
        return None

    def ask(self, model, system, user, mock_fn, max_tokens=2000):
        if self.mock:
            return mock_fn(), None
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=self.settings["temperature"], max_tokens=max_tokens,
                extra_headers={"X-OpenRouter-Title": "Inquiry Arena"})
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
        mv = t["move"]
        if mv == "assert":
            s = f"Turn {t['turn']} - {who}"
            if self.settings["show_targets"]:
                s += f" ({stance_text(t['stance'], t['target_kind'], t['target'])})"
            if t["fact_status"] in ("new", "on table"):
                s += f' states {t["fact"]}: "{self.fact_text[t["fact"]]}"'
            elif t["fact_status"] == "invalid":
                s += f' cites "{t["fact"]}", which is not a fact they hold'
            s += f'. Says: "{t["say"]}"' if t["say"] else "."
            if t.get("retracted_at"):
                s += f" [withdrawn by {who} at turn {t['retracted_at']}]"
            return s
        if mv in ("ask_stance", "ask_argument"):
            if t["problem"]:
                return f"Turn {t['turn']} - {who} asked about turn {t['turn_requested']}, but {t['problem']}."
            what = ("what turn {n} was aimed at" if mv == "ask_stance"
                    else "to spell out the argument behind turn {n}").format(n=t["turn_requested"])
            q = f' "{t["say"]}"' if t["say"] else ""
            return f"Turn {t['turn']} - {who} asks {t['owner']} {what}.{q}"
        if mv == "explain":
            if t["problem"]:
                return f"Turn {t['turn']} - {who} tried to explain turn {t['turn_requested']}, but {t['problem']}."
            s = f"Turn {t['turn']} - {who} explains turn {t['turn_requested']}"
            s += f': "{t["say"]}"' if t["say"] else "."
            if t["stance"]:
                s += f" Stance: {stance_text(t['stance'], t['target_kind'], t['target'])}."
            if t["argument"]:
                s += f" Argument: {part_text(t['argument'])}."
            return s
        if mv == "retract":
            if t["problem"]:
                return f"Turn {t['turn']} - {who} tried to withdraw turn {t['turn_requested']}, but {t['problem']}."
            r = f' "{t["say"]}"' if t["say"] else ""
            return f"Turn {t['turn']} - {who} withdraws turn {t['turn_requested']}.{r}"
        if mv in ("unreadable", "error"):
            return f"Turn {t['turn']} - {who} said nothing (no readable move)."
        return f"Turn {t['turn']} - {who} passes."

    def public_record(self):
        return "\n".join(self.public_line(t) for t in self.table) or "(nothing yet - you speak first)"

    def private_notes(self, agent):
        name = agent["name"]
        own, others = [], []
        for e in self.table:
            if e["move"] != "assert":
                continue
            tag = f"Turn {e['turn']} ({e['agent'] if e['agent'] != name else 'yours'}, {e['fact'] or 'no fact'})"
            if e.get("retracted_at"):
                tag += f" [withdrawn at turn {e['retracted_at']}]"
            if e["agent"] == name:
                own.append(f"{tag}: you meant it as: {stance_text(e['stance'], e['target_kind'], e['target'])}; "
                           f"{part_text(e['mine'])}")
            else:
                r = reading_at(self.readings, name, e["turn"], len(self.table) + 1)
                if r is None:
                    others.append(f"{tag}: NOT READ YET - add a reading")
                else:
                    others.append(f"{tag}: you read it as: {stance_text(r['stance'], r['target_kind'], r['target'])}; "
                                  f"{part_text(r)}")
        lean = self.leanings[name][-1][1] if self.leanings[name] else "undecided"
        return ("Your own arguments (what you meant):\n" + ("\n".join(own) or "(none yet)") +
                "\n\nYour readings of others' arguments:\n" + ("\n".join(others) or "(none yet)") +
                f"\n\nYour current leaning: {lean}")

    def agent_prompt(self, agent):
        q = ""
        p = self.pending
        if p and p["to"] == agent["name"]:
            want = ("its stance and target" if p["kind"] == "stance"
                    else "its stance, its target and the full argument behind it")
            q = (f"\n\nQUESTION FOR YOU: at turn {p['asked_at']}, {p['asked_by']} asked about your turn "
                 f"{p['turn']}: \"{p['say']}\". You must answer now with an explain move about turn {p['turn']}, "
                 f"giving {want}, in your own words.")
        return (f"Question: {self.sc['question']}\nPossible answers:\n{options_text(self.sc)}\n\n"
                f"Public facts (everyone sees these):\n{facts_text(self.sc.get('public_facts', {}))}\n\n"
                f"Your private facts (only you see these):\n{facts_text(agent['facts'])}\n\n"
                f"YOUR PRIVATE NOTES (only you see these):\n{self.private_notes(agent)}\n\n"
                f"Public conversation so far:\n{self.public_record()}{q}\n\n"
                "Your move. Reply with one JSON object only.")

    # -- private state updates
    def update_readings(self, name, obj, now):
        raw = obj.get("readings") if obj else None
        if isinstance(raw, dict):
            raw = [raw]
        out = []
        for r in raw if isinstance(raw, list) else []:
            if not isinstance(r, dict):
                continue
            t = parse_int(r.get("turn"))
            tgt = self.entry_at(t)
            if tgt is None or tgt["move"] != "assert" or tgt["agent"] == name:
                continue
            stance = parse_stance(r.get("stance")) or "unclear"
            kind, val = parse_target(r.get("target"), self.letters)
            if kind == "turn":
                te = self.entry_at(val)
                if te is None or te["move"] != "assert" or val == t:
                    kind = val = None
            if stance not in ("attack", "support"):
                kind = val = None
            rec = {"as_of": now, "stance": stance, "target_kind": kind, "target": val,
                   "premises": _s(r.get("premises")), "assumption": _s(r.get("assumption")),
                   "conclusion": _s(r.get("conclusion"))}
            self.readings[name].setdefault(t, []).append(rec)
            out.append({"turn": t, **rec})
        return out

    def update_leaning(self, name, obj, now):
        raw = _s((obj or {}).get("leaning")).upper()
        lean = next((k for k in self.letters if raw in (k.upper(), f"ANSWER {k.upper()}")), "undecided")
        self.leanings[name].append((now, lean))
        return lean

    # -- moves
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
        problems = []
        stance = parse_stance(obj.get("stance") or obj.get("relation"))
        if stance not in SPEAKER_STANCES:
            problems.append("no valid stance (attack, support or premise)")
            stance = ""
        kind, val = parse_target(obj.get("target"), self.letters)
        if stance in ("attack", "support"):
            if kind == "turn":
                tgt = self.entry_at(val)
                if tgt is None or val >= entry["turn"]:
                    problems.append(f"turn {val} doesn't exist yet")
                    kind = val = None
                elif tgt["move"] != "assert":
                    problems.append(f"turn {val} is not an argument")
                    kind = val = None
                elif tgt.get("retracted_at"):
                    problems.append(f"turn {val} was already withdrawn")
            elif kind is None:
                problems.append(f"unclear target {obj.get('target')!r}")
        else:
            kind = val = None
        repeat = any(e["move"] == "assert" and e["agent"] == agent["name"] and e["fact"] == fact
                     and same_stance((e["stance"], e["target_kind"], e["target"]), (stance, kind, val))
                     for e in self.table)
        if repeat:
            problems.append("repeats an earlier argument of theirs")
        entry.update(move="assert", fact=fact, fact_status=status, extra_facts=extra, stance=stance,
                     target_kind=kind, target=val, aim=_s(obj.get("aim")).lower(),
                     say=_s(obj.get("say") or obj.get("claim")), mine=norm_part(obj.get("mine")),
                     repeat=repeat, retracted_at=None, problem="; ".join(problems) or None)
        # the speaker's own reading of its argument = what it meant
        self.readings[agent["name"]].setdefault(entry["turn"], []).append(
            {"as_of": entry["turn"], "stance": stance, "target_kind": kind, "target": val, "own": True,
             **(entry["mine"] or {"premises": "", "assumption": "", "conclusion": ""})})

    def do_ask(self, entry, obj, mv):
        t = parse_int(obj.get("turn", obj.get("target")))
        tgt = self.entry_at(t)
        problem, owner = None, None
        if tgt is None:
            problem = f"turn {t} doesn't exist"
        elif tgt["move"] != "assert":
            problem = f"turn {t} is not an argument"
        elif tgt["agent"] == entry["agent"]:
            problem = f"turn {t} is their own argument"
        else:
            owner = tgt["agent"]
        entry.update(move=mv, turn_requested=t, owner=owner, say=_s(obj.get("say") or obj.get("question")),
                     problem=problem)
        if not problem:
            self.pending = {"to": owner, "turn": t, "kind": "stance" if mv == "ask_stance" else "argument",
                            "asked_by": entry["agent"], "asked_at": entry["turn"], "say": entry["say"]}
            self.questions.append({**self.pending, "answered_at": None, "answer_consistent": None})
        return owner

    def do_explain(self, entry, obj):
        t = parse_int(obj.get("turn", obj.get("target_turn")))
        tgt = self.entry_at(t)
        problem = None
        if tgt is None:
            problem = f"turn {t} doesn't exist"
        elif tgt["move"] != "assert":
            problem = f"turn {t} is not an argument"
        elif tgt["agent"] != entry["agent"]:
            problem = f"turn {t} is not their own argument"
        stance = parse_stance(obj.get("stance"))
        stance = stance if stance in SPEAKER_STANCES else ""
        kind, val = parse_target(obj.get("target"), self.letters) if stance in ("attack", "support") else (None, None)
        argument = norm_part(obj.get("argument"))
        consistent = None
        if not problem and stance:
            consistent = same_stance((stance, kind, val), (tgt["stance"], tgt["target_kind"], tgt["target"]))
        p = self.pending
        answers = bool(p and not problem and p["to"] == entry["agent"] and p["turn"] == t)
        entry.update(move="explain", turn_requested=t, stance=stance, target_kind=kind, target=val,
                     argument=argument, say=_s(obj.get("say")), problem=problem,
                     answers_question=answers, consistent_with_record=consistent)
        return answers

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
        entry.update(move="retract", turn_requested=t, say=_s(obj.get("say") or obj.get("reason")), problem=problem)

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
            system = (RULES.replace("<<NAME>>", name).replace("<<ROLE>>", role)
                      .replace("<<N>>", str(n)).replace("<<OTHERS>>", others))
            prompt = self.agent_prompt(agent)
            raw, err = self.ask(self.models[name], system, prompt, lambda: self.mock_agent(agent))
            obj, mv = read_reply(raw) if not err else (None, "")
            entry = {"turn": turn, "agent": name, "model": self.models[name], "raw": raw, "error": err}
            if not err and mv not in MOVE_NAMES:  # unreadable: ask once more, with the format spelled out
                entry["raw_first_try"] = raw
                raw, err = self.ask(self.models[name], system, prompt + FORMAT_REMINDER,
                                    lambda: self.mock_agent(agent, retry=True))
                obj2, mv = read_reply(raw) if not err else (None, "")
                obj = obj2 if obj2 is not None else obj
                entry.update(raw=raw, error=err, retried=True)
            entry["readings_update"] = self.update_readings(name, obj, turn)
            entry["leaning"] = self.update_leaning(name, obj, turn)
            was_asked = self.pending if (self.pending and self.pending["to"] == name) else None
            if was_asked:
                self.pending = None  # this turn is their chance to answer
            answered = False
            next_name = None
            if mv == "assert":
                self.do_assert(entry, obj, agent)
            elif mv == "explain":
                self.pending = was_asked  # do_explain checks the answer against the open question
                answered = self.do_explain(entry, obj)
                self.pending = None
                if answered:
                    q = next(q for q in reversed(self.questions) if q["asked_at"] == was_asked["asked_at"])
                    q["answered_at"], q["answer_consistent"] = turn, entry["consistent_with_record"]
                    next_name = was_asked["asked_by"]  # back to whoever asked
            elif mv in ("ask_stance", "stance?", "ask_argument", "show"):
                owner = self.do_ask(entry, obj, "ask_stance" if mv in ("ask_stance", "stance?") else "ask_argument")
                if owner:
                    next_name = owner  # the asked agent answers next
            elif mv in ("retract", "withdraw"):
                self.do_retract(entry, obj)
            elif mv == "pass":
                entry.update(move="pass")
            else:
                entry.update(move="unreadable" if not err else "error")
            if was_asked and not answered:
                entry["ignored_question"] = True
            self.table.append(entry)
            on_turn(entry)

            errors_in_row = errors_in_row + 1 if entry["move"] in ("error", "unreadable") else 0
            if errors_in_row >= 2 * n:
                return "stopped: agents kept failing (API errors or unreadable replies; check key, credits, models)"
            # only a real pass counts towards ending the dialogue
            passes_in_row = passes_in_row + 1 if entry["move"] == "pass" else 0
            if passes_in_row >= n and not self.pending:
                return "everyone passed"
            idx = self.names.index(next_name) if next_name else (idx + 1) % n
        return "turn limit reached"

    # -- the judge answers from the public record only
    def judge(self, pooled=False):
        if pooled:
            all_f = all_facts(self.sc)
            body = f"All known facts:\n{facts_text(all_f)}"
        else:
            shared = {fid: self.fact_text[fid] for fid in sorted(self.on_table - self.public)}
            body = (f"Public facts:\n{facts_text(self.sc.get('public_facts', {}))}\n\n"
                    f"Facts stated during the discussion:\n{facts_text(shared)}\n\n"
                    f"The discussion:\n{self.public_record()}")
        user = (f"Question: {self.sc['question']}\nPossible answers:\n{options_text(self.sc)}\n\n{body}\n\n"
                "Based only on the information above, which answer is best supported? Reply with ONLY: "
                '{"answer": "<letter>", "explanation": "<two or three sentences>", "facts_used": ["ID", ...]}')
        raw, err = self.ask(self.judge_model, JUDGE_SYSTEM, user, self.mock_judge)
        obj = parse_json(raw) or {}
        ans = str(obj.get("answer", "")).strip().upper()[:1]
        return {"model": self.judge_model, "answer": ans, "correct": ans == self.sc["answer"],
                "explanation": str(obj.get("explanation", "")), "facts_used": obj.get("facts_used", []),
                "raw": raw, "error": err}

    # -- did listeners reconstruct arguments the way their speakers meant them?
    def final_pairs(self):
        T = len(self.table)
        pairs = []
        for e in self.table:
            if e["move"] != "assert":
                continue
            for name in self.names:
                if name == e["agent"]:
                    continue
                r = reading_at(self.readings, name, e["turn"], T)
                if r is not None:
                    pairs.append((name, e, r))
        return pairs

    def check_reconstructions(self):
        pairs = [(n, e, r) for n, e, r in self.final_pairs() if (r["assumption"] or r["conclusion"]) and e["mine"]]
        if not pairs:
            return {"pairs": 0, "error": None, "raw": "", "ratings": {}}
        blocks = [f"Pair {n}@{e['turn']}\n"
                  f"  What {e['agent']} said at turn {e['turn']}: fact {e['fact']} "
                  f"(\"{self.fact_text.get(e['fact'], '?')}\"), \"{e['say']}\"\n"
                  f"  What {e['agent']} privately meant: {part_text(e['mine'])}\n"
                  f"  How {n} reconstructed it: {part_text(r)}" for n, e, r in pairs]
        user = ("For each pair, compare how a listener reconstructed an argument with what its speaker privately "
                "meant. Focus on the unstated assumption and the conclusion. Rate each pair:\n"
                '"match": the same argument;\n"partial": the same conclusion, but a different assumption or premises;\n'
                '"mismatch": a different argument.\n\n' + "\n\n".join(blocks) +
                '\n\nReply with ONLY: {"pairs": [{"pair": "Bob@1", "rating": "match", "note": "one short sentence"}]}')
        raw, err = self.ask(self.judge_model, CHECK_SYSTEM, user, lambda: self.mock_check(pairs), max_tokens=3000)
        got = {}
        for p in (parse_json(raw) or {}).get("pairs") or []:
            if isinstance(p, dict):
                got[re.sub(r"\s+", "", _s(p.get("pair"))).lower()] = p
        ratings = {}
        for n, e, r in pairs:
            p = got.get(f"{n}@{e['turn']}".lower())
            rating = _s(p.get("rating")).lower() if p else "unrated"
            ratings[f"{n}@{e['turn']}"] = {"rating": rating if rating in ("match", "partial", "mismatch", "unrated")
                                           else "unclear", "note": _s(p.get("note")) if p else ""}
        return {"pairs": len(pairs), "error": err, "raw": raw, "ratings": ratings}

    # -- everyone's graph at every turn
    def views(self, support_rule):
        T = len(self.table)
        out = {}
        for t in range(1, T + 1):
            out[t] = {INTENDED: compute_view(self.sc, self.table, self.readings, None, t, support_rule)}
            for name in self.names:
                out[t][name] = compute_view(self.sc, self.table, self.readings, name, t, support_rule)
        return out

    def timeline(self, views):
        rows = []
        for t, v in views.items():
            row = {"turn": t, "move": f"{self.table[t - 1]['agent']}: {self.table[t - 1]['move']}"}
            sets = {}
            for name in self.names:
                lean = [l for (tt, l) in self.leanings[name] if tt <= t]
                row[f"{name}: graph verdict"] = v[name]["verdict"] or "none"
                row[f"{name}: leaning"] = lean[-1] if lean else "-"
                row[f"{name}: matches intended"] = round(jaccard(edge_set(v[name]), edge_set(v[INTENDED])), 2)
                sets[name] = edge_set(v[name])
            pairs = [(a, b) for i, a in enumerate(self.names) for b in self.names[i + 1:]]
            row["agents agree (edges)"] = round(sum(jaccard(sets[a], sets[b]) for a, b in pairs) / len(pairs), 2)
            row["intended verdict"] = v[INTENDED]["verdict"] or "none"
            rows.append(row)
        return rows

    # -- fake players for testing without spending money
    def mock_agent(self, agent, retry=False):
        name = agent["name"]
        p = self.pending
        reads = []
        for e in self.table:
            if e["move"] == "assert" and e["agent"] != name and random.random() < 0.8:
                st_ = random.choice(READING_STANCES)
                tgt = (random.choice(self.letters) if random.random() < 0.6 else e["target"] or random.choice(self.letters)) \
                    if st_ in ("attack", "support") else None
                reads.append({"turn": e["turn"], "stance": st_, "target": tgt, "premises": "(mock)",
                              "assumption": "(mock) what I think they assume", "conclusion": "(mock)"})
        base = {"readings": reads, "leaning": random.choice(self.letters + ["undecided"])}
        if not retry and random.random() < 0.05:
            return "I think we should look at the oxygen data."
        if p and p["to"] == name:
            e = self.entry_at(p["turn"])
            honest = random.random() < 0.8
            return json.dumps({**base, "move": "explain", "turn": p["turn"],
                               "stance": e["stance"] if honest else random.choice(SPEAKER_STANCES),
                               "target": e["target"] if honest else random.choice(self.letters),
                               "argument": {"premises": "(mock)", "assumption": "(mock)", "conclusion": "(mock)"},
                               "say": "(mock) Here is what I meant."})
        others = [e for e in self.table if e["move"] == "assert" and e["agent"] != name and not e.get("retracted_at")]
        r = random.random()
        if others and r < 0.15:
            return json.dumps({**base, "move": random.choice(["ask_stance", "ask_argument"]),
                               "turn": random.choice(others)["turn"], "say": "(mock) What did you mean?"})
        unshared = [f for f in agent["facts"] if f not in self.on_table]
        if unshared and r < 0.8:
            st_ = random.choice(SPEAKER_STANCES)
            if st_ == "premise":
                tgt = None
            elif others and random.random() < 0.5:
                tgt = random.choice(others)["turn"]
            else:
                tgt = random.choice(self.letters)
            mv = {"move": "assert", "fact": random.choice(unshared), "stance": st_,
                  "target": tgt, "aim": "assumption", "say": "(mock) This matters.",
                  "mine": {"premises": "(mock)", "assumption": "(mock)", "conclusion": "(mock)"}}
            if random.random() < 0.3:  # the nested format some models use
                return json.dumps({**base, "move": mv})
            return json.dumps({**base, **mv})
        return json.dumps({**base, "move": "pass"})

    def mock_judge(self):
        return json.dumps({"answer": random.choice(self.letters), "explanation": "(mock judge)", "facts_used": []})

    def mock_check(self, pairs):
        return json.dumps({"pairs": [{"pair": f"{n}@{e['turn']}", "rating": random.choice(["match", "partial", "mismatch"]),
                                      "note": "(mock)"} for n, e, r in pairs]})

    # -- numbers for the paper
    def metrics(self, recon=None):
        T = len(self.table)
        private_shared = self.on_table - self.public
        key = set(self.sc.get("key_facts", [])) - self.public
        asserts = [e for e in self.table if e["move"] == "assert"]
        given = matched = misread = irrelevant_on_relevant = unclear = unread = repaired = 0
        for e in asserts:
            meant = (e["stance"], e["target_kind"], e["target"])
            for name in self.names:
                if name == e["agent"]:
                    continue
                hist = self.readings[name].get(e["turn"], [])
                if not hist:
                    unread += 1
                    continue
                given += 1
                last = hist[-1]
                ok = same_stance((last["stance"], last["target_kind"], last["target"]), meant)
                matched += ok
                misread += not ok
                if last["stance"] == "irrelevant" and e["stance"] in ("attack", "support"):
                    irrelevant_on_relevant += 1
                unclear += last["stance"] == "unclear"
                first = hist[0]
                if ok and not same_stance((first["stance"], first["target_kind"], first["target"]), meant):
                    repaired += 1
        ratings = (recon or {}).get("ratings", {})
        return {
            "turns": T,
            "arguments": len(asserts),
            "private_facts_total": len(self.owner),
            "private_facts_shared": len(private_shared),
            "shared_by_agent": {a["name"]: [len(set(a["facts"]) & private_shared), len(a["facts"])]
                                for a in self.agents},
            "key_facts_total": len(key),
            "key_facts_shared": len(key & private_shared),
            "extra_facts_shared": len(private_shared - key) if key else None,
            "readings_given": given,
            "readings_unread": unread,
            "stance_readings_matching_intent": matched,
            "stance_misreadings": misread,
            "misreadings_repaired": repaired,
            "relevant_moves_read_as_irrelevant": irrelevant_on_relevant,
            "readings_unclear": unclear,
            "stance_questions": sum(1 for q in self.questions if q["kind"] == "stance"),
            "argument_questions": sum(1 for q in self.questions if q["kind"] == "argument"),
            "questions_answered": sum(1 for q in self.questions if q["answered_at"]),
            "answers_consistent_with_record": sum(1 for q in self.questions if q["answer_consistent"] is True),
            "answers_inconsistent_with_record": sum(1 for q in self.questions if q["answer_consistent"] is False),
            "reconstructions_match": sum(1 for v in ratings.values() if v["rating"] == "match"),
            "reconstructions_partial": sum(1 for v in ratings.values() if v["rating"] == "partial"),
            "reconstructions_mismatch": sum(1 for v in ratings.values() if v["rating"] == "mismatch"),
            "retractions": sum(1 for e in self.table if e["move"] == "retract" and not e["problem"]),
            "repeated_arguments": sum(1 for e in asserts if e.get("repeat")),
            "invalid_citations": sum(1 for e in asserts if e["fact_status"] == "invalid"),
            "moves_with_problems": sum(1 for e in self.table if e.get("problem")),
            "unreadable_replies": sum(1 for e in self.table if e["move"] == "unreadable"),
            "retries": sum(1 for e in self.table if e.get("retried")),
            "api_errors": sum(1 for e in self.table if e.get("error")),
        }


# ----------------------------------------------------------------- display

def show_turn(entry, avatar, facts, reveal, table=None):
    """reveal: {"stance": set, "argument": set, "readings": set} of agent names whose private notes to show."""
    mv = entry["move"]
    who = entry["agent"]
    with st.chat_message(who, avatar=avatar):
        head = f"**{who}** · `{entry['model']}` · turn {entry['turn']} · **{mv.replace('_', ' ')}**"
        if mv in ("ask_stance", "ask_argument", "explain", "retract") and entry.get("turn_requested"):
            head += f" → turn {entry['turn_requested']}"
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
            if who in reveal["stance"]:
                st.markdown(f"🔒 *Meant:* {stance_text(entry['stance'], entry['target_kind'], entry['target'])}"
                            + (f", aimed at its {entry['aim']}" if entry.get("aim") and entry["stance"] != "premise" else ""))
            if who in reveal["argument"]:
                st.markdown(f"🔒 *Full argument:* {part_text(entry['mine'])}")
            notes = []
            if entry.get("repeat"):
                notes.append("repeats an earlier argument")
            if entry.get("problem"):
                notes.append(entry["problem"])
            if entry.get("retracted_at"):
                notes.append(f"withdrawn at turn {entry['retracted_at']}")
            if notes:
                st.caption(":orange[" + " · ".join(notes) + "]")
        elif mv in ("ask_stance", "ask_argument"):
            if entry["say"]:
                st.markdown(f"> {entry['say']}")
            if entry["problem"]:
                st.caption(f":orange[{entry['problem']}]")
            else:
                st.caption(f"{entry['owner']} must answer next.")
        elif mv == "explain":
            if entry["say"]:
                st.markdown(f"> {entry['say']}")
            if entry["stance"]:
                st.markdown(f"- *Stance:* {stance_text(entry['stance'], entry['target_kind'], entry['target'])}")
            if entry["argument"]:
                st.markdown(f"- *Argument:* {part_text(entry['argument'])}")
            if entry["problem"]:
                st.caption(f":orange[{entry['problem']}]")
            elif who in reveal["stance"] and entry["consistent_with_record"] is not None:
                st.caption("🔒 " + ("Matches what they meant at the time." if entry["consistent_with_record"]
                                    else ":red[Differs from what they meant at the time.]"))
        elif mv == "retract":
            if entry["say"]:
                st.markdown(f"> {entry['say']}")
            if entry["problem"]:
                st.caption(f":orange[Not withdrawn: {entry['problem']}]")
        elif mv == "unreadable":
            st.caption(":red[The reply could not be read, even after one retry. Nothing was said; "
                       "this does not count as a pass.]")
        if entry.get("retried") and mv != "unreadable":
            st.caption("First reply was unreadable; this is the retry.")
        if entry.get("ignored_question"):
            st.caption(":orange[Did not answer the question put to them.]")
        if who in reveal["readings"]:
            lines = [f"turn {r['turn']}: {stance_text(r['stance'], r['target_kind'], r['target'])}"
                     + (f" — {part_text(r)}" if (r['assumption'] or r['conclusion']) else "")
                     for r in entry.get("readings_update", [])]
            st.markdown("🔒 *Readings updated:* " + ("\n" + "\n".join(f"- {l}" for l in lines) if lines else "none")
                        + f"\n\n🔒 *Leaning:* {entry.get('leaning', '-')}")
        with st.expander("raw model output"):
            st.code(entry.get("raw") or "(empty)", language=None)


def verdict_metric(col, label, v, gold):
    col.metric(label, v or "none", ("correct" if v == gold else "wrong") if v else "no single answer accepted",
               delta_color="green" if v == gold else ("red" if v else "gray"), delta_arrow="off")


def show_inside(run):
    sc, table, views = run["scenario"], run["transcript"], run["views"]
    names = [a["name"] for a in sc["agents"]]
    T = len(table)
    st.subheader("Inside the interlocutors")
    if T == 0:
        st.caption("No turns were played.")
        return
    st.caption("Each agent's private picture of the argument: its readings of every argument on the table, "
               "and the graph those readings imply (labelled with grounded semantics). Compare with what the "
               "speakers actually meant.")
    c1, c2 = st.columns([1, 2])
    t = c1.slider("After turn", 1, T, T, key="inside_turn") if T > 1 else 1
    who = c2.radio("Whose view", [INTENDED] + names, horizontal=True, key="inside_who")
    g = views[t][who] if t in views else views[str(t)][who]
    intended = views[t][INTENDED] if t in views else views[str(t)][INTENDED]
    diff = [] if who == INTENDED else [list(e[1:]) for e in edge_set(g) - edge_set(intended)]
    m1, m2, m3 = st.columns(3)
    verdict_metric(m1, f"{who}: graph verdict", g["verdict"], sc["answer"])
    if who != INTENDED:
        m2.metric("Edges matching what speakers meant", f"{jaccard(edge_set(g), edge_set(intended)):.0%}")
        m3.metric("Arguments not yet read", len(g.get("unread", [])))
    st.graphviz_chart(graph_dot(g, highlight=diff), width="stretch")
    st.caption("Green = accepted · red = rejected · grey = undecided. Solid red = attack · dashed green = support · "
               "orange = a link this agent sees that the speaker didn't mean · dashed box = not read yet.")
    if who != INTENDED:
        rows = []
        for e in table[:t]:
            if e["move"] != "assert":
                continue
            meant = stance_text(e["stance"], e["target_kind"], e["target"])
            if e["agent"] == who:
                rows.append({"turn": e["turn"], "by": "self", "fact": e["fact"], "read as": meant, "speaker meant": meant,
                             "match": "", "reconstruction": part_text(e["mine"])})
                continue
            r = reading_at(run["readings"], who, e["turn"], t)
            read = stance_text(r["stance"], r["target_kind"], r["target"]) if r else "not read yet"
            ok = (same_stance((r["stance"], r["target_kind"], r["target"]),
                              (e["stance"], e["target_kind"], e["target"])) if r else None)
            hist = (run["readings"].get(who) or {}).get(e["turn"]) or (run["readings"].get(who) or {}).get(str(e["turn"])) or []
            revised = len([h for h in hist if h["as_of"] <= t]) > 1
            rows.append({"turn": e["turn"], "by": e["agent"], "fact": e["fact"], "read as": read, "speaker meant": meant,
                         "match": "" if ok is None else ("✓" if ok else "✗"), "revised": "yes" if revised else "",
                         "reconstruction": part_text(r) if r else "",
                         "reconstruction check": (run.get("recon_check") or {}).get("ratings", {}).get(f"{who}@{e['turn']}", {}).get("rating", "")})
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
    with st.expander("Turn-by-turn: verdicts, leanings and agreement"):
        st.dataframe(run["timeline"], width="stretch", hide_index=True)


def show_results(run):
    sc = run["scenario"]
    names = [a["name"] for a in sc["agents"]]
    avatar = {a["name"]: AVATARS[i % len(AVATARS)] for i, a in enumerate(sc["agents"])}
    facts = all_facts(sc)
    stamp = run["timestamp"].replace(":", "-")
    m = run["metrics"]

    st.subheader("Transcript")
    st.caption(f"Ended because: {run['ended']}. By default you see what the audience heard; "
               "pick interlocutors below to reveal their private notes.")
    c1, c2, c3 = st.columns(3)
    reveal = {"stance": set(c1.multiselect("Show what they meant (stance)", names, key="rv_stance")),
              "argument": set(c2.multiselect("Show their full arguments", names, key="rv_arg")),
              "readings": set(c3.multiselect("Show their readings and leaning", names, key="rv_read"))}
    for entry in run["transcript"]:
        show_turn(entry, avatar[entry["agent"]], facts, reveal)

    st.subheader("Verdict")
    j = run["judge"]
    cols = st.columns(4)
    cols[0].metric("Judge (public record only)", j["answer"] or "?", verdict(j),
                   delta_color="green" if j["correct"] else "red", delta_arrow="off")
    if run.get("pooled"):
        p = run["pooled"]
        cols[1].metric("Pooled baseline (all facts)", p["answer"] or "?", verdict(p),
                       delta_color="green" if p["correct"] else "red", delta_arrow="off")
    cols[2].metric("Gold answer", sc["answer"])
    cols[3].metric("Est. cost (USD)", f"{run['usage']['est_cost_usd']:.4f}")
    T = len(run["transcript"])
    if T:
        final = run["views"][T] if T in run["views"] else run["views"][str(T)]
        cols = st.columns(len(names) + 1)
        verdict_metric(cols[0], "Intended graph", final[INTENDED]["verdict"], sc["answer"])
        for i, n in enumerate(names):
            verdict_metric(cols[i + 1], f"{n}'s graph", final[n]["verdict"], sc["answer"])
    st.markdown(f"**Judge's explanation:** {j['explanation'] or '(none)'}")
    if j.get("error"):
        st.error(j["error"])
    if sc.get("gold_explanation"):
        st.markdown(f"**Gold explanation:** {sc['gold_explanation']}")

    show_inside(run)

    st.subheader("Readings, questions and repair")
    c = st.columns(6)
    c[0].metric("Readings given", m["readings_given"])
    c[1].metric("Stance misreadings (final)", m["stance_misreadings"])
    c[2].metric("Misreadings repaired", m["misreadings_repaired"])
    c[3].metric("Relevant moves read as irrelevant", m["relevant_moves_read_as_irrelevant"])
    c[4].metric("Questions asked / answered",
                f"{m['stance_questions'] + m['argument_questions']} / {m['questions_answered']}")
    c[5].metric("Answers that differ from the record", m["answers_inconsistent_with_record"])
    rc = run.get("recon_check")
    if rc and rc.get("pairs"):
        st.caption(f"Reconstruction check (judge): {m['reconstructions_match']} match, "
                   f"{m['reconstructions_partial']} partial, {m['reconstructions_mismatch']} mismatch.")
        if rc.get("error"):
            st.error(rc["error"])

    st.subheader("Disclosure")
    d = st.columns(4)
    d[0].metric("Turns", m["turns"])
    d[1].metric("Private facts shared", f"{m['private_facts_shared']} / {m['private_facts_total']}")
    if m["key_facts_total"]:
        d[2].metric("Key facts shared", f"{m['key_facts_shared']} / {m['key_facts_total']}")
    d[3].metric("Repeated arguments", m["repeated_arguments"])
    rows = [{"fact": fid, "holder": a["name"], "key": fid in sc.get("key_facts", []),
             "shared at turn": run["first_shared"].get(fid), "text": txt}
            for a in sc["agents"] for fid, txt in a["facts"].items()]
    st.dataframe(rows, width="stretch", hide_index=True)

    st.download_button("Download this run (JSON)", json.dumps(run, indent=2, ensure_ascii=False, default=str),
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
        api_key = st.text_input("Your OpenRouter API key", type="password", help="Not stored. Get one at openrouter.ai.")
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

    max_turns = st.slider("Max turns", 4, 80, 30, help="One fact per move, so dialogues run longer.")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    show_targets = st.checkbox("Agents see what each move targets", value=True,
                               help="Off: others hear only the fact and the spoken line, not the stance "
                                    "(attack / support / premise) or its target.")
    check_recon = st.checkbox("Check reconstructions at the end (one judge call)", value=True,
                              help="The judge compares each listener's reconstruction with what the speaker meant.")
    run_pooled = st.checkbox("Also run pooled baseline (judge sees all facts)", value=True)
    support_rule = st.checkbox("Support for one answer counts as attack on the others", value=True,
                               help="Answers are mutually exclusive, so arguing for B argues against A and C.")

sc = scenarios[chosen]
problems = validate(sc)
if problems:
    st.error("This scenario has problems: " + "; ".join(problems))
    st.stop()

st.title("🔎 Inquiry Arena")
st.caption("LLM agents with split knowledge try to answer a question together. Each argument states one fact; "
           "what it is for stays private unless someone asks its author.")
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
        return st.selectbox(label, model_ids, index=model_ids.index(default_model(i)), key=key,
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

if st.button("▶ Run dialogue", type="primary", width="stretch"):
    if not mock and not api_key:
        st.error("Add an OpenRouter key in the sidebar, or switch on mock mode.")
        st.stop()
    settings = {"max_turns": max_turns, "temperature": temperature, "mock": mock, "show_targets": show_targets,
                "check_reconstructions": check_recon, "support_rule": support_rule}
    ref = Referee(sc, models, judge_model, settings, api_key, models_available)
    avatar = {a["name"]: AVATARS[i % len(AVATARS)] for i, a in enumerate(sc["agents"])}
    facts = all_facts(sc)
    status = st.empty()
    live = st.container()
    no_reveal = {"stance": set(), "argument": set(), "readings": set()}

    def on_turn(entry):
        with live:
            show_turn(entry, avatar[entry["agent"]], facts, no_reveal)

    ended = ref.run(on_turn, lambda s: status.info(s))
    status.info("The judge is reading the public record...")
    judge = ref.judge(pooled=False)
    pooled = None
    if run_pooled:
        status.info("Running the pooled baseline...")
        pooled = ref.judge(pooled=True)
    recon = None
    if check_recon:
        status.info("The judge is checking reconstructions...")
        recon = ref.check_reconstructions()
    views = ref.views(support_rule)
    status.empty()
    st.session_state.last_run = {
        "app": APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenario": sc, "models": models, "judge_model": judge_model, "settings": settings,
        "ended": ended, "transcript": ref.table, "first_shared": ref.first_shared,
        "readings": ref.readings, "leanings": ref.leanings, "questions": ref.questions,
        "judge": judge, "pooled": pooled, "recon_check": recon,
        "views": views, "timeline": ref.timeline(views),
        "metrics": ref.metrics(recon), "usage": ref.usage,
    }
    st.rerun()

if st.session_state.get("last_run", {}).get("app") not in (None, APP_VERSION):
    del st.session_state["last_run"]
if "last_run" in st.session_state:
    st.divider()
    show_results(st.session_state.last_run)
