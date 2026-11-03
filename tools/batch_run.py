"""Run many Inquiry Arena dialogues without the web page.

It uses the Referee from your app.py (v0.5 or later), so batch runs follow exactly the same rules,
strategies and judges as the app. With app v0.5.2 or later, --strategies also accepts orchestrated
conditions: orch-rules, orch-llm, orch-schedule, orch-random. For every scenario x strategy x repeat it saves the full run as
JSON (the same format as the app's "Download this run") and adds one row to summary.csv.

Examples:
  # free test with fake players (no API key, no cost)
  python batch_run.py --app app.py --scenarios scenarios/dev_n2_step_s1 --out results/mock --mock --limit 5

  # a real pilot: 10 scenarios, two strategies, stop if the estimated cost passes $1
  export OPENROUTER_API_KEY=sk-or-...
  python batch_run.py --app app.py --scenarios scenarios/dev_n2_step_s1 --out results/pilot \
      --strategies free goals --models openai/gpt-4o-mini --judge openai/gpt-4o-mini --limit 10 --max-cost 1

  # the four fixed strategies against two orchestrators and the random control, with a last call
  python batch_run.py --app app.py --scenarios scenarios/dev_n2_step_s1 --out results/orch \
      --strategies free goals assumptions contrast orch-rules orch-llm orch-random --last-call --limit 10

Runs that already have a result file are skipped, so you can stop and restart safely.
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
PAGE_MARKER = "# ----------------------------------------------------------------- page"

FIELDS = ["run_id", "scenario", "item", "split", "allocation", "agents", "strategy", "repeat", "models", "judge_model",
          "ended", "turns", "gold", "judge_answer", "judge_correct", "pooled_correct", "question_only_correct",
          "key_facts_shared", "key_facts_total", "all_key_facts_public", "justified_proxy",
          "private_facts_shared", "private_facts_total", "extra_facts_shared",
          "goals_posted", "goals_answered", "goals_open_at_end", "show_requests", "retractions",
          "readings_checked", "readings_mismatch", "mismatches_unrepaired",
          "unreadable_replies", "replies_needing_retry", "api_errors", "cost_usd", "prompt_tokens", "completion_tokens",
          "rounds", "strategy_switches", "orchestrator_calls", "last_call_at", "new_facts_after_last_call",
          "turns_free", "turns_goals", "turns_assumptions", "turns_contrast"]


def load_app(path):
    """Load everything in app.py above the page section (helpers, Referee, strategies), without the web page."""
    src = open(path, encoding="utf-8").read()
    if PAGE_MARKER not in src:
        sys.exit(f"{path} does not look like the Inquiry Arena app (page marker not found).")
    import logging
    import streamlit  # noqa: F401  (imported first so its loggers exist and can be quietened)
    for name in list(logging.root.manager.loggerDict):
        if name.startswith("streamlit"):
            logging.getLogger(name).setLevel(logging.ERROR)
    ns = {"__file__": os.path.abspath(path), "__name__": "inquiry_arena_app"}
    exec(compile(src.split(PAGE_MARKER)[0], path, "exec"), ns)
    if "STRATEGIES" not in ns:
        sys.exit("This app.py has no strategies. Use v0.5 or later.")
    return ns


def question_only(app, ref, sc):
    """Control: can the judge answer from the question and public facts alone (prior knowledge)?"""
    user = (f"Question: {sc['question']}\nPossible answers:\n{app['options_text'](sc)}\n\n"
            f"Public facts:\n{app['facts_text'](sc.get('public_facts', {}))}\n\n"
            "Based only on the information above and general knowledge, which answer is best? Reply with ONLY: "
            '{"answer": "<letter>", "explanation": "<one sentence>"}')
    raw, err = ref.ask(ref.judge_model, app["JUDGE_SYSTEM"], user, ref.mock_judge)
    obj = app["parse_json"](raw) or {}
    ans = str(obj.get("answer", "")).strip().upper()[:1]
    return {"answer": ans, "correct": ans == sc["answer"], "raw": raw, "error": err}


def one_run(app, sc, scen_file, strategy, repeat, a, prices, api_key):
    names = [ag["name"] for ag in sc["agents"]]
    models = {n: a.models[i % len(a.models)] for i, n in enumerate(names)}
    base, policy = app["parse_condition"](strategy) if "parse_condition" in app else (strategy, None)
    settings = {"max_turns": a.max_turns, "temperature": a.temperature, "mock": a.mock,
                "show_targets": not a.hide_targets, "check_readings": not a.no_reading_check,
                "support_rule": not a.no_support_rule, "strategy": base, "policy": policy,
                "orchestrator_model": a.orchestrator_model or a.judge, "orchestrator_sees_private": a.oracle,
                "last_call": a.last_call, "seed": f"{a.seed}|{scen_file}|{strategy}|{repeat}"}
    ref = app["Referee"](sc, models, a.judge, settings, api_key, prices)
    ended = ref.run(lambda e: None, lambda s: None)
    judge = ref.judge(pooled=False)
    pooled = None if a.no_pooled else ref.judge(pooled=True)
    qonly = None if a.no_question_only else question_only(app, ref, sc)
    reading = ref.check_readings() if settings["check_readings"] else None
    graph = ref.declared_graph(settings["support_rule"])
    if policy:
        label, text = app["condition_label"](base, policy), ""
    else:
        label, text = app["STRATEGIES"][strategy]["label"], app["STRATEGIES"][strategy]["text"]
    return {
        "app": app["APP_VERSION"],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenario": sc, "scenario_file": scen_file, "repeat": repeat,
        "models": models, "judge_model": a.judge, "settings": settings, "ended": ended,
        "strategy": {"key": strategy, "label": label, "text": text},
        "rounds": getattr(ref, "rounds", []), "notes": getattr(ref, "notes", []),
        "transcript": ref.table, "goals": ref.goals, "first_shared": ref.first_shared,
        "judge": judge, "pooled": pooled, "question_only": qonly, "reading_check": reading,
        "graph": graph, "metrics": ref.metrics(), "usage": ref.usage,
    }


def summary_row(run_id, run):
    sc, m, j = run["scenario"], run["metrics"], run["judge"]
    src = sc.get("source", {})
    all_key = bool(m["key_facts_total"]) and m["key_facts_shared"] == m["key_facts_total"]
    return {
        "run_id": run_id, "scenario": run["scenario_file"], "item": src.get("item", sc.get("title", "")),
        "split": src.get("split", ""), "allocation": (sc.get("allocation") or {}).get("mode", ""),
        "agents": len(sc["agents"]), "strategy": run["strategy"]["key"], "repeat": run["repeat"],
        "models": "|".join(sorted(set(run["models"].values()))), "judge_model": run["judge_model"],
        "ended": run["ended"], "turns": m["turns"], "gold": sc["answer"],
        "judge_answer": j["answer"], "judge_correct": int(j["correct"]),
        "pooled_correct": "" if not run["pooled"] else int(run["pooled"]["correct"]),
        "question_only_correct": "" if not run["question_only"] else int(run["question_only"]["correct"]),
        "key_facts_shared": m["key_facts_shared"], "key_facts_total": m["key_facts_total"],
        "all_key_facts_public": int(all_key),
        # proxy until proof checks exist: right answer AND every proof fact made public
        "justified_proxy": int(j["correct"] and all_key),
        "private_facts_shared": m["private_facts_shared"], "private_facts_total": m["private_facts_total"],
        "extra_facts_shared": m["extra_facts_shared"],
        "goals_posted": m["goals_posted"], "goals_answered": m["goals_answered"],
        "goals_open_at_end": m["goals_open_at_end"], "show_requests": m["show_requests"],
        "retractions": m["retractions"], "readings_checked": m["readings_checked"],
        "readings_mismatch": m["readings_mismatch"], "mismatches_unrepaired": m["mismatches_unrepaired"],
        "unreadable_replies": m["unreadable_replies"], "replies_needing_retry": m["replies_needing_retry"],
        "api_errors": m["api_errors"], "cost_usd": round(run["usage"]["est_cost_usd"], 6),
        "prompt_tokens": run["usage"]["prompt_tokens"], "completion_tokens": run["usage"]["completion_tokens"],
        "rounds": m.get("rounds", ""), "strategy_switches": m.get("strategy_switches", ""),
        "orchestrator_calls": m.get("orchestrator_calls", ""), "last_call_at": m.get("last_call_at") or "",
        "new_facts_after_last_call": m.get("new_facts_after_last_call", ""),
        **{f"turns_{k}": (m.get("turns_by_strategy") or {}).get(k, "")
           for k in ("free", "goals", "assumptions", "contrast")},
    }


def print_summary(csv_path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if not rows:
        return
    print("\nPer strategy (all runs in summary.csv):")
    print(f"{'strategy':<15}{'runs':>6}{'judge ok':>10}{'justified*':>12}{'pooled ok':>11}{'q-only ok':>11}"
          f"{'private shared':>16}{'turns':>7}{'cost $':>9}")
    for s in sorted({r["strategy"] for r in rows}):
        rs = [r for r in rows if r["strategy"] == s]
        mean = lambda k: sum(float(r[k]) for r in rs if r[k] != "") / max(1, sum(1 for r in rs if r[k] != ""))
        shared = sum(int(r["private_facts_shared"]) for r in rs) / max(1, sum(int(r["private_facts_total"]) for r in rs))
        print(f"{s:<15}{len(rs):>6}{mean('judge_correct'):>10.2f}{mean('justified_proxy'):>12.2f}"
              f"{mean('pooled_correct'):>11.2f}{mean('question_only_correct'):>11.2f}{shared:>16.2f}"
              f"{mean('turns'):>7.1f}{sum(float(r['cost_usd']) for r in rs):>9.3f}")
    print("* justified = correct answer and every proof fact made public (a proxy until proof checks exist)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app", required=True, help="path to app.py (v0.5 or later)")
    ap.add_argument("--scenarios", required=True, help="folder with scenario_*.json files")
    ap.add_argument("--out", required=True, help="folder for results")
    ap.add_argument("--strategies", nargs="+", default=["free", "goals", "assumptions", "contrast"])
    ap.add_argument("--models", nargs="+", default=["openai/gpt-4o-mini"],
                    help="one model for everyone, or one per agent (Ana, Bob, Cara...)")
    ap.add_argument("--judge", default="openai/gpt-4o-mini")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N scenarios (0 = all)")
    ap.add_argument("--shuffle", action="store_true", help="shuffle scenarios before --limit")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--hide-targets", action="store_true", help="agents don't see what each move targets")
    ap.add_argument("--no-support-rule", action="store_true")
    ap.add_argument("--no-reading-check", action="store_true")
    ap.add_argument("--no-pooled", action="store_true")
    ap.add_argument("--no-question-only", action="store_true")
    ap.add_argument("--max-cost", type=float, default=5.0, help="stop starting new runs after this many USD")
    ap.add_argument("--workers", type=int, default=1, help="dialogues run in parallel")
    ap.add_argument("--mock", action="store_true", help="fake players, no API calls, free")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--orchestrator-model", default=None, help="model for orch-llm (default: the judge model)")
    ap.add_argument("--oracle", action="store_true", help="orch-llm also sees every agent's private facts")
    ap.add_argument("--last-call", action="store_true",
                    help="when everyone passes, one more round before ending (all conditions)")
    a = ap.parse_args()

    app = load_app(a.app)
    known = list(app["STRATEGIES"]) + [f"orch-{p}" for p in app.get("POLICIES", {})]
    bad = [s for s in a.strategies if s not in known]
    if bad:
        sys.exit(f"Unknown strategies {bad}. Choose from {known}.")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not a.mock and not api_key:
        sys.exit("Set OPENROUTER_API_KEY first, or use --mock.")
    prices = {} if a.mock else app["fetch_models"]()
    if not a.mock:
        unknown = [m for m in set(a.models + [a.judge] + ([a.orchestrator_model] if a.orchestrator_model else []))
                   if prices and m not in prices]
        if unknown:
            sys.exit(f"OpenRouter does not list these models: {unknown}")

    files = sorted(glob.glob(os.path.join(a.scenarios, "scenario_*.json")))
    if a.shuffle:
        random.Random(a.seed).shuffle(files)
    if a.limit:
        files = files[: a.limit]
    if not files:
        sys.exit(f"No scenario_*.json files in {a.scenarios}")
    runs_dir = os.path.join(a.out, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    csv_path = os.path.join(a.out, "summary.csv")
    new_csv = not os.path.exists(csv_path)
    lock = threading.Lock()
    spent = [0.0]
    if not new_csv:
        header = next(csv.reader(open(csv_path, encoding="utf-8")), [])
        if header != FIELDS:
            sys.exit(f"{csv_path} was written by another version of this script (different columns). "
                     "Use a new --out folder.")
    if not new_csv:  # count what earlier sessions already spent
        spent[0] = sum(float(r["cost_usd"] or 0) for r in csv.DictReader(open(csv_path, encoding="utf-8")))

    jobs = []
    for fn in files:
        for s in a.strategies:
            for r in range(1, a.repeats + 1):
                run_id = f"{os.path.splitext(os.path.basename(fn))[0]}__{s}__r{r}"
                if not os.path.exists(os.path.join(runs_dir, run_id + ".json")):
                    jobs.append((fn, s, r, run_id))
    print(f"{len(files)} scenarios x {len(a.strategies)} strategies x {a.repeats} repeats: "
          f"{len(jobs)} runs to do ({'mock' if a.mock else 'real models'}), already spent ${spent[0]:.3f}")

    def work(job):
        fn, s, r, run_id = job
        with lock:
            if spent[0] >= a.max_cost:
                return run_id, "skipped (cost cap reached)"
        sc = json.load(open(fn, encoding="utf-8"))
        problems = app["validate"](sc)
        if problems:
            return run_id, "invalid scenario: " + "; ".join(problems)
        t0 = time.time()
        run = one_run(app, sc, os.path.basename(fn), s, r, a, prices, api_key)
        with open(os.path.join(runs_dir, run_id + ".json"), "w", encoding="utf-8") as f:
            json.dump(run, f, indent=1, ensure_ascii=False)
        row = summary_row(run_id, run)
        with lock:
            global_new = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if global_new:
                    w.writeheader()
                w.writerow(row)
            spent[0] += run["usage"]["est_cost_usd"]
        return run_id, (f"judge {'right' if row['judge_correct'] else 'wrong'}, {row['turns']} turns, "
                        f"{row['private_facts_shared']}/{row['private_facts_total']} facts shared, "
                        f"${row['cost_usd']:.4f}, {time.time() - t0:.0f}s")

    done = 0
    if a.workers <= 1:
        for job in jobs:
            run_id, msg = work(job)
            done += 1
            print(f"[{done}/{len(jobs)}] {run_id}: {msg}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(work, j) for j in jobs]
            for fut in as_completed(futs):
                run_id, msg = fut.result()
                done += 1
                print(f"[{done}/{len(jobs)}] {run_id}: {msg}", flush=True)
    print(f"\nEstimated total cost so far: ${spent[0]:.3f}. Results in {a.out}")
    if os.path.exists(csv_path):
        print_summary(csv_path)


if __name__ == "__main__":
    main()
