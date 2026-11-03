# Tools: EntailmentBank scenarios and batch runs

These scripts sit next to the app; Streamlit ignores this folder.

- `convert_entailmentbank.py` turns EntailmentBank task 2 into scenario files for the app.
- `batch_run.py` runs many dialogues without the web page. It loads the Referee from `app.py` (v0.5 or later), so batch runs use exactly the same rules, strategies and judges as the app.
- `run_in_colab.ipynb` walks through both in Google Colab, step by step.

## What the converter does, per EntailmentBank item

1. It takes the answer options from ARC, joined on the item id. Items whose EntailmentBank answer doesn't match ARC's keyed option are skipped.
2. Proof facts that only restate the question become public facts.
3. The other proof facts are split between the agents (`--mode`). Every agent gets at least one, and nobody gets all of them:
   - `step`: the premises of each proof step go to different agents.
   - `subtree`: each agent gets whole branches of the proof. When the proof is a single chain, this falls back to `step`.
   - `random`: facts are dealt at random.
4. EntailmentBank's own distractors are dealt, most relevant first, so every agent ends up with the same number of facts.
5. Fact IDs are shuffled.
6. Gold fields are written for evaluation only: `key_facts`, `gold_tree`, `gold_explanation`, `fact_types`, `fact_sources`, `difficulty` (including how many proof steps need more than one agent) and `allocation`.

Items with too few private proof facts (fewer than `--min-private`, default 3, or fewer than the number of agents) are skipped. `skipped.csv` says why.

## What the batch runner records

One JSON file per dialogue (the same content as the app's download), and one row per dialogue in `summary.csv`. The main columns:

- `judge_correct`: the judge, reading only the public record, picks the right answer.
- `pooled_correct`: the judge, given all the facts, picks the right answer.
- `question_only_correct`: the judge, given only the question and the public facts, picks the right answer. This measures prior knowledge.
- `justified_proxy`: the answer is right and every proof fact was made public. It stands in for a real proof check.
- Disclosure, goals, shows, retractions, reading checks, unreadable replies and cost.

Runs that already have a result file are skipped, so a stopped batch can be restarted. `--max-cost` stops new runs once the estimated cost reaches the limit.

## Orchestrated conditions (app v0.5.2)

Besides the four fixed strategies, `--strategies` accepts four orchestrated conditions. In these, an orchestrator assigns a strategy to each agent at the start of every round (one turn per agent). It reads the public record only, never the gold answer, and the agents never see its reasons.

- `orch-rules`: fixed if-then rules over the public record. The first rule that applies decides the round, and everyone gets the same strategy:
  1. first round: free;
  2. everyone just passed (last call): assumptions;
  3. an open goal is waiting: goals;
  4. two or more answers have support: contrast;
  5. the last round added no new fact: assumptions;
  6. otherwise: goals.
- `orch-llm`: a model reads the public record, the open goals, the earlier rounds and their results, and picks a strategy per agent. One call per round. `--orchestrator-model` sets the model (default: the judge). `--oracle` also shows it every agent's private facts, as an upper bound. An unreadable reply falls back to the rules, and the run records that.
- `orch-schedule`: free, then goals, assumptions and contrast in turn, whatever happens.
- `orch-random`: a random strategy per agent per round. This is the control: it tells you whether switching itself helps, apart from choosing well.

`--last-call` applies to every condition: when everyone passes, the dialogue gets one more round before it ends. Under the orchestrator, that round gets its own assignment.

Extra columns: `rounds`, `strategy_switches`, `orchestrator_calls`, `last_call_at`, `new_facts_after_last_call`, and `turns_free` / `turns_goals` / `turns_assumptions` / `turns_contrast`. Each run's JSON also has `rounds`, one record per decision: the assignments, the reason, and the state the decision was based on.

The columns differ from v0.5.1, so start a new `--out` folder; the runner refuses to append to an old `summary.csv`.
