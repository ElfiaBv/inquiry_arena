# Inquiry Arena

LLM agents with split knowledge try to answer a question together.
Only the referee (this app) sees every fact. Each agent sees the public facts,
its own private facts, and the conversation so far.

**Moves:** `assert` (share facts by ID and make a claim), `why` (ask another agent to justify something), `pass`.
The dialogue ends when every agent passes in a row, or at the turn limit.
A judge model then reads only the public record and picks an answer.
The pooled baseline lets the judge see all facts, for comparison.

## Files
- `app.py`: the app
- `requirements.txt`: libraries Streamlit installs
- `scenario_*.json`: scenarios shown in the dropdown (add your own with the same format)

## Secrets (set in Streamlit Cloud, never commit them)
```toml
OPENROUTER_API_KEY = "sk-or-..."
APP_PASSWORD = "choose-a-password"
```

## Run on your own computer
```
pip install -r requirements.txt
streamlit run app.py
```
