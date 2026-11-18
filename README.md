# GPT Sol Agent

`agent_gpt_sol` plays [Berkeley + Any Kriegspiel](https://kriegspiel.org/rules/berkeley)
as a separate T4 Expert account. One service owns the identity, with a runner
per assigned game and one concurrent live model request.

The live player reasons from its projected board, referee announcements,
clock, and the opponent's public name, role and rating. The runner budgets
from the server clock and can submit a locally ranked attempt when inference
is slow or unavailable. It rechecks the state after a model request, re-ranks
after every referee answer and journals submission intents before posting them.

A separate worker reviews the agent's own completed games when it is idle. It
can compare full replay with the evidence available to the player at the time.
It proposes bounded lessons with evidence and confidence, either general or
scoped to an opponent's stable account ID. A game uses a frozen set of approved
lessons. The live provider prompt receives approved summaries and public
opponent context. Keep the SQLite database private. It contains player-visible referee and
decision evidence and is created under a restrictive process umask.

## Run locally

Use Python 3.15 in production. Install dependencies with `python -m pip install -r requirements.txt`.
Set the following in the process environment or in a private file passed with
`--env-file`.

| Setting | Use |
| --- | --- |
| `KRIEGSPIEL_API_BASE` | Game API base URL |
| `KRIEGSPIEL_BOT_TOKEN` | Token for the `agent_gpt_sol` identity |
| `LLM_API_KEY` | Direct OpenAI API key |
| `LLM_MODEL` | Defaults to `gpt-6.2-sol`. The operator verifies model access on the production key. |
| `OPENAI_MONTHLY_BUDGET_USD` | Existing host-wide provider ceiling |
| `OPENAI_MONTHLY_BUDGET_STATE_PATH` | The existing shared provider ledger |
| `AGENT_USE_MEMORY` | Set to `false` for a match without retrieved lessons |

Run `python bot.py --env-file /path/to/private.env --state-file /path/to/memory.sqlite3`.
Registration uses `--register-only` and requires a private token capture. The
platform wrapper manages production setup and listing.

Published costs are estimated from provider usage. Application logs identify
play and review separately. The shared ledger reserves a conservative upper
bound when a provider request's final usage is unknown. `provider_budget.py`
is reused under the BSD-3-Clause license from
[bot-openai-compatible at d995a19](https://github.com/Kriegspiel/bot-openai-compatible/blob/d995a19/provider_budget.py)
to keep both workers on the same ledger format.

Run `python -m unittest discover -s tests -v` for tests and
`python -m compileall -q .` for the compile check.
