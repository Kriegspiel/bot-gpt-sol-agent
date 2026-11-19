# GPT Sol Agent

`agent_gpt_sol` is a T4 Expert bot for all seven playable Kriegspiel rulesets.
It supports Berkeley, Berkeley + Any, Cincinnati, Wild 16, RAND, English, and
CrazyKrieg. See the [ruleset references](rules.md). One service owns the
identity, with a runner per assigned game and up to two concurrent live
model requests. When both slots are busy, a turn can wait up to three seconds
while leaving time for inference and the fallback move submission.

The live player reasons from the assigned ruleset, its projected board, referee
announcements, clock, and the opponent's public name, role and rating. For
CrazyKrieg it reads the public reserves and can submit drop attempts. The
runner budgets from the server clock and can submit a locally ranked attempt
when inference is slow or unavailable. It rechecks the state after a model request, re-ranks
after every referee answer and journals submission intents before posting them.

Live decisions budget from the remaining clock and a historical 75th-percentile
checkmate horizon based on public board material, the opponent's role, and the
drop variant. The agent protects a clock reserve and can spend more of a large
clock surplus on a turn. Paid play selects `low`, `medium`, `high`, `xhigh`, or
`max` as time permits. It uses local fallback if there is too little time to
start inference. The live turn budget is capped at 90 seconds, with a separate
30-second cap while the clock is idle before the first move. Both budgets
leave four seconds for refreshing and submitting a move.

The journal records elapsed turn time from first observing the turn through
each referee reply. The agent tracks live provider latency separately for
each model and effort. It keeps up to 20 measurements per effort for seven
days. The cold-start thresholds are 5, 18, 30, 45, and 60 seconds. A padded
75th percentile of observations recalibrates which effort fits the available
time. Timeouts supply a lower bound, and three completed requests are
required before a threshold can fall below its default. Inspect current
requirements after the first observations with `python -m memory --db PATH status`.

A separate worker reviews one of the agent's own completed games at a time,
even while other games are active. It uses a separate model request slot and
can compare full replay with the evidence available to the player at the time.
It records validated lessons with evidence and confidence, either general or
scoped to an opponent's stable account ID. A game loads its initial lessons
and reloads them after every 10 full moves, each comprising a move by White and
Black. Each refresh can pick up lessons from reviews completed since the last
one. The worker keeps each selection in SQLite and resumes it after a restart.
Live play receives the lesson summaries and public opponent context. An operator
can pause learning or revoke a lesson, which affects the next selection. Keep
the SQLite database private. It contains player-visible referee and decision
evidence and is created under a restrictive process umask.

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
