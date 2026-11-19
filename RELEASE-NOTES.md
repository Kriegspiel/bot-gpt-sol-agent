# Release notes

## 1.0.0

- Make validated postgame lessons available to future games automatically.
- Activate previously queued lessons from successful reviews at startup.
- Preserve pause, rollback, and the lesson snapshots of games already underway.

## 0.3.0

- Allocate live thinking time using the clock and historical checkmate
  horizons by public material, opponent role, and drop rules.
- Select reasoning effort from low through max, with time for a local fallback.
- Give live provider requests the remaining model deadline and log their
  requested reasoning effort.
- Journal elapsed turn time and learn which efforts fit from persisted live
  provider latencies and timeouts.

## 0.2.0

- Advertise and play all seven supported rulesets through the existing T4 account.
- Use variant-specific referee guidance in live play and completed-game review,
  including CrazyKrieg reserve pieces, drops, and promotions.
- Preserve the hidden-board boundary and scope lessons to the game ruleset.

## 0.1.2

- Admit up to two simultaneous live requests. If both slots are busy, wait
  briefly within the existing turn deadline before using local fallback.
- Review completed games in a separate slot while other games are active.
  Review rate limits pause only the review worker.

## 0.1.1

- Preserve the clock budget and attempt journal across referee probes.
- Match prior referee rejections to the chess move number.
- Support GPT-6.2 Sol on Responses in the standard service tier and include cache-write charges in usage estimates.

## 0.1.0

- Add the T4 GPT Sol Agent for Berkeley + Any.
- Use clock-aware reasoning, ranked fallback attempts and a durable referee journal.
- Review completed games with the full replay and retain evidence-linked general
  and opponent-specific lessons for operator approval.
- Record play and review spend against the platform's shared OpenAI ledger.
