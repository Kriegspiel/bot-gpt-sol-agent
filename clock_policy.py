"""Plan turns remaining and select an effort that fits the measured response time."""

from math import ceil

MIN_PLAY_REQUEST_SECONDS = 5.0
_EFFORTS = (
    ("low", MIN_PLAY_REQUEST_SECONDS, 1024),
    ("medium", 18.0, 2048),
    ("high", 30.0, 4096),
    ("xhigh", 45.0, 6144),
    ("max", 60.0, 8192),
)
REASONING_EFFORTS = tuple(effort for effort, _, _ in _EFFORTS)
_MIN_FASTER_SAMPLES = 3

# Position-weighted 75th percentiles for accepted moves in 8,913 archived
# checkmates (April 2 through November 18, 2026 Pacific time). Kings count;
# CrazyKrieg pockets do not. Tables start at the indicated piece count.
_BOT_FROM_THREE = (
    112, 96, 207, 140, 157, 101, 111, 110, 108, 100,
    89, 85, 86, 84, 77, 85, 81, 82, 83, 83,
    86, 85, 88, 88, 93, 90, 96, 98, 97, 102,
)
_HUMAN_FROM_EIGHT = (
    22, 20, 21, 29, 27, 26, 21, 26, 28, 27,
    33, 34, 34, 35, 36, 39, 38, 37, 43, 45,
    45, 49, 49, 47, 53,
)
_CRAZYKRIEG_FROM_FOURTEEN = (
    33, 27, 28, 34, 37, 44, 46, 48, 56, 58,
    58, 58, 65, 68, 67, 72, 73, 75, 80,
)


def remaining_turns(state: dict, opponent_role: str | None) -> int:
    material = state.get("material_summary", {})
    counts = [material.get(side, {}).get("pieces_remaining") for side in ("white", "black")]
    if not all(type(count) is int and count > 0 for count in counts):
        return _BOT_FROM_THREE[-1]
    pieces = sum(counts)
    if not 2 <= pieces <= 32:
        return _BOT_FROM_THREE[-1]

    # The low-material human and CrazyKrieg bins have too few distinct games
    # for their observed tails to justify a more aggressive clock allowance.
    if state.get("rule_variant") == "crazykrieg":
        return _CRAZYKRIEG_FROM_FOURTEEN[pieces - 14] if pieces >= 14 else 45
    if opponent_role in {"user", "guest"}:
        return _HUMAN_FROM_EIGHT[pieces - 8] if pieces >= 8 else 32
    return _BOT_FROM_THREE[pieces - 3] if pieces >= 3 else _BOT_FROM_THREE[-1]


def _padded_upper_quartile(samples: list[float]) -> float:
    ordered = sorted(samples)
    return ordered[ceil(len(ordered) * 0.75) - 1] * 1.2 + 2.0


def reasoning_expectations(timings: dict[str, list[tuple[float, bool]]] | None = None) -> dict[str, float]:
    requirements = {}
    for effort, initial, _ in _EFFORTS:
        samples = (timings or {}).get(effort, ())
        completed = [seconds for seconds, timed_out in samples if not timed_out]
        timeouts = [seconds for seconds, expired in samples if expired]
        required = _padded_upper_quartile(completed) if completed else initial
        if len(completed) < _MIN_FASTER_SAMPLES:
            required = max(initial, required)
        if timeouts:
            required = max(required, _padded_upper_quartile(timeouts))
        requirements[effort] = max(MIN_PLAY_REQUEST_SECONDS, required)
    return requirements


def reasoning_for(
    model_seconds: float, timings: dict[str, list[tuple[float, bool]]] | None = None,
) -> tuple[str, int] | None:
    if model_seconds <= MIN_PLAY_REQUEST_SECONDS:
        return None
    requirements = reasoning_expectations(timings)
    for effort, _, max_tokens in reversed(_EFFORTS):
        if model_seconds >= requirements[effort]:
            return effort, max_tokens
    return None
