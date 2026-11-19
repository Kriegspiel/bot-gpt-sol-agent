"""Local journal and curated advice for a single bot identity."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterator

from clock_policy import REASONING_EFFORTS, reasoning_expectations

_MAX_ADVICE = 6
_CHECK_INTERVAL = 120
_RETRY_INTERVAL = 300
_TIMING_HISTORY_PER_EFFORT = 20
_TIMING_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
_TIMING_MAX_DURATION_SECONDS = 300
_TIMING_DB_TIMEOUT_SECONDS = 0.25
_SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    game_code TEXT NOT NULL,
    variant TEXT NOT NULL,
    own_color TEXT NOT NULL,
    opponent_key TEXT,
    opponent_role TEXT NOT NULL,
    opponent_elo INTEGER,
    snapshot TEXT NOT NULL,
    review_state TEXT NOT NULL DEFAULT 'tracking',
    next_check REAL NOT NULL,
    result TEXT,
    last_error TEXT
);
CREATE INDEX IF NOT EXISTS games_pending_review ON games(review_state, next_check);
CREATE TABLE IF NOT EXISTS journal (
    game_id TEXT NOT NULL REFERENCES games(game_id),
    turn_key TEXT NOT NULL,
    kind TEXT NOT NULL,
    action TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL,
    PRIMARY KEY (game_id, turn_key, kind, action)
);
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY,
    game_id TEXT NOT NULL REFERENCES games(game_id),
    variant TEXT NOT NULL,
    opponent_key TEXT,
    phase TEXT NOT NULL,
    observation TEXT NOT NULL,
    adjustment TEXT NOT NULL,
    confidence TEXT NOT NULL,
    evidence_plies TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'candidate'
);
CREATE INDEX IF NOT EXISTS lessons_retrieval ON lessons(variant, status, opponent_key);
CREATE TABLE IF NOT EXISTS model_timings (
    id INTEGER PRIMARY KEY,
    model TEXT NOT NULL,
    effort TEXT NOT NULL,
    seconds REAL NOT NULL,
    timed_out INTEGER NOT NULL CHECK (timed_out IN (0, 1)),
    observed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS model_timings_recent ON model_timings(model, effort, id DESC);
"""

_CLOCK_FIELDS = ("white_remaining", "black_remaining", "active_color")
_ANSWER_FIELDS = (
    "move_done", "announcement", "special_announcement", "capture_square",
    "checks", "game_over", "turn", "has_any", "next_turn_pawn_tries",
    "next_turn_has_pawn_capture", "next_turn_pawn_try_squares",
    "captured_piece_announcement", "dropped_piece_announcement",
    "en_passant_announced", "promotion_announced",
)
_ENTRY_FIELDS = ("kind", "actor", "prompt", "message", "messages", "question_type")
_DECISION_FIELDS = ("source", "rationale", "reason", "elapsed_ms", "budget_seconds", "reasoning_effort")


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _pick(source: object, fields: tuple[str, ...]) -> dict:
    if not isinstance(source, dict):
        return {}
    return {field: source[field] for field in fields if field in source}


def _valid_timing_model(model: str) -> bool:
    return (
        isinstance(model, str) and 1 <= len(model) <= 200
        and model.isprintable() and not any(char.isspace() for char in model)
    )


def _visible_turns(state: dict, own_color: str) -> list[dict]:
    scoresheet = state.get("scoresheet")
    scoresheet = scoresheet if isinstance(scoresheet, dict) else {}
    turns = scoresheet.get("turns") or state.get("referee_turns") or []
    visible = []
    for turn in turns[-2:]:
        if not isinstance(turn, dict):
            continue
        output: dict = {"turn": turn.get("turn")}
        for color in ("white", "black"):
            fields = _ENTRY_FIELDS + ("move_uci",) if color == own_color else _ENTRY_FIELDS
            output[color] = [
                _pick(item, fields) if isinstance(item, dict) else item
                for item in turn.get(color, []) if isinstance(item, (dict, str))
            ]
        visible.append(output)
    return visible


class MemoryStore:
    def __init__(self, path: str | Path, *, enabled: bool = True, use_memory: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self.use_memory = use_memory
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # The service opens this before workers start, and the operator CLI may
        # open the same file while the workers are running.
        with self._connect() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.executescript(_SCHEMA)

    @contextmanager
    def _connect(self, *, timeout: float = 5) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=timeout)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def _learning_enabled(self, db: sqlite3.Connection) -> bool:
        setting = db.execute("SELECT value FROM settings WHERE key = 'learning_enabled'").fetchone()
        return self.enabled and (setting is None or bool(setting["value"]))

    def is_enabled(self) -> bool:
        with self._connect() as db:
            return self._learning_enabled(db)

    def set_enabled(self, enabled: bool) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT INTO settings(key, value) VALUES ('learning_enabled', ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (int(enabled),),
            )

    def record_model_timing(self, model: str, effort: str, seconds: float, *, timed_out: bool) -> None:
        """Keep a bounded window of live provider observations."""
        if (
            not _valid_timing_model(model) or effort not in REASONING_EFFORTS
            or type(seconds) not in (float, int) or not 0 < seconds <= _TIMING_MAX_DURATION_SECONDS
            or not isinstance(timed_out, bool)
        ):
            return
        observed_at = time.time()
        with self._connect(timeout=_TIMING_DB_TIMEOUT_SECONDS) as db:
            db.execute("DELETE FROM model_timings WHERE observed_at < ?", (observed_at - _TIMING_MAX_AGE_SECONDS,))
            db.execute(
                "INSERT INTO model_timings(model, effort, seconds, timed_out, observed_at) VALUES (?, ?, ?, ?, ?)",
                (model, effort, seconds, int(timed_out), observed_at),
            )
            db.execute(
                """DELETE FROM model_timings WHERE model = ? AND effort = ? AND id IN (
                       SELECT id FROM model_timings WHERE model = ? AND effort = ?
                       ORDER BY id DESC LIMIT -1 OFFSET ?
                   )""",
                (model, effort, model, effort, _TIMING_HISTORY_PER_EFFORT),
            )

    def model_timings(self, model: str) -> dict[str, list[tuple[float, bool]]]:
        """Return unexpired durations and timeout flags, newest first within each effort."""
        if not _valid_timing_model(model):
            return {}
        timings: dict[str, list[tuple[float, bool]]] = {}
        with self._connect(timeout=_TIMING_DB_TIMEOUT_SECONDS) as db:
            rows = db.execute(
                """SELECT effort, seconds, timed_out FROM model_timings
                   WHERE model = ? AND observed_at >= ? ORDER BY id DESC LIMIT ?""",
                (model, time.time() - _TIMING_MAX_AGE_SECONDS, _TIMING_HISTORY_PER_EFFORT * len(REASONING_EFFORTS)),
            )
            for row in rows:
                timings.setdefault(row["effort"], []).append((row["seconds"], bool(row["timed_out"])))
        return timings

    def _snapshot(self, db: sqlite3.Connection, variant: str, opponent_key: str | None) -> list[dict]:
        # Keep room for general play even when there is lots of head-to-head evidence.
        notes = []
        targets = [(opponent_key, 2), (None, 4)] if opponent_key else [(None, 4)]
        for key, limit in targets:
            notes.extend(
                db.execute(
                    """SELECT id, phase, observation, adjustment, confidence, opponent_key
                       FROM lessons WHERE variant = ? AND status = 'approved' AND opponent_key IS ?
                       ORDER BY CASE confidence WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                                id DESC LIMIT ?""",
                    (variant, key, limit),
                ).fetchall()
            )
        return [
            {
                "id": row["id"],
                "scope": "opponent" if row["opponent_key"] else "general",
                "phase": row["phase"],
                "observation": row["observation"],
                "adjustment": row["adjustment"],
                "confidence": row["confidence"],
            }
            for row in notes[:_MAX_ADVICE]
        ]

    def open_game(
        self, game_id: str, *, game_code: str | None = None, variant: str, own_color: str, opponent_key: str | None,
        opponent_role: str, opponent_elo: int | None = None,
    ) -> list[dict]:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            enabled = self._learning_enabled(db) and self.use_memory
            game = db.execute("SELECT snapshot FROM games WHERE game_id = ?", (game_id,)).fetchone()
            if game is None:
                advice = self._snapshot(db, variant, opponent_key) if enabled else []
                db.execute(
                    """INSERT INTO games(game_id, game_code, variant, own_color, opponent_key, opponent_role,
                                         opponent_elo, snapshot, next_check)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (game_id, game_code or game_id, variant, own_color, opponent_key, opponent_role, opponent_elo,
                     _json(advice), time.time() + _CHECK_INTERVAL),
                )
                return advice
            return json.loads(game["snapshot"])

    def record_observation(self, game_id: str, state: dict[str, Any]) -> None:
        own_color = state.get("your_color")
        if own_color not in {"white", "black"}:
            return
        observation = _pick(state, ("move_number", "ply_count", "turn", "your_fen"))
        observation["clock"] = _pick(state.get("clock"), _CLOCK_FIELDS)
        observation["referee_turns"] = _visible_turns(state, own_color)
        ply = state.get("ply_count", state.get("move_number", ""))
        self._record(game_id, f"{own_color}:{ply}", observation)

    def record_attempt(
        self, game_id: str, *, turn_key: str, action: str, answer: dict | None,
        decision: str | dict | None = None,
    ) -> None:
        payload: dict = {}
        if answer is not None:
            result = _pick(answer, _ANSWER_FIELDS)
            result["clock"] = _pick(answer.get("clock"), _CLOCK_FIELDS)
            payload["answer"] = result
        if isinstance(decision, str):
            payload["decision"] = decision[:200]
        elif isinstance(decision, dict):
            payload["decision"] = {
                field: value[:250] if isinstance(value, str) else value
                for field, value in _pick(decision, _DECISION_FIELDS).items()
                if isinstance(value, (str, int, float))
            }
        with self._connect() as db:
            db.execute(
                """INSERT INTO journal(game_id, turn_key, kind, action, payload)
                   VALUES (?, ?, 'attempt', ?, ?)
                   ON CONFLICT(game_id, turn_key, kind, action)
                   DO UPDATE SET payload = json_patch(journal.payload, excluded.payload)""",
                (game_id, turn_key, action, _json(payload)),
            )

    def tried_actions(self, game_id: str, turn_key: str) -> set[str]:
        with self._connect() as db:
            return {
                row["action"] for row in db.execute(
                    """SELECT action FROM journal WHERE game_id = ? AND turn_key = ? AND kind = 'attempt'""",
                    (game_id, turn_key),
                )
            }

    def _record(self, game_id: str, turn_key: str, payload: dict) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT INTO journal(game_id, turn_key, kind, payload) VALUES (?, ?, 'observation', ?)
                   ON CONFLICT(game_id, turn_key, kind, action) DO NOTHING""",
                (game_id, turn_key, _json(payload)),
            )

    def review_context(self, game_id: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT turn_key, kind, action, payload FROM journal
                   WHERE game_id = ? ORDER BY rowid""",
                (game_id,),
            ).fetchall()
        context = []
        for row in rows:
            value = json.loads(row["payload"])
            if row["kind"] == "observation":
                context.append({
                    "turn": f"{value.get('turn')}:{value.get('move_number')}",
                    "referee_event_index": value.get("ply_count"),
                    "clock_at_start": value["clock"],
                    "recent_referee_at_start": value["referee_turns"],
                })
            else:
                context.append({"turn": row["turn_key"], "action": row["action"], **value})
        return context

    def mark_completed(self, game_id: str) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE games SET review_state = 'queued', next_check = 0 WHERE game_id = ? AND review_state = 'tracking'",
                (game_id,),
            )

    def pending_games(self, limit: int = 10) -> list[dict]:
        with self._connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    """SELECT game_id, game_code, variant, own_color, opponent_key, opponent_role, opponent_elo, review_state
                       FROM games WHERE review_state IN ('queued', 'tracking') AND next_check <= ?
                       ORDER BY CASE review_state WHEN 'queued' THEN 0 ELSE 1 END, next_check LIMIT ?""",
                    (time.time(), max(1, min(limit, 50))),
                )
            ]

    def defer_review(self, game_id: str, error: str | None = None) -> None:
        with self._connect() as db:
            db.execute(
                """UPDATE games SET next_check = ?, last_error = ?
                   WHERE game_id = ? AND review_state IN ('queued', 'tracking')""",
                (time.time() + (_RETRY_INTERVAL if error else _CHECK_INTERVAL), error, game_id),
            )

    def record_result(self, game_id: str, result: dict) -> None:
        with self._connect() as db:
            db.execute("UPDATE games SET result = ? WHERE game_id = ?", (_json(result), game_id))

    def finish_review(self, game: dict, lessons: list[dict], *, error: str | None = None) -> bool:
        """Publish all candidate lessons and the completion marker atomically."""
        with self._connect() as db:
            changed = db.execute(
                """UPDATE games SET review_state = 'reviewed', last_error = ?
                   WHERE game_id = ? AND review_state IN ('queued', 'tracking')""",
                (error, game["game_id"]),
            ).rowcount
            if not changed:
                return False
            db.executemany(
                """INSERT INTO lessons(game_id, variant, opponent_key, phase, observation,
                                       adjustment, confidence, evidence_plies)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (game["game_id"], game["variant"], game["opponent_key"] if lesson["scope"] == "opponent" else None,
                     lesson["phase"], lesson["observation"], lesson["adjustment"], lesson["confidence"],
                     _json(lesson["evidence_plies"]))
                    for lesson in lessons
                ],
            )
        return True

    def approve(self, lesson_id: int) -> bool:
        with self._connect() as db:
            return bool(db.execute(
                "UPDATE lessons SET status = 'approved' WHERE id = ? AND status = 'candidate'", (lesson_id,)
            ).rowcount)

    def rollback(self, lesson_id: int) -> bool:
        with self._connect() as db:
            # Existing snapshots are retained as part of the record of how each
            # game was played. New games will never pick up the revoked lesson.
            return bool(db.execute(
                "UPDATE lessons SET status = 'revoked' WHERE id = ? AND status != 'revoked'", (lesson_id,)
            ).rowcount)

    def status(self, limit: int = 20) -> dict:
        with self._connect() as db:
            games = [
                {**dict(row), "result": json.loads(row["result"]) if row["result"] else None}
                for row in db.execute(
                    """SELECT game_id, game_code, variant, own_color, opponent_key, opponent_role,
                              opponent_elo, review_state, result, last_error
                       FROM games ORDER BY rowid DESC LIMIT ?""", (limit,)
                )
            ]
            lessons = [
                {**dict(row), "evidence_plies": json.loads(row["evidence_plies"])}
                for row in db.execute("SELECT * FROM lessons ORDER BY id DESC LIMIT ?", (limit,))
            ]
            model_timings = [
                dict(row) for row in db.execute(
                    """SELECT model, effort, COUNT(*) AS samples, SUM(timed_out) AS timeouts
                       FROM model_timings WHERE observed_at >= ? GROUP BY model, effort
                       ORDER BY model, effort""", (time.time() - _TIMING_MAX_AGE_SECONDS,),
                )
            ]
            learning_enabled = self._learning_enabled(db)
        return {
            "learning_enabled": learning_enabled, "games": games, "lessons": lessons,
            "model_timing_counts": model_timings,
            "model_reasoning_seconds": {
                model: {
                    effort: round(seconds, 2)
                    for effort, seconds in reasoning_expectations(self.model_timings(model)).items()
                }
                for model in sorted({row["model"] for row in model_timings})
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or manage the Sol agent's learned advice.")
    parser.add_argument("--db", type=Path, required=True, help="Path to the bot's memory SQLite database.")
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status")
    status.add_argument("--limit", type=int, default=20)
    for verb in ("approve", "rollback"):
        command = commands.add_parser(verb)
        command.add_argument("lesson_id", type=int)
    commands.add_parser("disable", help="Pause reviews and new-game lesson snapshots.")
    commands.add_parser("enable", help="Resume reviews and new-game lesson snapshots.")
    args = parser.parse_args()
    if not args.db.is_file():
        parser.error("the bot's memory database does not exist")
    store = MemoryStore(args.db)
    if args.command == "status":
        print(json.dumps(store.status(max(1, min(args.limit, 1000))), ensure_ascii=False, indent=2))
    elif args.command == "approve":
        if not store.approve(args.lesson_id):
            parser.error("candidate lesson not found")
    elif args.command == "rollback":
        if not store.rollback(args.lesson_id):
            parser.error("active lesson not found")
    else:
        store.set_enabled(args.command == "enable")


if __name__ == "__main__":
    main()
