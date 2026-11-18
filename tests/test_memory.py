import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from memory import MemoryStore


def open_game(store, game_id, *, code=None, opponent="account-2", variant="berkeley_any"):
    return store.open_game(
        game_id, game_code=code or game_id, variant=variant, own_color="white",
        opponent_key=opponent, opponent_role="user", opponent_elo=1410,
    )


def lesson(scope="general"):
    return {
        "scope": scope,
        "phase": "opening",
        "observation": "Center control may be costly after dubious probes.",
        "adjustment": "Prefer probes that prepare development.",
        "confidence": "medium",
        "evidence_plies": [1, 3],
    }


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "memory.sqlite3"
        self.store = MemoryStore(self.path)

    def test_approved_scoped_lessons_and_frozen_snapshots(self):
        open_game(self.store, "finished")
        self.store.mark_completed("finished")
        finished = self.store.pending_games()[0]
        self.assertTrue(self.store.finish_review(finished, [lesson(), lesson("opponent")]))
        self.assertFalse(self.store.finish_review(finished, [lesson()]))
        self.assertEqual(open_game(self.store, "before-approval"), [])

        lessons = self.store.status()["lessons"]
        for note in lessons:
            self.assertTrue(self.store.approve(note["id"]))
        same_opponent = open_game(self.store, "same")
        self.assertEqual([note["scope"] for note in same_opponent], ["opponent", "general"])
        self.assertEqual(len(open_game(self.store, "different", opponent="account-3")), 1)
        self.assertEqual(open_game(self.store, "rules-changed", variant="wild16"), [])
        self.assertFalse({"game_id", "opponent_key", "evidence_plies"} & set(same_opponent[0]))

        restarted = MemoryStore(self.path)
        self.assertEqual(open_game(restarted, "before-approval"), [])
        self.assertEqual(open_game(restarted, "same"), same_opponent)
        self.assertTrue(restarted.rollback(same_opponent[0]["id"]))
        self.assertEqual(open_game(restarted, "same"), same_opponent)
        self.assertEqual(len(open_game(restarted, "after-rollback")), 1)
        self.assertEqual(open_game(MemoryStore(self.path, use_memory=False), "baseline"), [])
        restarted.set_enabled(False)
        self.assertEqual(open_game(restarted, "same"), same_opponent)
        self.assertEqual(open_game(restarted, "after-disable"), [])
        self.assertFalse(MemoryStore(self.path).is_enabled())

    def test_intent_is_durable_and_enriched_by_result(self):
        open_game(self.store, "first", code="SAME23")
        key = "white:0"
        self.store.record_attempt("first", turn_key=key, action="d2d4", answer=None, decision={"source": "fallback"})

        restarted = MemoryStore(self.path)
        self.assertEqual(restarted.tried_actions("first", key), {"d2d4"})
        self.assertEqual(restarted.tried_actions("first", "white:2"), set())
        restarted.record_attempt(
            "first", turn_key=key, action="d2d4", answer={"move_done": False, "full_fen": "private"},
        )
        context = restarted.review_context("first")
        self.assertEqual(len(context), 1)
        self.assertEqual(context[0]["decision"], {"source": "fallback"})
        self.assertFalse(context[0]["answer"]["move_done"])
        self.assertNotIn("full_fen", context[0]["answer"])

        open_game(restarted, "second", code="SAME23")
        self.assertEqual(restarted.tried_actions("second", key), set())

    def test_review_context_keeps_only_projected_observations(self):
        open_game(self.store, "match")
        state = {
            "your_color": "white", "ply_count": 2, "turn": "white",
            "clock": {"white_remaining": 42, "private": "discard"},
            "referee_turns": [{
                "turn": 1,
                "white": [{"kind": "move", "move_uci": "e2e4", "message": "OK"}],
                "black": [{"kind": "status", "move_uci": "private", "replay_fen": "private", "message": "OK"}],
            }],
        }
        self.store.record_observation("match", state)
        self.store.record_observation("match", {**state, "clock": {"white_remaining": 25}})
        entry, = self.store.review_context("match")
        self.assertEqual(entry["turn"], "white:2")
        self.assertEqual(entry["clock_at_start"], {"white_remaining": 42})
        turn, = entry["recent_referee_at_start"]
        self.assertEqual(turn["white"][0]["move_uci"], "e2e4")
        self.assertEqual(turn["black"], [{"kind": "status", "message": "OK"}])

    def test_pending_completed_game_survives_restart(self):
        open_game(self.store, "archive", code="GAME23")
        self.store.mark_completed("archive")
        restarted = MemoryStore(self.path)
        pending = restarted.pending_games()
        self.assertEqual([(game["game_id"], game["game_code"]) for game in pending], [("archive", "GAME23")])

    def test_tracked_game_will_be_reconciled_after_restart_without_completion_signal(self):
        with patch("memory.time.time", return_value=1000):
            open_game(self.store, "unobserved-finish")
            self.assertEqual(self.store.pending_games(), [])
        restarted = MemoryStore(self.path)
        with patch("memory.time.time", return_value=1300):
            pending, = restarted.pending_games()
            self.assertEqual(pending["game_id"], "unobserved-finish")
            restarted.defer_review("unobserved-finish")
            self.assertEqual(restarted.pending_games(), [])
