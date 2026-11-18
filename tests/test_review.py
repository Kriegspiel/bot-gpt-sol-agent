from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from memory import MemoryStore
from review import ReviewWorker, validate_lessons


FULL_FEN = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1"
OWN_FEN = "8/8/8/8/3P4/8/PPP1PPPP/RNBQKBNR b - - 0 1"


def complete_review():
    return {
        "game": {
            "game_id": "match",
            "game_code": "PLAY23",
            "rule_variant": "berkeley_any",
            "state": "completed",
            "white": {"username": "agent_gpt_sol"},
            "black": {"username": "opponent"},
            "result": {"winner": "white", "reason": "checkmate"},
        },
        "transcript": {
            "game_id": "match",
            "rule_variant": "berkeley_any",
            "viewer_color": "white",
            "moves": [{
                "ply": 1, "color": "white", "question_type": "COMMON", "uci": "d2d4", "move_done": True,
                "answer": {"main": "OK", "capture_square": None},
                "replay_fen": {"full": FULL_FEN, "white": OWN_FEN, "black": "black perspective"},
            }],
        },
    }


def candidate():
    return {
        "scope": "general", "phase": "opening", "confidence": "medium",
        "observation": "An accepted center pawn move opened development options.",
        "adjustment": "Consider central development when the center is not already contested.",
        "evidence_plies": [1],
    }


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = MemoryStore(Path(self.directory.name) / "memory.sqlite3")
        self.store.open_game(
            "match", game_code="PLAY23", variant="berkeley_any", own_color="white",
            opponent_key="stable-opponent", opponent_role="bot", opponent_elo=1500,
        )
        self.store.mark_completed("match")

    def test_review_uses_full_replay_and_stores_only_candidate_notes(self):
        response = complete_review()
        fetch = Mock(return_value=response)
        model = Mock(return_value={"lessons": [candidate()]})
        self.store.record_attempt(
            "match", turn_key="white:0", action="d2d4", answer={"move_done": True},
            decision={"source": "model", "rationale": "Contests the center."},
        )
        worker = ReviewWorker(self.store, fetch_review=fetch, model_review=model)

        self.assertTrue(worker.run_once())
        fetch.assert_called_once_with("match")
        payload = model.call_args.args[0]
        self.assertEqual(payload["game_id"], "match")
        self.assertEqual(payload["replay"][0]["full_fen_after"], FULL_FEN)
        self.assertEqual(payload["replay"][0]["own_fen_after"], OWN_FEN)
        self.assertNotIn("black perspective", str(payload))
        data = self.store.status()
        self.assertNotIn(FULL_FEN, str(data))
        self.assertEqual(data["lessons"][0]["status"], "candidate")
        self.assertEqual(data["games"][0]["result"]["accepted_moves"], 1)
        self.assertEqual(data["games"][0]["result"]["model_attempts"], 1)
        self.assertFalse(worker.run_once())
        model.assert_called_once()

    def test_review_requires_completed_own_game_and_exact_archive(self):
        game = self.store.pending_games()[0]
        changes = [
            ("game", "state", "active"),
            ("game", "white", {"username": "other"}),
            ("game", "game_code", "ELSE23"),
            ("game", "game_id", "not-match"),
            ("transcript", "viewer_color", "black"),
            ("transcript", "game_id", "not-match"),
        ]
        for section, key, wrong_value in changes:
            with self.subTest(section=section, key=key):
                response = complete_review()
                response[section][key] = wrong_value
                model = Mock()
                worker = ReviewWorker(self.store, fetch_review=lambda _: response, model_review=model)
                self.assertFalse(worker.review_game(game))
                model.assert_not_called()
        self.assertEqual(self.store.status()["lessons"], [])

    def test_memory_advice_rejects_private_replay_and_invalid_evidence(self):
        game = self.store.pending_games()[0]
        for field, value in (
            ("observation", FULL_FEN), ("adjustment", "Play d2d4 here"), ("adjustment", "Try N@e4"),
            ("adjustment", "Replay PLAY23"), ("evidence_plies", [2]), ("evidence_plies", [True]),
            ("evidence_plies", []),
        ):
            with self.subTest(field=field, value=value):
                note = deepcopy(candidate())
                note[field] = value
                self.assertEqual(validate_lessons({"lessons": [note]}, game=game, replay_events=1), [])
        self.assertEqual(validate_lessons({"lessons": [candidate()]}, game=game, replay_events=1), [candidate()])

    def test_disabling_persists_without_running_review(self):
        self.store.set_enabled(False)
        model, fetch = Mock(), Mock()
        worker = ReviewWorker(self.store, fetch_review=fetch, model_review=model)
        self.assertFalse(worker.run_once())
        fetch.assert_not_called()
        model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
