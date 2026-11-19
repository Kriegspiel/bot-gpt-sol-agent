from contextlib import closing
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from memory import MemoryStore
from provider import EMBEDDING_DIMENSIONS


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


def embedding(axis):
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[axis] = 1.0
    return vector


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "memory.sqlite3"
        self.store = MemoryStore(self.path)

    def publish_review(self, game_id, lessons, **game_options):
        open_game(self.store, game_id, **game_options)
        self.store.mark_completed(game_id)
        game = next(item for item in self.store.pending_games() if item["game_id"] == game_id)
        self.assertTrue(self.store.finish_review(game, lessons))

    def test_automatically_approved_scoped_lessons_and_initial_snapshots(self):
        open_game(self.store, "finished")
        self.assertEqual(open_game(self.store, "before-review"), [])
        self.store.mark_completed("finished")
        finished = self.store.pending_games()[0]
        self.assertTrue(self.store.finish_review(finished, [lesson(), lesson("opponent")]))
        self.assertFalse(self.store.finish_review(finished, [lesson()]))

        self.assertEqual([note["status"] for note in self.store.status()["lessons"]], ["approved", "approved"])

        same_opponent = open_game(self.store, "same")
        self.assertEqual([note["scope"] for note in same_opponent], ["opponent", "general"])
        self.assertEqual(len(open_game(self.store, "different", opponent="account-3")), 1)
        self.assertEqual(open_game(self.store, "rules-changed", variant="wild16"), [])
        self.assertFalse({"game_id", "opponent_key", "evidence_plies"} & set(same_opponent[0]))

        restarted = MemoryStore(self.path)
        self.assertEqual(open_game(restarted, "before-review"), [])
        self.assertEqual(open_game(restarted, "same"), same_opponent)
        self.assertTrue(restarted.rollback(same_opponent[0]["id"]))
        self.assertEqual(open_game(restarted, "same"), same_opponent)
        self.assertEqual(len(open_game(restarted, "after-rollback")), 1)
        self.assertEqual(open_game(MemoryStore(self.path, use_memory=False), "baseline"), [])
        restarted.set_enabled(False)
        self.assertEqual(open_game(restarted, "same"), same_opponent)
        self.assertEqual(open_game(restarted, "after-disable"), [])
        self.assertFalse(MemoryStore(self.path).is_enabled())

    def test_refresh_picks_up_current_scoped_lessons_and_persists_each_checkpoint(self):
        self.publish_review("first-review", [lesson()])
        initial = open_game(self.store, "in-progress")
        self.publish_review("same-opponent", [lesson("opponent")])
        self.publish_review("other-opponent", [lesson("opponent"), lesson()], opponent="account-3")
        self.publish_review("other-ruleset", [lesson(), lesson("opponent")], variant="wild16")
        self.assertTrue(self.store.rollback(initial[0]["id"]))
        self.assertEqual(open_game(self.store, "in-progress"), initial)

        refreshed = self.store.refresh_lessons("in-progress", 10)
        matching, general = refreshed
        expected = {
            note["id"] for note in self.store.status()["lessons"]
            if note["game_id"] in {"same-opponent", "other-opponent"}
            and note["opponent_key"] != "account-3"
        }
        self.assertEqual({note["id"] for note in refreshed}, expected)
        self.assertEqual([matching["scope"], general["scope"]], ["opponent", "general"])
        self.assertFalse({"game_id", "opponent_key", "evidence_plies"} & set(matching))

        self.assertTrue(self.store.rollback(matching["id"]))
        restarted = MemoryStore(self.path)
        self.assertEqual(open_game(restarted, "in-progress"), refreshed)
        self.assertEqual(restarted.refresh_lessons("in-progress", 10), refreshed)
        self.assertEqual(restarted.refresh_lessons("in-progress", 20), [general])
        tracked = next(game for game in restarted.status()["games"] if game["game_id"] == "in-progress")
        self.assertEqual(tracked["lesson_refresh_move"], 20)

    def test_refresh_respects_memory_controls_at_the_checkpoint(self):
        self.publish_review("finished", [lesson()])
        initial = open_game(self.store, "in-progress")
        self.assertEqual(self.store.refresh_lessons("in-progress", 10), initial)

        self.store.set_enabled(False)
        self.assertEqual(self.store.refresh_lessons("in-progress", 20), [])
        self.store.set_enabled(True)
        self.assertEqual(self.store.refresh_lessons("in-progress", 20), [])
        self.assertEqual(self.store.refresh_lessons("in-progress", 30), initial)

        baseline = MemoryStore(self.path, use_memory=False)
        self.assertEqual(baseline.refresh_lessons("in-progress", 40), [])

    def test_semantic_selection_ranks_relevance_within_scope_and_persists_checkpoint_zero(self):
        for key, scope, confidence in (
            ("old-relevant", "general", "low"),
            ("high-relevant", "general", "high"),
            ("new-high-relevant", "general", "high"),
            ("unrelated", "general", "high"),
            ("unindexed", "general", "high"),
            ("new-unrelated", "general", "high"),
            ("opponent-relevant", "opponent", "low"),
            ("opponent-unrelated", "opponent", "high"),
        ):
            self.publish_review(key, [{**lesson(scope), "confidence": confidence}])
        self.publish_review("wrong-variant", [lesson()], variant="wild16")
        self.publish_review("wrong-opponent", [lesson("opponent")], opponent="account-3")
        status = self.store.status()
        ids = {note["game_id"]: note["id"] for note in status["lessons"]}
        self.store.save_lesson_embeddings([
            (lesson_id, embedding(int("unrelated" in key)))
            for key, lesson_id in ids.items() if key != "unindexed"
        ])

        initial = open_game(self.store, "in-progress")
        self.assertNotIn(ids["old-relevant"], {note["id"] for note in initial})
        self.assertTrue(self.store.has_indexed_lessons("in-progress"))
        self.assertIsNone(self.store.saved_lesson_selection("in-progress", 0))
        selected = self.store.refresh_lessons("in-progress", 0, query_vector=embedding(0))
        self.assertEqual([note["id"] for note in selected], [
            ids[key] for key in (
                "opponent-relevant", "opponent-unrelated", "new-high-relevant", "high-relevant",
                "old-relevant", "new-unrelated",
            )
        ])
        self.assertEqual(set(selected[0]), {"id", "scope", "phase", "observation", "adjustment", "confidence"})

        reopened = MemoryStore(self.path)
        self.assertEqual(open_game(reopened, "in-progress"), selected)
        self.assertEqual(reopened.saved_lesson_selection("in-progress", 0), selected)
        self.assertIsNone(reopened.saved_lesson_selection("in-progress", 10))
        self.assertTrue(reopened.rollback(ids["opponent-relevant"]))
        self.assertEqual(reopened.refresh_lessons("in-progress", 0, query_vector=embedding(1)), selected)
        refreshed = reopened.refresh_lessons("in-progress", 10, query_vector=embedding(1))
        self.assertNotIn(ids["opponent-relevant"], {note["id"] for note in refreshed})
        self.assertEqual(reopened.saved_lesson_selection("in-progress", 10), refreshed)

    def test_embedding_backfill_is_idempotent_and_filters_disabled_or_ineligible_lessons(self):
        self.publish_review("prior-review", [lesson("opponent"), lesson("opponent")])
        first, second = self.store.unindexed_lessons()
        self.assertEqual(set(first), {"id", "phase", "observation", "adjustment"})
        self.assertEqual(self.store.unindexed_lessons(limit=1), [first])
        self.assertTrue(self.store.rollback(second["id"]))

        with patch("memory.EMBEDDING_MODEL", "old-embedding-model"):
            self.store.save_lesson_embeddings([(first["id"], embedding(0))])
        open_game(self.store, "eligible")
        self.assertFalse(self.store.has_indexed_lessons("eligible"))
        self.assertEqual(self.store.unindexed_lessons(), [first])
        self.store.save_lesson_embeddings([(first["id"], embedding(0)), (second["id"], embedding(1))])
        self.store.save_lesson_embeddings([(first["id"], embedding(0))])
        self.assertEqual(self.store.status()["indexed_lesson_count"], 1)
        self.assertEqual(self.store.unindexed_lessons(), [])
        self.assertTrue(self.store.has_indexed_lessons("eligible"))

        open_game(self.store, "other-opponent", opponent="account-3")
        open_game(self.store, "other-variant", variant="wild16")
        self.assertFalse(self.store.has_indexed_lessons("other-opponent"))
        self.assertFalse(self.store.has_indexed_lessons("other-variant"))
        self.assertFalse(MemoryStore(self.path, use_memory=False).has_indexed_lessons("eligible"))
        self.publish_review("later-review", [lesson()])
        self.store.set_enabled(False)
        self.assertEqual(self.store.unindexed_lessons(), [])
        self.assertFalse(self.store.has_indexed_lessons("eligible"))

    def test_activates_legacy_candidates_only_from_successful_reviews(self):
        for game_id in ("successful", "failed", "unfinished"):
            open_game(self.store, game_id)
            self.store.mark_completed(game_id)
        pending = {game["game_id"]: game for game in self.store.pending_games()}
        self.assertTrue(self.store.finish_review(pending["successful"], []))
        self.assertTrue(self.store.finish_review(pending["failed"], [lesson()], error="invalid_model_output"))
        self.assertEqual(self.store.status()["lessons"], [])

        with closing(sqlite3.connect(self.path)) as db, db:
            db.executemany(
                """INSERT INTO lessons(id, game_id, variant, phase, observation, adjustment, confidence,
                                       evidence_plies, status)
                   VALUES (?, ?, 'berkeley_any', 'opening', 'Center control allowed development.',
                           'Develop pieces while protecting the center.', 'medium', '[1]', ?)""",
                [
                    (1, "successful", "candidate"),
                    (2, "failed", "candidate"),
                    (3, "unfinished", "candidate"),
                    (4, "successful", "approved"),
                    (5, "successful", "revoked"),
                ],
            )

        existing_snapshot = open_game(self.store, "already-playing")
        self.assertEqual([note["id"] for note in existing_snapshot], [4])
        self.assertEqual(self.store.activate_prior_lessons(), 1)
        self.assertEqual(self.store.activate_prior_lessons(), 0)

        statuses = {note["id"]: note["status"] for note in self.store.status()["lessons"]}
        self.assertEqual(statuses, {1: "approved", 2: "candidate", 3: "candidate", 4: "approved", 5: "revoked"})
        self.assertEqual(open_game(self.store, "already-playing"), existing_snapshot)
        self.assertEqual({note["id"] for note in open_game(self.store, "new-game")}, {1, 4})

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
            "your_color": "white", "ply_count": 4, "move_number": 3, "turn": "white",
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
        self.assertEqual(entry["turn"], "white:3")
        self.assertEqual(entry["referee_event_index"], 4)
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
