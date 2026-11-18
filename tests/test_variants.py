import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from bot import Player, Worker
from memory import MemoryStore
from play import fallback_rank, project
from rules import SUPPORTED_VARIANTS, rules_for
from test_player import GAME, FakeAPI, FakeProvider, state


PLAYABLE_VARIANTS = {
    "berkeley", "berkeley_any", "cincinnati", "wild16", "rand", "english", "crazykrieg",
}


class VariantTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.memory = MemoryStore(Path(folder.name) / "memory.sqlite3")

    def test_assignments_use_the_current_variant_in_play_and_memory(self):
        self.assertCountEqual(SUPPORTED_VARIANTS, PLAYABLE_VARIANTS)
        for index, variant in enumerate(SUPPORTED_VARIANTS, start=1):
            with self.subTest(variant=variant):
                game = {**GAME, "game_id": f"{index:024x}", "rule_variant": variant}
                live = state(ask=variant in {"berkeley_any", "english", "crazykrieg"})
                live["reserve_summary"] = {"white": {}, "black": {}}
                api = FakeAPI([live, live], [{"move_done": True}])
                provider = FakeProvider({"attempts": ["e2e4"], "hypotheses": [], "rationale": "Opening."})
                player = Player(game, api, self.memory, provider, now=lambda: 100.0)
                projected, sampled = player.poll()
                player.play(projected, sampled, threading.Event())
                self.assertEqual(projected["rule_variant"], variant)
                request, = provider.requests
                self.assertEqual(request["context"]["ruleset"], variant)
                self.assertIn(rules_for(variant), request["instructions"])
                self.assertEqual(api.calls[-1], ("POST", f"/game/{game['game_id']}/move", {"uci": "e2e4"}))
                self.memory.mark_completed(game["game_id"])
                journal = next(item for item in self.memory.pending_games() if item["game_id"] == game["game_id"])
                self.assertEqual(journal["variant"], variant)

    def test_crazykrieg_projects_public_pockets_and_hides_opponent_drop_square(self):
        live = state(actions=("P@e3", "n@d4", "K@e3", "e2e4"))
        live["reserve_summary"] = {"white": {"pawns": 1}, "black": {"queens": 2}}
        live["referee_turns"][0]["black"][0].update(
            message="Opponent move — Queen dropped", move_uci="Q@h7",
        )
        live["full_fen"] = "opponent pieces"
        projected = project(live, "white", "crazykrieg")
        self.assertEqual(projected["allowed_moves"], ["P@e3", "e2e4"])
        self.assertEqual(projected["reserve_summary"]["white"]["pawns"], 1)
        self.assertEqual(projected["reserve_summary"]["black"]["queens"], 2)
        drop = projected["referee_turns"][0]["black"][0]
        self.assertEqual(drop["message"], "Opponent move — Queen dropped")
        self.assertNotIn("move_uci", drop)
        self.assertNotIn("full_fen", projected)
        classical = project(live, "white", "berkeley_any")
        self.assertEqual(classical["allowed_moves"], ["e2e4"])
        self.assertNotIn("reserve_summary", classical)

    def test_low_clock_can_submit_a_crazykrieg_drop(self):
        live = state(remaining=3, actions=("N@e3", "e2e4"), ask=False)
        live["reserve_summary"] = {"white": {"knights": 1}, "black": {}}
        projected = project(live, "white", "crazykrieg")
        self.assertEqual(fallback_rank(projected, {"e2e4"}), ["N@e3"])

        game = {**GAME, "rule_variant": "crazykrieg"}
        self.memory.open_game(
            game["game_id"], variant="crazykrieg", own_color="white", opponent_key=None, opponent_role="bot",
        )
        self.memory.record_attempt(game["game_id"], turn_key="white:3", action="e2e4", answer=None)
        provider = FakeProvider()
        api = FakeAPI([], [{"move_done": True, "dropped_piece_announcement": "KNIGHT", "full_fen": "hidden"}])
        Player(game, api, self.memory, provider, now=lambda: 100.0).play(projected, 100, threading.Event())
        self.assertEqual(provider.requests, [])
        self.assertEqual(api.calls, [("POST", f"/game/{game['game_id']}/move", {"uci": "N@e3"})])
        receipt = next(entry["answer"] for entry in self.memory.review_context(game["game_id"]) if entry.get("action") == "N@e3")
        self.assertEqual(receipt["dropped_piece_announcement"], "KNIGHT")
        self.assertNotIn("full_fen", receipt)

    def test_promoted_crazykrieg_fen_uses_only_own_piece_markers(self):
        live = state(actions=("e8d8",), ask=False)
        live["reserve_summary"] = {"white": {}, "black": {}}
        live["your_fen"] = "4Q~3/8/8/8/8/8/8/4K3 w - - 0 1"
        projected = project(live, "white", "crazykrieg")
        self.assertEqual(projected["your_fen"], live["your_fen"])
        self.assertEqual(fallback_rank(projected, set()), ["e8d8"])
        with self.assertRaises(ValueError):
            project(live, "white", "berkeley")
        for malformed in ("4q~3", "4Q~~3", "4~Q3"):
            with self.subTest(fen_rank=malformed), self.assertRaises(ValueError):
                project({**live, "your_fen": live["your_fen"].replace("4Q~3", malformed)}, "white", "crazykrieg")

    def test_startup_advertises_every_playable_variant(self):
        api = Mock()
        worker = Worker(api, self.memory, Mock(model="test-sol"), Mock(empty=lambda: True))
        worker.stop.set()
        worker.run()
        api.post.assert_called_once()
        path, payload = api.post.call_args.args
        self.assertEqual(path, "/bots/profile")
        self.assertCountEqual(payload["supported_rule_variants"], PLAYABLE_VARIANTS)
