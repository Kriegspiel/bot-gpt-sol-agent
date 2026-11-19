import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from bot import Player
from clock_policy import reasoning_expectations, remaining_turns
from memory import MemoryStore
from play import TurnBudget, project, reasoning_for
from test_player import FakeAPI, GAME, completed_future, state


class ClockPolicyTests(unittest.TestCase):
    def test_archive_horizon_uses_public_material_and_opponent_role(self):
        live = project(state(), "white")
        self.assertEqual(remaining_turns(live, "user"), 22)
        self.assertEqual(remaining_turns(live, "bot"), 101)

        live["rule_variant"] = "crazykrieg"
        self.assertEqual(remaining_turns(live, "user"), 45)
        live["material_summary"]["black"]["pieces_remaining"] = 13
        self.assertEqual(remaining_turns(live, "user"), 28)
        self.assertEqual(remaining_turns(live, "bot"), 28)

    def test_reasoning_uses_low_through_max_in_order(self):
        self.assertIsNone(reasoning_for(5.0))
        expected = [("low", 1024), ("medium", 2048), ("high", 4096), ("xhigh", 6144), ("max", 8192)]
        for seconds, rung in zip((5.1, 18, 30, 45, 60), expected, strict=True):
            with self.subTest(seconds=seconds):
                self.assertEqual(reasoning_for(seconds), rung)

    def test_cold_timing_gates_match_reasoning_ladder(self):
        expected = {"low": 5, "medium": 18, "high": 30, "xhigh": 45, "max": 60}
        self.assertEqual(reasoning_expectations(), expected)
        self.assertEqual(reasoning_expectations({}), expected)

    def test_fast_calls_lower_gate_only_after_three_completed_calls(self):
        cold = reasoning_expectations()["max"]
        fast = [(4, False), (5, False), (6, False)]
        self.assertEqual(reasoning_expectations({"max": fast[:2]})["max"], cold)
        learned = reasoning_expectations({"max": fast})["max"]
        self.assertLess(learned, cold)
        self.assertEqual(reasoning_for((cold + learned) / 2, {"max": fast}), ("max", 8192))
        self.assertIsNone(reasoning_for(5, {"max": fast}))

    def test_slow_completion_and_timeout_raise_gate_immediately(self):
        cold = reasoning_expectations()["high"]
        for timed_out in (False, True):
            with self.subTest(timed_out=timed_out):
                timing = {"high": [(40, timed_out)]}
                self.assertGreater(reasoning_expectations(timing)["high"], max(cold, 40))
                self.assertEqual(reasoning_for(35, timing), ("medium", 2048))

    def test_timeout_prevents_three_fast_calls_from_overriding_it(self):
        timing = {"max": [(4, False), (5, False), (6, False), (120, True)]}
        self.assertGreater(reasoning_expectations(timing)["max"], 120)
        self.assertEqual(reasoning_for(80, timing), ("xhigh", 6144))

    def test_very_slow_max_is_skipped_even_with_a_large_budget(self):
        timing = {"max": [(120, False)]}
        self.assertEqual(reasoning_for(80, timing), ("xhigh", 6144))

    def test_high_clock_can_select_max_even_with_long_bot_horizon(self):
        live = project(state(remaining=1500), "white")
        for side in ("white", "black"):
            live["material_summary"][side]["pieces_remaining"] = 16
        budget = TurnBudget.start(live, 100, 100, opponent_role="bot")
        self.assertEqual(reasoning_for(budget.model_until - 100), ("max", 8192))
        self.assertLess(budget.model_until, budget.act_until)

    def test_model_and_submission_respect_current_clock(self):
        live = project(state(remaining=8), "white")
        budget = TurnBudget.start(live, 100, 103, opponent_role="user")
        self.assertLessEqual(budget.model_until, budget.act_until)
        self.assertLess(budget.act_until, 108)
        self.assertIsNone(reasoning_for(budget.model_until - 103))
        expired = TurnBudget.start(live, 100, 109, opponent_role="user")
        self.assertEqual((expired.model_until, expired.act_until), (109, 109))


class AdaptivePlayerTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.memory = MemoryStore(Path(folder.name) / "memory.sqlite3")

    def test_referee_rejection_uses_same_deadline_and_lower_effort(self):
        live = state(remaining=1500, ask=False)
        rejected = {**state(remaining=1440, actions=("d2d4",), ask=False), "ply_count": live["ply_count"] + 1}
        api = FakeAPI([live, rejected, rejected], [{"move_done": False}, {"move_done": True}])
        now = [100.0]

        def first_plan(*, deadline, **_):
            now[0] = deadline - 8
            return completed_future({"attempts": ["e2e4"], "hypotheses": [], "rationale": ""})

        def second_plan(**_):
            return completed_future({"attempts": ["d2d4"], "hypotheses": [], "rationale": ""})

        provider = Mock()
        plans = iter((first_plan, second_plan))
        provider.start.side_effect = lambda **request: next(plans)(**request)
        player = Player(GAME, api, self.memory, provider, now=lambda: now[0])
        player.play(project(live, "white"), 100, threading.Event())

        first, second = [call.kwargs for call in provider.start.call_args_list]
        self.assertEqual([first["effort"], second["effort"]], ["max", "low"])
        self.assertEqual(first["deadline"], second["deadline"])
        self.assertEqual([call[2]["uci"] for call in api.calls if call[0] == "POST"], ["e2e4", "d2d4"])

    def test_below_low_uses_deterministic_fallback(self):
        live = state(remaining=3, actions=("e2e4", "d2d4"))
        api = FakeAPI([], [{"move_done": True}])
        provider = Mock()
        player = Player(GAME, api, self.memory, provider, now=lambda: 100.0)
        player.play(project(live, "white"), 100, threading.Event())
        provider.start.assert_not_called()
        self.assertEqual([call[2]["uci"] for call in api.calls if call[0] == "POST"], ["d2d4"])
