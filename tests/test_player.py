import threading
import unittest
from concurrent.futures import Future
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, call, patch

import requests

from bot import Player, Worker
from memory import MemoryStore
from play import TurnBudget, project
from rules import rules_for


GAME = {
    "game_id": "1234567890abcdeffedcba01", "game_code": "ABC234", "rule_variant": "berkeley_any",
    "white": {"username": "agent_gpt_sol", "role": "bot"},
    "black": {"username": "rival", "role": "user", "elo": 1800}, "opponent_id": "stable-id",
}


def state(*, remaining=90, actions=("d2d4", "e2e4"), ask=True, active_color="white"):
    return {
        "state": "active", "turn": "white", "your_color": "white", "move_number": 3, "ply_count": 4,
        "your_fen": "8/8/8/8/8/8/3PP3/4K3 w - - 0 1",
        "allowed_moves": list(actions), "possible_actions": ["move", "ask_any"] if ask else ["move"],
        "material_summary": {"white": {"pieces_remaining": 3}, "black": {"pieces_remaining": 5}},
        "clock": {"white_remaining": remaining, "black_remaining": 23, "active_color": active_color},
        "referee_turns": [
            {"turn": 2, "white": [], "black": [{"actor": "opponent", "kind": "move",
                                              "message": "Opponent move — Move complete", "move_uci": "e7e6"}]}
        ],
    }


def completed_future(value):
    future = Future()
    future.set_result(value)
    return future


class FakeProvider:
    model = "gpt-6.2-sol"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def start(self, **kwargs):
        self.requests.append(kwargs)
        value = self.responses.pop(0) if self.responses else None
        if isinstance(value, Future):
            return value
        return completed_future(value) if value is not None else None


class FakeAPI:
    def __init__(self, states, answers):
        self.states = list(states)
        self.answers = list(answers)
        self.calls = []

    def get(self, path, **kwargs):
        self.calls.append(("GET", path))
        return self.states.pop(0)

    def post(self, path, payload, **kwargs):
        self.calls.append(("POST", path, payload))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class PlayerTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.memory = MemoryStore(Path(folder.name) / "memory.sqlite3")

    def create(self, provider, api):
        return Player(GAME, api, self.memory, provider, now=lambda: 100.0)

    def test_live_requires_active_side_and_rejects_a_full_board(self):
        live = state()
        safe = project(live, "white")
        self.assertNotIn("move_uci", safe["referee_turns"][0]["black"][0])
        differing = state()
        differing["full_fen"] = "hidden other board"
        self.assertEqual(project(live, "white"), project(differing, "white"))
        for invalid in (
            {**live, "state": "completed"},
            {**live, "your_fen": "4k3/8/8/8/8/8/3PP3/4K3 w - - 0 1"},
            {**live, "your_color": "black"},
        ):
            with self.assertRaises(ValueError):
                project(invalid, "white")

    def test_low_time_uses_local_fallback_without_provider(self):
        live = state(remaining=3)
        api = FakeAPI([], [{"move_done": True}])
        provider = FakeProvider()
        self.create(provider, api).play(project(live, "white"), 100, threading.Event())
        self.assertEqual(provider.requests, [])
        self.assertEqual(len(api.calls), 1)
        self.assertIn(api.calls[0][2]["uci"], live["allowed_moves"])

    def test_live_play_sends_max_with_time_for_deeper_reasoning(self):
        live = state(remaining=1500, ask=False)
        api = FakeAPI([live], [{"move_done": True}])
        provider = FakeProvider({"attempts": ["e2e4"], "hypotheses": [], "rationale": "Contests the center."})
        self.create(provider, api).play(project(live, "white"), 100, threading.Event())
        self.assertEqual(provider.requests[0]["effort"], "max")
        self.assertEqual(api.calls[-1][2], {"uci": "e2e4"})

    def test_slow_measured_max_sends_a_lower_effort(self):
        live = state(remaining=1500, ask=False)
        api = FakeAPI([live], [{"move_done": True}])
        provider = FakeProvider({"attempts": ["e2e4"], "hypotheses": [], "rationale": ""})
        self.memory.record_model_timing(provider.model, "max", 80, timed_out=True)
        self.create(provider, api).play(project(live, "white"), 100, threading.Event())
        self.assertEqual(provider.requests[0]["effort"], "xhigh")

    def test_accepted_fallback_move_records_turn_elapsed(self):
        live = state(remaining=4, ask=False)
        api = FakeAPI([], [{"move_done": True}])
        original_post = api.post
        now = [100.0]

        def submit(*args, **kwargs):
            now[0] += 0.5
            return original_post(*args, **kwargs)

        api.post = submit
        player = Player(GAME, api, self.memory, FakeProvider(), now=lambda: now[0])
        player.play(project(live, "white"), 100, threading.Event())
        accepted = self.memory.review_context(GAME["game_id"])[-1]
        self.assertTrue(accepted["answer"]["move_done"])
        self.assertEqual(accepted["decision"]["elapsed_ms"], 500)

    def test_stale_model_output_is_discarded(self):
        live = state(remaining=350)
        api = FakeAPI([{**live, "turn": "black", "ply_count": 5}], [])
        provider = FakeProvider({"attempts": ["e2e4"], "hypotheses": ["possibility"], "rationale": "center"})
        self.create(provider, api).play(project(live, "white"), 100, threading.Event())
        self.assertEqual([call[0] for call in api.calls], ["GET"])
        prompt = provider.requests[0]["context"]
        self.assertEqual(prompt["opponent"], {"username": "rival", "role": "user", "elo": 1800})
        self.assertNotIn("opponent_id", prompt)
        self.assertNotIn("e7e6", str(prompt))

    def test_model_stall_falls_back(self):
        class StalledFuture(Future):
            def result(self, timeout=None):
                self.sampled_timeout = timeout
                raise TimeoutError

        future = StalledFuture()
        live = state()
        api = FakeAPI([live], [{"move_done": True}])
        provider = FakeProvider(future)
        self.create(provider, api).play(project(live, "white"), 100, threading.Event())
        self.assertEqual(provider.requests[0]["effort"], "low")
        self.assertLess(future.sampled_timeout, 18)
        self.assertEqual(api.calls[-1][0], "POST")

    def test_fallback_submission_survives_bounded_provider_admission(self):
        now = [100.0]
        live = state(remaining=1500)
        answer = {"move_done": True}
        api = FakeAPI([], [answer])
        provider = Mock(model="gpt-6.2-sol")

        def admission(*, deadline, **_):
            now[0] = deadline
            return None

        provider.start.side_effect = admission
        player = Player(GAME, api, self.memory, provider, now=lambda: now[0])
        player.play(project(live, "white"), 100, threading.Event())
        provider.start.assert_called_once()
        self.assertLess(now[0], player.budget.act_until)
        self.assertEqual(api.calls[0][0], "POST")

    def test_ask_any_reranks_under_same_turn_deadline(self):
        live = state(remaining=1000)
        after_any = {**state(remaining=990, actions=("d2e3",), ask=False), "ply_count": live["ply_count"] + 1}
        provider = FakeProvider(
            {"attempts": ["ask_any"], "hypotheses": ["A pawn capture may be possible"], "rationale": "information"},
            {"attempts": ["d2e3"], "hypotheses": [], "rationale": "must capture"},
        )
        api = FakeAPI([live, after_any, after_any], [
            {"move_done": False, "has_any": True, "announcement": "HAS_ANY"}, {"move_done": True},
        ])
        self.create(provider, api).play(project(live, "white"), 100, threading.Event())
        self.assertEqual([call[1].rsplit("/", 1)[1] for call in api.calls if call[0] == "POST"], ["ask-any", "move"])
        self.assertEqual(provider.requests[0]["deadline"], provider.requests[1]["deadline"])
        self.assertEqual(provider.requests[1]["context"]["available_attempts"], ["d2e3"])

    def test_ambiguous_post_is_not_retried_after_restart(self):
        live = state(remaining=4, ask=False)
        provider = FakeProvider()
        first = FakeAPI([], [requests.Timeout()])
        second = FakeAPI([], [{"move_done": True}])
        self.create(provider, first).play(project(live, "white"), 100, threading.Event())
        self.create(provider, second).play(project(live, "white"), 100, threading.Event())
        self.assertNotEqual(first.calls[0][2], second.calls[0][2])

    def test_prior_referee_rejection_is_excluded_at_full_move_index(self):
        live = state(remaining=4, ask=False)
        live["referee_turns"][0]["white"].append({
            "actor": "self", "kind": "illegal_move", "message": "Refused", "move_uci": "e2e4",
        })
        api = FakeAPI([], [{"move_done": True}])
        self.create(FakeProvider(), api).play(project(live, "white"), 100, threading.Event())
        self.assertEqual(api.calls[0][2], {"uci": "d2d4"})

    def test_paused_first_move_has_cap(self):
        live = state(remaining=1500, active_color=None)
        budget = TurnBudget.start(project(live, "white"), 98, 100)
        self.assertEqual((budget.model_until, budget.act_until), (126, 130))

    def test_worker_activates_prior_lessons_before_discovering_games(self):
        provider = Mock(model="gpt-6.2-sol")
        provider.availability.return_value = (True, "ready")
        worker = Worker(Mock(), self.memory, provider, None)
        calls = Mock()

        with (
            patch.object(self.memory, "activate_prior_lessons", return_value=1) as activate,
            patch.object(worker, "discover", side_effect=worker.stop.set) as discover,
            patch("bot.threading.Thread"),
        ):
            calls.attach_mock(activate, "activate_prior_lessons")
            calls.attach_mock(discover, "discover")
            worker.run()

        self.assertEqual(calls.mock_calls, [call.activate_prior_lessons(), call.discover()])

    def test_review_fetch_requires_completed_owned_metadata(self):
        active = {**GAME, "state": "active"}
        completed = {**GAME, "state": "completed"}
        mismatch = {**completed, "white": {"username": "someone_else"}}
        api = FakeAPI([active, mismatch, completed, {"transcript": "completed replay"}], [])
        provider = FakeProvider({"lessons": []})
        worker = Worker(api, self.memory, provider, None)
        worker.runners["unrelated_active_game"] = Mock(is_alive=lambda: True)
        lookup = GAME["game_id"]
        self.assertIsNone(worker.fetch_review(lookup))
        self.assertIsNone(worker.fetch_review(lookup))
        self.assertEqual(worker.fetch_review(lookup), {"transcript": "completed replay"})
        self.assertEqual(sum(path.endswith("/review") for _, path in api.calls), 1)
        context = {"game_code": GAME["game_code"], "rule_variant": "english"}
        self.assertEqual(worker.model_review(context), {"lessons": []})
        self.assertEqual(provider.requests[0]["purpose"], "review")
        self.assertIn(rules_for("english"), provider.requests[0]["instructions"])


if __name__ == "__main__":
    unittest.main()
