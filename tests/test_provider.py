import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import requests

from provider import Provider, response_text
from provider_budget import MonthlyBudgetLedger


def response(output=None):
    return {
        "status": "completed",
        "id": "resp_01",
        "output": [
            {"type": "reasoning", "content": [{"type": "output_text", "text": "private reasoning"}]},
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": json.dumps(output or {"attempts": ["e2e4"]})}
            ]},
        ],
        "usage": {"input_tokens": 1000, "output_tokens": 200, "input_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 300}},
    }


class ProviderTests(unittest.TestCase):
    @staticmethod
    def start(provider, purpose, game_id="ABC234", *, seconds=10):
        return provider.start(
            purpose=purpose, game_id=game_id, instructions="test", context={},
            schema={}, deadline=time.monotonic() + seconds, effort="low", max_tokens=200,
        )

    def test_only_completed_structured_output_is_parsed(self):
        self.assertEqual(response_text(response()), {"attempts": ["e2e4"]})
        for payload in ({**response(), "status": "incomplete"}, {**response(), "output": response()["output"][:1]}):
            with self.assertRaises(ValueError):
                response_text(payload)

    def test_shared_ledger_counts_live_and_review_but_only_reports_live(self):
        with TemporaryDirectory() as directory:
            ledger = MonthlyBudgetLedger(Path(directory) / "openai.json", limit_usd=1)
            reports = []
            provider = Provider("mock", "test-sol", ledger, reports.append)
            mock_response = Mock(status_code=200)
            mock_response.json.return_value = response()
            with patch("provider.requests.post", return_value=mock_response) as post:
                for purpose in ("play", "review"):
                    result = self.start(provider, purpose)
                    self.assertIsNotNone(result)
                    self.assertEqual(result.result(timeout=5), {"attempts": ["e2e4"]})
                self.assertEqual(post.call_count, 2)
                request = post.call_args.kwargs["json"]
                self.assertEqual(request["input"][0], {"role": "developer", "content": "test"})
                self.assertEqual(request["service_tier"], "default")
            self.assertAlmostEqual(ledger.status().spent_usd, 0.00792)
            self.assertAlmostEqual(reports[0]["cost_usd"], 0.00396)
            self.assertEqual(len(reports), 1)

    def test_two_live_requests_and_one_review_run_together(self):
        provider = Provider("mock", "test-sol", Mock(), Mock())
        started = {key: threading.Event() for key in ("LIVE22", "LIVE23", "PAST22")}
        finish = threading.Event()

        def request(*, game_id, **_):
            started[game_id].set()
            if not finish.wait(5):
                raise TimeoutError
            return {"ok": True}

        with patch.object(provider, "_request", side_effect=request):
            try:
                running = [
                    self.start(provider, "play", "LIVE22"),
                    self.start(provider, "play", "LIVE23"),
                    self.start(provider, "review", "PAST22"),
                ]
                self.assertTrue(all(event.wait(2) for event in started.values()))
                self.assertIsNone(self.start(provider, "review", "PAST23"))
                self.assertIsNone(self.start(provider, "play", "LIVE24", seconds=5.01))
            finally:
                finish.set()
            for future in running:
                self.assertEqual(future.result(timeout=2), {"ok": True})

    def test_saturated_live_admission_waits_with_both_bounds(self):
        provider = Provider("mock", "test-sol", Mock(), Mock())
        waiting = threading.Event()
        timeouts = []

        class ObservedSemaphore(threading.BoundedSemaphore):
            def acquire(self, blocking=True, timeout=None):
                if timeout is not None:
                    timeouts.append(timeout)
                    waiting.set()
                return super().acquire(blocking, timeout)

        gate = ObservedSemaphore(1)
        provider.gates["play"] = gate
        held = gate.acquire()
        try:
            with patch.object(provider, "_request", return_value={"ok": True}) as request:
                self.assertIsNone(self.start(provider, "play", seconds=4))
                self.assertEqual(timeouts, [])
                with ThreadPoolExecutor(max_workers=1) as scheduler:
                    entering = scheduler.submit(self.start, provider, "play", seconds=20)
                    self.assertTrue(waiting.wait(2))
                    self.assertEqual(timeouts[0], 3)
                    gate.release()
                    held = False
                    future = entering.result(timeout=2)
                    self.assertEqual(future.result(timeout=2), {"ok": True})
                request.assert_called_once()
        finally:
            if held:
                gate.release()

        busy = Mock()
        busy.acquire.return_value = False
        provider.gates["play"] = busy
        with patch("provider.time.monotonic", return_value=100):
            self.assertIsNone(self.start(provider, "play", seconds=5.5))
        busy.acquire.assert_called_with(timeout=0.5)

    def test_review_rate_limit_does_not_block_live_play(self):
        provider = Provider("mock", "test-sol", Mock(), Mock())
        limited = Mock(status_code=429)
        limited.raise_for_status.side_effect = requests.HTTPError()
        healthy = Mock(status_code=200)
        healthy.json.return_value = response()
        with patch("provider.requests.post", side_effect=[limited, healthy]) as post:
            first = self.start(provider, "review")
            with self.assertRaises(requests.HTTPError):
                first.result(timeout=2)
            self.assertIsNone(self.start(provider, "review"))
            live = self.start(provider, "play")
            self.assertEqual(live.result(timeout=2), {"attempts": ["e2e4"]})
            self.assertEqual(post.call_count, 2)
        with patch("provider.requests.get", return_value=healthy):
            self.assertEqual(provider.availability(), (True, ""))


if __name__ == "__main__":
    unittest.main()
