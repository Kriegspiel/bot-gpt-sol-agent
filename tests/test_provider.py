import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, call, patch

import requests

from provider import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, ModelUnavailable, Provider, response_text
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


def embedding_response(*rows):
    mock_response = Mock(status_code=200)
    mock_response.json.return_value = {
        "data": [{"index": index, "embedding": vector} for index, vector in rows],
        "usage": {"prompt_tokens": 100, "total_tokens": 100},
    }
    return mock_response


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

    def test_live_max_request_uses_remaining_deadline(self):
        provider = Provider("mock", "gpt-6.2-sol", Mock(), Mock())
        mock_response = Mock(status_code=200)
        mock_response.json.return_value = response()
        with patch("provider.time.monotonic", return_value=100), patch(
            "provider.requests.post", return_value=mock_response,
        ) as post:
            provider._request(
                purpose="play", game_id="ABC234", instructions="test", context={},
                schema={}, deadline=170, effort="max", max_tokens=8192,
            )
        request = post.call_args.kwargs
        self.assertEqual(request["json"]["reasoning"], {"effort": "max"})
        self.assertEqual(request["json"]["max_output_tokens"], 8192)
        self.assertEqual(request["timeout"], (2, 68))

    def test_live_latency_is_recorded_with_a_distinct_timeout_flag(self):
        with TemporaryDirectory() as directory:
            ledger = MonthlyBudgetLedger(Path(directory) / "openai.json", limit_usd=1)
            recorder = Mock()
            provider = Provider("mock", "gpt-6.2-sol", ledger, Mock(), record_timing=recorder)
            mock_response = Mock(status_code=200)
            mock_response.json.return_value = response()
            now = [100]

            def respond(*_args, **_kwargs):
                now[0] += 20
                return mock_response

            def timeout(*_args, **_kwargs):
                now[0] += 35
                raise requests.ReadTimeout()

            def request(purpose):
                return provider._request(
                    purpose=purpose, game_id="ABC234", instructions="test", context={},
                    schema={}, deadline=now[0] + 70, effort="max", max_tokens=8192,
                )

            with patch("provider.time.monotonic", side_effect=lambda: now[0]):
                with patch("provider.requests.post", side_effect=respond):
                    request("play")
                    request("review")
                    mock_response.json.return_value = {**response(), "status": "incomplete"}
                    with self.assertRaises(ValueError):
                        request("play")
                with patch("provider.requests.post", side_effect=timeout):
                    with self.assertRaises(requests.ReadTimeout):
                        request("play")

            self.assertEqual(recorder.call_args_list, [
                call("gpt-6.2-sol", "max", 20, timed_out=False),
                call("gpt-6.2-sol", "max", 35, timed_out=True),
            ])

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

    def test_embeddings_preserve_input_order_and_settle_shared_budget(self):
        with TemporaryDirectory() as directory:
            ledger = MonthlyBudgetLedger(Path(directory) / "openai.json", limit_usd=1)
            live_report = Mock()
            provider = Provider("mock", "test-sol", ledger, live_report)
            first = [3, 4, *([0] * (EMBEDDING_DIMENSIONS - 2))]
            second = [0, -2, *([0] * (EMBEDDING_DIMENSIONS - 2))]
            inputs = ["lesson about pawn captures", "lesson about checking rays"]
            with self.assertLogs("provider", level="INFO") as logs, patch(
                "provider.requests.post", return_value=embedding_response((1, second), (0, first)),
            ) as post:
                future = provider.start_embedding(
                    inputs=inputs, deadline=time.monotonic() + 2, purpose="lesson_index", game_id="ABC234",
                )
                vectors = future.result(timeout=2)

            self.assertEqual(vectors[0], [0.6, 0.8, *first[2:]])
            self.assertEqual(vectors[1], [0.0, -1.0, *second[2:]])
            self.assertEqual(post.call_args.args, ("https://api.openai.com/v1/embeddings",))
            self.assertEqual(post.call_args.kwargs["json"], {
                "model": EMBEDDING_MODEL, "input": inputs,
                "dimensions": EMBEDDING_DIMENSIONS, "encoding_format": "float",
            })
            self.assertLessEqual(sum(post.call_args.kwargs["timeout"]), 2)
            self.assertAlmostEqual(ledger.status().spent_usd, 0.000002)
            self.assertEqual(ledger.status().reserved_usd, 0.0)
            self.assertIn("purpose=lesson_index game=ABC234 model=text-embedding-3-small", logs.output[0])
            self.assertIn("cost_usd=0.00000200 accounting=reported input_tokens=100", logs.output[0])
            live_report.assert_not_called()

    def test_bad_vectors_are_rejected_and_reported_usage_is_settled(self):
        valid = [1, *([0] * (EMBEDDING_DIMENSIONS - 1))]
        invalid_responses = (
            embedding_response((0, valid[:-1]), (1, valid)),
            embedding_response((0, valid), (0, valid)),
            embedding_response((0, [float("nan"), *valid[1:]]), (1, valid)),
            embedding_response((0, [0] * EMBEDDING_DIMENSIONS), (1, valid)),
        )
        with TemporaryDirectory() as directory:
            ledger = MonthlyBudgetLedger(Path(directory) / "openai.json", limit_usd=1)
            for response in invalid_responses:
                provider = Provider("mock", "test-sol", ledger, Mock())
                with self.subTest(response=response.json.return_value), patch(
                    "provider.requests.post", return_value=response,
                ):
                    future = provider.start_embedding(
                        inputs=["one", "two"], deadline=time.monotonic() + 2,
                        purpose="lesson_index", game_id="ABC234",
                    )
                    with self.assertRaises(ValueError):
                        future.result(timeout=2)
            self.assertAlmostEqual(ledger.status().spent_usd, 0.000002 * len(invalid_responses))
            self.assertEqual(ledger.status().reserved_usd, 0.0)

    def test_embedding_budget_and_deadline_can_skip_provider_request(self):
        ledger = Mock()
        ledger.reserve.return_value = None
        provider = Provider("mock", "test-sol", ledger, Mock())

        def start(seconds):
            return provider.start_embedding(
                inputs=["lesson query"], deadline=time.monotonic() + seconds,
                purpose="lesson_search", game_id="ABC234",
            )

        with patch("provider.requests.post") as post:
            self.assertIsNone(start(0))
            ledger.reserve.assert_not_called()
            future = start(2)
            with self.assertRaisesRegex(ModelUnavailable, "budget exhausted"):
                future.result(timeout=2)
            post.assert_not_called()
            ledger.settle.assert_not_called()

    def test_embedding_backoff_and_admission_are_separate_from_play(self):
        provider = Provider("mock", "test-sol", Mock(), Mock())
        limited = Mock(status_code=429)
        limited.raise_for_status.side_effect = requests.HTTPError()
        vector = [1, *([0] * (EMBEDDING_DIMENSIONS - 1))]

        def start(purpose):
            return provider.start_embedding(
                inputs=["lesson query"], deadline=time.monotonic() + 2, purpose=purpose, game_id="ABC234",
            )

        with patch("provider.requests.post", side_effect=[limited, embedding_response((0, vector))]) as post:
            gate = provider.gates["lesson_search"]
            gate.acquire()
            try:
                self.assertIsNone(start("lesson_search"))
                post.assert_not_called()
            finally:
                gate.release()
            with self.assertRaises(requests.HTTPError):
                start("lesson_search").result(timeout=2)
            self.assertIsNone(start("lesson_search"))
            self.assertEqual(start("lesson_index").result(timeout=2), [vector])
            self.assertEqual(provider.retry_ready_at["play"], 0.0)
            self.assertEqual(post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
