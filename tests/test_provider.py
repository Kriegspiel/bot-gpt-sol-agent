import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

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
                    result = provider.start(
                        purpose=purpose, game_id="ABC234", instructions="test", context={},
                        schema={}, deadline=time.monotonic() + 5, effort="low", max_tokens=200,
                    )
                    self.assertIsNotNone(result)
                    self.assertEqual(result.result(timeout=5), {"attempts": ["e2e4"]})
                self.assertEqual(post.call_count, 2)
                request = post.call_args.kwargs["json"]
                self.assertEqual(request["input"][0], {"role": "developer", "content": "test"})
                self.assertEqual(request["service_tier"], "default")
            self.assertAlmostEqual(ledger.status().spent_usd, 0.00792)
            self.assertAlmostEqual(reports[0]["cost_usd"], 0.00396)
            self.assertEqual(len(reports), 1)


if __name__ == "__main__":
    unittest.main()
