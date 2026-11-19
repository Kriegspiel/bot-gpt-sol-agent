from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from memory import MemoryStore


class TimingStoreTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "memory.sqlite3"
        self.memory = MemoryStore(self.path)

    def test_timing_persists_independently_of_game_lessons(self):
        self.memory.set_enabled(False)
        self.memory.record_model_timing("gpt-6.2-sol", "high", 12, timed_out=False)
        self.memory.record_model_timing("gpt-6.2-sol", "high", 19.2, timed_out=True)

        restarted = MemoryStore(self.path, use_memory=False)
        self.assertFalse(restarted.is_enabled())
        self.assertEqual(restarted.model_timings("gpt-6.2-sol"), {"high": [(19.2, True), (12.0, False)]})
        self.assertEqual(
            restarted.status()["model_timing_counts"],
            [{"model": "gpt-6.2-sol", "effort": "high", "samples": 2, "timeouts": 1}],
        )

    def test_latest_twenty_are_kept_for_each_model_and_effort(self):
        self.memory.record_model_timing("gpt-6.2-sol", "low", 2, timed_out=False)
        self.memory.record_model_timing("gpt-6.1-sol", "high", 11, timed_out=False)
        for duration in range(1, 26):
            self.memory.record_model_timing("gpt-6.2-sol", "high", duration, timed_out=False)

        self.assertEqual(
            self.memory.model_timings("gpt-6.2-sol"),
            {"low": [(2.0, False)], "high": [(float(i), False) for i in range(25, 5, -1)]},
        )
        self.assertEqual(self.memory.model_timings("gpt-6.1-sol"), {"high": [(11.0, False)]})
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM model_timings").fetchone(), (22,))

    def test_expired_samples_are_ignored_then_pruned_on_write(self):
        observed_at = 10_000
        with patch("memory.time.time", return_value=observed_at):
            self.memory.record_model_timing("gpt-6.1-sol", "max", 22, timed_out=False)
        with patch("memory.time.time", return_value=observed_at + 7 * 24 * 60 * 60 + 1):
            self.assertEqual(self.memory.model_timings("gpt-6.1-sol"), {})
            self.assertEqual(self.memory.status()["model_timing_counts"], [])
            self.memory.record_model_timing("gpt-6.2-sol", "max", 23, timed_out=False)
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT model FROM model_timings").fetchall(), [("gpt-6.2-sol",)])

    def test_invalid_samples_do_not_affect_timing_history(self):
        for model, effort, duration, timed_out in (
            ("", "low", 5, False),
            ("bad\nmodel", "low", 5, False),
            ("gpt-6.2-sol", "minimal", 5, False),
            ("gpt-6.2-sol", "low", float("nan"), False),
            ("gpt-6.2-sol", "low", float("inf"), False),
            ("gpt-6.2-sol", "low", -1, False),
            ("gpt-6.2-sol", "low", 301, False),
            ("gpt-6.2-sol", "low", True, False),
            ("gpt-6.2-sol", "low", 5, 1),
        ):
            with self.subTest(model=model, effort=effort, duration=duration, timed_out=timed_out):
                self.memory.record_model_timing(model, effort, duration, timed_out=timed_out)
        self.assertEqual(self.memory.model_timings("gpt-6.2-sol"), {})
        self.assertEqual(self.memory.status()["model_timing_counts"], [])


if __name__ == "__main__":
    unittest.main()
