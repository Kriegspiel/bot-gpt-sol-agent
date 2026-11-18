"""Production runner for the GPT Sol Agent."""

from __future__ import annotations

import argparse
import logging
import os
import queue
import re
import signal
import threading
import time
from pathlib import Path
from sqlite3 import OperationalError

import requests

from memory import MemoryStore
from play import (
    PLAN_SCHEMA, PLAY_INSTRUCTIONS, VARIANT, TurnBudget, clock_seconds, fallback_rank, full_move_number,
    model_context, parse_plan, project, turn_key,
)
from provider import ModelUnavailable, Provider
from provider_budget import MonthlyBudgetLedger
from review import REVIEW_INSTRUCTIONS, REVIEW_OUTPUT_SCHEMA, ReviewWorker

LOG = logging.getLogger(__name__)
USERNAME = "agent_gpt_sol"
DISPLAY_NAME = "GPT Sol Agent"
DESCRIPTION = "Plays Berkeley + Any with clock-aware reasoning and lessons from completed games."
AUTHOR_NOTE = (
    "GPT Sol Agent reasons from its own board and the referee's announcements. "
    "It reviews its completed games and retains approved lessons, which may include opponent tendencies. "
    "Source: https://github.com/Kriegspiel/bot-gpt-sol-agent"
)
GAME_ID = re.compile(r"[a-zA-Z0-9]{6,32}")


class GameAPI:
    def __init__(self, base: str, token: str | None) -> None:
        self.base = base.rstrip("/")
        self.token = token
        self.local = threading.local()

    def call(self, method: str, path: str, *, payload: dict | None = None, timeout: float = 8) -> dict:
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        response = self.local.session.request(
            method, self.base + path, headers=headers, json=payload,
            timeout=timeout, allow_redirects=False,
        )
        response.raise_for_status()
        if not 200 <= response.status_code < 300:
            raise ValueError("unexpected API response status")
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("expected API object")
        return data

    def close(self) -> None:
        if hasattr(self.local, "session"):
            self.local.session.close()
            del self.local.session

    def get(self, path: str, *, timeout: float = 8) -> dict:
        return self.call("GET", path, timeout=timeout)

    def post(self, path: str, payload: dict, *, timeout: float = 8) -> dict:
        return self.call("POST", path, payload=payload, timeout=timeout)


def is_not_found(exc: requests.HTTPError) -> bool:
    return exc.response is not None and exc.response.status_code == 404


def player_color(game: dict) -> str | None:
    for side in ("white", "black"):
        if (game.get(side) or {}).get("username", "").lower() == USERNAME:
            return side
    return None


def ongoing_rejections(state: dict) -> set[str]:
    side = state["your_color"]
    return {
        item["move_uci"]
        for turn in state["referee_turns"][-2:]
        if turn["turn"] == full_move_number(state)
        for item in turn[side]
        if item.get("kind") == "illegal_move" and item.get("move_uci")
    }


class Player:
    def __init__(
        self, game: dict, api: GameAPI, memory: MemoryStore, provider: Provider,
        *, now=time.monotonic,
    ) -> None:
        self.game_id = game["game_id"]
        self.game_code = game["game_code"]
        self.color = player_color(game)
        opponent_color = "black" if self.color == "white" else "white"
        opponent = game[opponent_color]
        self.opponent = {key: opponent[key] for key in ("username", "role", "elo") if key in opponent}
        self.api = api
        self.memory = memory
        self.provider = provider
        self.now = now
        self.beliefs: list[str] = []
        self.current_turn = None
        self.budget = None
        self.model_calls = 0
        self.lessons = memory.open_game(
            self.game_id, game_code=self.game_code, variant=VARIANT, own_color=self.color,
            opponent_key=game.get("opponent_id"), opponent_role=opponent["role"],
            opponent_elo=opponent.get("elo"),
        )

    def poll(self, *, timeout: float = 4) -> tuple[dict, float]:
        sampled_at = self.now()
        state = self.api.get(f"/game/{self.game_id}/state", timeout=timeout)
        if state.get("state") != "active":
            return {"state": state.get("state")}, sampled_at
        return project(state, self.color), sampled_at

    def play(self, state: dict, sampled_at: float, stop: threading.Event) -> None:
        expected = turn_key(state)
        if self.current_turn != expected:
            self.current_turn = expected
            self.turn_started = self.now()
            self.budget = TurnBudget.start(state, sampled_at, self.turn_started)
            self.model_calls = 0
        budget = self.budget
        journal_key = f"{self.color}:{state['move_number']}"
        rejected = self.memory.tried_actions(self.game_id, journal_key) | ongoing_rejections(state)
        self.memory.record_observation(self.game_id, state)
        while not stop.is_set() and turn_key(state) == expected:
            candidates = fallback_rank(state, rejected)
            if not candidates:
                return

            model_seconds = budget.model_until - self.now()
            ranked, rationale = [], ""
            if model_seconds >= 5 and self.model_calls < 2:
                context = model_context(
                    state, attempts=candidates, beliefs=self.beliefs,
                    lessons=self.lessons, opponent=self.opponent, budget_seconds=model_seconds,
                )
                future = self.provider.start(
                    purpose="play", game_id=self.game_id, instructions=PLAY_INSTRUCTIONS,
                    context=context, schema=PLAN_SCHEMA, deadline=budget.model_until,
                    effort="medium" if model_seconds >= 18 else "low",
                    max_tokens=2048 if model_seconds >= 18 else 1024,
                )
                if future is not None:
                    self.model_calls += 1
                    try:
                        plan = future.result(timeout=max(0, budget.model_until - self.now()))
                        ranked, beliefs, rationale = parse_plan(plan, candidates)
                        if ranked:
                            self.beliefs = beliefs
                    except (TimeoutError, OSError, RuntimeError, ValueError, requests.RequestException):
                        LOG.info("play fallback game=%s cause=model_unavailable_or_deadline", self.game_code)
                    # The game may have ended or another attempt may have restricted this turn.
                    seconds = budget.act_until - self.now()
                    if seconds <= 0:
                        return
                    last_referee_index = state["ply_count"]
                    state, sampled_at = self.poll(timeout=min(1.5, seconds))
                    if state["state"] != "active" or turn_key(state) != expected:
                        return
                    if state["ply_count"] != last_referee_index:
                        ranked = []
                    candidates = fallback_rank(state, rejected | ongoing_rejections(state))
                    if not candidates:
                        return

            ranked = [attempt for attempt in ranked if attempt in candidates]
            action = (ranked or candidates)[0]
            remaining = min(budget.act_until - self.now(), clock_seconds(state, sampled_at, self.now()))
            if remaining <= 0 or stop.is_set():
                return
            rejected.add(action)
            source = "model" if ranked else "fallback"
            decision = {"source": source, "rationale": rationale if ranked else ""}
            self.memory.record_attempt(
                self.game_id, turn_key=journal_key, action=action, answer=None, decision=decision,
            )
            endpoint, data = ("ask-any", {}) if action == "ask_any" else ("move", {"uci": action})
            try:
                remaining = min(remaining, budget.act_until - self.now(), clock_seconds(state, sampled_at, self.now()))
                if remaining <= 0 or stop.is_set():
                    return
                answer = self.api.post(f"/game/{self.game_id}/{endpoint}", data, timeout=min(2, remaining))
            except requests.RequestException:
                LOG.info("play submission uncertain game=%s", self.game_code)
                # Resolve from the next server read. A journaled attempt isn't submitted again.
                return
            self.memory.record_attempt(
                self.game_id, turn_key=journal_key, action=action, decision=decision,
                answer={field: answer[field] for field in (
                    "move_done", "announcement", "special_announcement", "checks", "capture_square",
                    "game_over", "has_any", "clock",
                ) if field in answer},
            )
            LOG.info(
                "attempt game=%s source=%s accepted=%s elapsed_seconds=%.3f",
                self.game_code, source, answer.get("move_done"), self.now() - self.turn_started,
            )
            if answer.get("move_done") or answer.get("game_over"):
                return
            seconds = budget.act_until - self.now()
            if seconds <= 0:
                return
            state, sampled_at = self.poll(timeout=min(1.5, seconds))
            if state["state"] != "active" or turn_key(state) != expected:
                return
            self.memory.record_observation(self.game_id, state)

    def run(self, stop: threading.Event) -> None:
        try:
            while not stop.is_set():
                try:
                    state, sampled_at = self.poll()
                    if state["state"] == "completed":
                        self.memory.mark_completed(self.game_id)
                        return
                    if state["state"] != "active":
                        return
                    if state["turn"] == self.color:
                        self.play(state, sampled_at, stop)
                except requests.HTTPError as exc:
                    if exc.response is not None and exc.response.status_code in {403, 404, 409}:
                        return
                    LOG.warning("game poll HTTP error game=%s status=%s", self.game_code, exc.response.status_code)
                except (KeyError, OperationalError, ValueError, requests.RequestException):
                    LOG.warning("game poll failed game=%s", self.game_code)
                stop.wait(2)
        finally:
            self.api.close()


class Worker:
    def __init__(self, api: GameAPI, memory: MemoryStore, provider: Provider, reports: queue.Queue) -> None:
        self.api = api
        self.memory = memory
        self.provider = provider
        self.reports = reports
        self.runners: dict[str, threading.Thread] = {}
        self.stop = threading.Event()
        self.in_game = threading.Event()

    def fetch_review(self, game_id: str) -> dict | None:
        if self.in_game.is_set() or not GAME_ID.fullmatch(game_id):
            return None
        try:
            metadata = self.api.get(f"/game/{game_id}")
            if metadata.get("state") != "completed" or not player_color(metadata):
                return None
            review = self.api.get(f"/game/{game_id}/review")
        except requests.HTTPError as exc:
            if is_not_found(exc):
                return None
            raise
        return review

    def model_review(self, context: dict) -> dict:
        if self.in_game.is_set() or self.stop.is_set():
            raise ModelUnavailable("live game takes priority")
        deadline = time.monotonic() + 65
        future = self.provider.start(
            purpose="review", game_id=context["game_code"], context=context,
            instructions=REVIEW_INSTRUCTIONS, schema=REVIEW_OUTPUT_SCHEMA, deadline=deadline,
            effort="medium", max_tokens=4096,
        )
        if future is None:
            raise ModelUnavailable("review provider busy")
        return future.result(timeout=65)

    def review_loop(self) -> None:
        reviewer = ReviewWorker(self.memory, fetch_review=self.fetch_review, model_review=self.model_review)
        try:
            while not self.stop.wait(10):
                if self.in_game.is_set():
                    continue
                try:
                    reviewer.run_once()
                except (TimeoutError, RuntimeError, OSError, ValueError, requests.RequestException):
                    LOG.warning("postgame review postponed")
        finally:
            self.api.close()

    def report_loop(self) -> None:
        try:
            while not self.stop.is_set() or not self.reports.empty():
                try:
                    report = self.reports.get(timeout=1)
                except queue.Empty:
                    continue
                try:
                    self.api.post("/bots/usage", report, timeout=3)
                except requests.RequestException:
                    LOG.warning("usage report failed game=%s", report["game_id"])
        finally:
            self.api.close()

    def discover(self) -> None:
        games = self.api.get("/game/mine/active?limit=100")["games"]
        self.runners = {key: thread for key, thread in self.runners.items() if thread.is_alive()}
        for game in games:
            if game.get("state") != "active" or game.get("rule_variant") != VARIANT or not player_color(game):
                continue
            if game["game_id"] in self.runners or not GAME_ID.fullmatch(game["game_id"]):
                continue
            self.in_game.set()
            player = Player(game, self.api, self.memory, self.provider)
            thread = threading.Thread(target=player.run, args=(self.stop,), name=game["game_code"], daemon=True)
            self.runners[game["game_id"]] = thread
            thread.start()
        if not self.runners:
            self.in_game.clear()

    def run(self) -> None:
        self.api.post("/bots/profile", {
            "supported_rule_variants": [VARIANT],
            "author_note": f"OpenAI {self.provider.model}. {AUTHOR_NOTE}",
        })
        review_thread = threading.Thread(target=self.review_loop, name="postgame", daemon=True)
        report_thread = threading.Thread(target=self.report_loop, name="accounting", daemon=True)
        review_thread.start()
        report_thread.start()
        next_preflight = 0.0
        try:
            while not self.stop.is_set():
                try:
                    self.discover()
                    if time.monotonic() >= next_preflight:
                        ready, reason = self.provider.availability()
                        self.api.post("/bots/availability", {"provider": "openai", "ready": ready, "reason": reason})
                        next_preflight = time.monotonic() + 30
                except (KeyError, OperationalError, ValueError, requests.RequestException):
                    LOG.warning("assignment or availability poll failed")
                self.stop.wait(5)
        finally:
            self.stop.set()
            for thread in self.runners.values():
                thread.join(timeout=3)
            review_thread.join(timeout=1)
            report_thread.join(timeout=4)
            self.api.close()


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and re.fullmatch(r"[A-Z][A-Z0-9_]*", key.strip()):
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> None:
    default_state = Path.home() / ".local/state/kriegspiel/bot-gpt-sol-agent/memory.sqlite3"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--state-file", type=Path, default=default_state)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    load_env(args.env_file)
    os.umask(0o077)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    api = GameAPI(os.getenv("KRIEGSPIEL_API_BASE", "https://api.kriegspiel.org"), os.getenv("KRIEGSPIEL_BOT_TOKEN"))
    if os.getenv("KRIEGSPIEL_BOT_USERNAME", USERNAME) != USERNAME:
        parser.error(f"this worker requires the {USERNAME} account")
    if args.register_only:
        if api.token:
            parser.error("registration requires an empty bot token")
        registration = api.post("/auth/bots/register", {
            "username": USERNAME, "display_name": DISPLAY_NAME, "description": DESCRIPTION,
            "author_note": AUTHOR_NOTE, "listed": os.getenv("KRIEGSPIEL_BOT_LISTED", "false") == "true",
            "owner_email": os.getenv("KRIEGSPIEL_BOT_OWNER_EMAIL", "bot-gpt-sol-agent@kriegspiel.org"),
            "supported_rule_variants": [VARIANT],
        })
        print(registration["api_token"])
        return

    key = os.getenv("LLM_API_KEY")
    if not api.token or not key:
        parser.error("KRIEGSPIEL_BOT_TOKEN and LLM_API_KEY are required")
    account = api.get("/auth/me")
    if account.get("username") != USERNAME or account.get("role") != "bot":
        parser.error("bot token belongs to a different account")
    ledger = MonthlyBudgetLedger(
        os.getenv("OPENAI_MONTHLY_BUDGET_STATE_PATH", str(Path.home() / ".local/state/kriegspiel/provider-budgets/openai.json")),
        limit_usd=float(os.getenv("OPENAI_MONTHLY_BUDGET_USD", "18")),
    )
    reports: queue.Queue = queue.Queue()
    provider = Provider(key, os.getenv("LLM_MODEL", "gpt-6.2-sol"), ledger, reports.put_nowait)
    memory = MemoryStore(args.state_file, use_memory=os.getenv("AGENT_USE_MEMORY", "true") == "true")
    worker = Worker(api, memory, provider, reports)
    signal.signal(signal.SIGTERM, lambda *_: worker.stop.set())
    signal.signal(signal.SIGINT, lambda *_: worker.stop.set())
    worker.run()


if __name__ == "__main__":
    main()
