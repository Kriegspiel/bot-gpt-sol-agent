"""Bounded OpenAI Responses requests and shared provider accounting."""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import Future

import requests

from provider_budget import MonthlyBudgetLedger, estimate_request_cost_upper_bound_usd

LOG = logging.getLogger(__name__)
BASE = "https://api.openai.com/v1"
INPUT_PRICE = 2.0
CACHE_PRICE = 0.10
OUTPUT_PRICE = 10.0


class ModelUnavailable(RuntimeError):
    pass


def response_text(response: dict) -> dict:
    if response.get("status") != "completed":
        raise ValueError("response incomplete")
    parts = [
        part.get("text", "")
        for item in response.get("output", [])
        if item.get("type") == "message" and item.get("role") == "assistant"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    ]
    if not parts:
        raise ValueError("response has no structured output")
    value = json.loads("".join(parts))
    if not isinstance(value, dict):
        raise ValueError("response must be an object")
    return value


def usage_record(response: dict, model: str) -> dict | None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    inputs = usage.get("input_tokens", 0)
    outputs = usage.get("output_tokens", 0)
    cached = min(inputs, (usage.get("input_tokens_details") or {}).get("cached_tokens", 0))
    return {
        "provider": "openai", "model": model, "response_id": response.get("id"),
        "input_tokens": inputs, "output_tokens": outputs, "cached_input_tokens": cached,
        "total_tokens": inputs + outputs,
        "cost_usd": ((inputs - cached) * INPUT_PRICE + cached * CACHE_PRICE + outputs * OUTPUT_PRICE) / 1_000_000,
    }


class Provider:
    def __init__(self, key: str, model: str, ledger: MonthlyBudgetLedger, report_live) -> None:
        self.headers = {"Authorization": f"Bearer {key}"}
        self.model = model
        self.ledger = ledger
        self.report_live = report_live
        self.gates = {"play": threading.Lock(), "review": threading.Lock()}
        self.retry_ready_at = 0.0

    def availability(self) -> tuple[bool, str]:
        try:
            if time.monotonic() < self.retry_ready_at:
                return False, "openai_rate_limited"
            if not self.ledger.status().remaining_microusd:
                return False, "openai_monthly_budget_exhausted"
            response = requests.get(
                f"{BASE}/models/{self.model}", headers=self.headers, timeout=(3, 5), allow_redirects=False,
            )
            return response.status_code == 200, "" if response.status_code == 200 else f"openai_metadata_{response.status_code}"
        except (OSError, RuntimeError, ValueError, requests.RequestException):
            return False, "openai_preflight_failed"

    def start(
        self, *, purpose: str, game_id: str, instructions: str, context: dict,
        schema: dict, deadline: float, effort: str, max_tokens: int,
    ) -> Future | None:
        gate = self.gates[purpose]
        if deadline <= time.monotonic() or not gate.acquire(blocking=False):
            return None
        future: Future = Future()

        def run():
            try:
                result = self._request(
                    purpose=purpose, game_id=game_id, instructions=instructions, context=context,
                    schema=schema, deadline=deadline, effort=effort, max_tokens=max_tokens,
                )
                future.set_result(result)
            except Exception as exc:
                future.set_exception(exc)
            finally:
                gate.release()

        threading.Thread(target=run, name=f"provider-{purpose}", daemon=True).start()
        return future

    def _request(
        self, *, purpose: str, game_id: str, instructions: str, context: dict,
        schema: dict, deadline: float, effort: str, max_tokens: int,
    ) -> dict:
        payload = {
            "model": self.model,
            "store": False,
            "instructions": instructions,
            "input": json.dumps(context, separators=(",", ":"), ensure_ascii=True),
            "reasoning": {"effort": effort},
            "max_output_tokens": max_tokens,
            "text": {"format": {"type": "json_schema", "name": f"kriegspiel_{purpose}", "strict": True, "schema": schema}},
        }
        if time.monotonic() < self.retry_ready_at:
            raise ModelUnavailable("provider backoff")
        maximum_cost = estimate_request_cost_upper_bound_usd(
            payload, input_usd_per_million_tokens=INPUT_PRICE,
            output_usd_per_million_tokens=OUTPUT_PRICE, maximum_output_tokens=max_tokens,
        )
        reservation = self.ledger.reserve(maximum_cost)
        if reservation is None:
            raise ModelUnavailable("shared budget exhausted")
        usage = None
        charged = 0.0
        try:
            seconds = deadline - time.monotonic()
            if seconds <= 0.25:
                raise ModelUnavailable("deadline elapsed before provider request")
            charged = None  # A transport failure can still incur provider charges.
            response = requests.post(
                f"{BASE}/responses", headers=self.headers, json=payload,
                timeout=(min(2.0, seconds / 3), seconds * 2 / 3), allow_redirects=False,
            )
            if response.status_code == 429:
                self.retry_ready_at = time.monotonic() + 60
            if 400 <= response.status_code < 500:
                charged = 0.0
            response.raise_for_status()
            if response.status_code != 200:
                raise ModelUnavailable(f"unexpected provider status {response.status_code}")
            data = response.json()
            usage = usage_record(data, self.model)
            if usage is not None:
                charged = usage["cost_usd"]
            return response_text(data)
        finally:
            self.ledger.settle(reservation, charged)
            LOG.info(
                "usage purpose=%s game=%s cost_usd=%.6f accounting=%s input_tokens=%s output_tokens=%s",
                purpose, game_id, charged if charged is not None else reservation.amount_microusd / 1_000_000,
                "reported" if usage else "reservation" if charged is None else "unbilled",
                usage["input_tokens"] if usage else 0, usage["output_tokens"] if usage else 0,
            )
            if purpose == "play" and usage is not None:
                self.report_live({"game_id": game_id, **usage})
