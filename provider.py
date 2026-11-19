"""Bounded OpenAI requests and shared provider accounting."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from sqlite3 import Error as SQLiteError

import requests

from clock_policy import MIN_PLAY_REQUEST_SECONDS
from provider_budget import MonthlyBudgetLedger, estimate_request_cost_upper_bound_usd

LOG = logging.getLogger(__name__)
BASE = "https://api.openai.com/v1"
INPUT_PRICE = 2.0
CACHE_PRICE = 0.10
CACHE_WRITE_PRICE = 2.50
OUTPUT_PRICE = 10.0
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 256
EMBEDDING_INPUT_PRICE = 0.02
MAX_PLAY_ADMISSION_SECONDS = 3.0
MIN_REQUEST_SECONDS = 0.25


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
    if not inputs and not outputs:
        return None
    details = usage.get("input_tokens_details") or {}
    cached = min(inputs, details.get("cached_tokens") or 0)
    written = min(inputs - cached, details.get("cache_write_tokens") or 0)
    cost = (
        (inputs - cached - written) * INPUT_PRICE + cached * CACHE_PRICE
        + written * CACHE_WRITE_PRICE + outputs * OUTPUT_PRICE
    ) / 1_000_000
    return {
        "provider": "openai", "model": model, "response_id": response.get("id"),
        "input_tokens": inputs, "output_tokens": outputs, "cached_input_tokens": cached,
        "cache_creation_input_tokens": written, "total_tokens": inputs + outputs, "cost_usd": cost,
    }


def _embedding_vectors(response: dict, count: int) -> list[list[float]]:
    data = response.get("data") if isinstance(response, dict) else None
    if not isinstance(data, list) or len(data) != count:
        raise ValueError("unexpected number of embeddings")
    vectors = {}
    for row in data:
        if not isinstance(row, dict):
            raise ValueError("invalid embedding row")
        index, values = row.get("index"), row.get("embedding")
        if type(index) is not int or not 0 <= index < count or index in vectors:
            raise ValueError("invalid embedding index")
        if not isinstance(values, list) or len(values) != EMBEDDING_DIMENSIONS:
            raise ValueError("invalid embedding dimensions")
        if any(type(value) not in (int, float) for value in values):
            raise ValueError("invalid embedding coordinates")
        try:
            norm = math.hypot(*values)
        except OverflowError as exc:
            raise ValueError("invalid embedding magnitude") from exc
        if not norm or not math.isfinite(norm):
            raise ValueError("invalid embedding magnitude")
        vectors[index] = [value / norm for value in values]
    return [vectors[index] for index in range(count)]


class Provider:
    def __init__(self, key: str, model: str, ledger: MonthlyBudgetLedger, report_live, *, record_timing=None) -> None:
        self.headers = {"Authorization": f"Bearer {key}"}
        self.model = model
        self.ledger = ledger
        self.report_live = report_live
        self.record_timing = record_timing
        self.gates = {
            "play": threading.BoundedSemaphore(2),
            "review": threading.BoundedSemaphore(1),
            "lesson_index": threading.BoundedSemaphore(1),
            "lesson_search": threading.BoundedSemaphore(1),
        }
        self.retry_ready_at = dict.fromkeys(self.gates, 0.0)

    def availability(self) -> tuple[bool, str]:
        try:
            if time.monotonic() < self.retry_ready_at["play"]:
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
        return self._start_request(
            purpose=purpose, deadline=deadline,
            request=lambda: self._request(
                purpose=purpose, game_id=game_id, instructions=instructions, context=context,
                schema=schema, deadline=deadline, effort=effort, max_tokens=max_tokens,
            ),
        )

    def start_embedding(
        self, *, inputs: list[str], deadline: float, purpose: str, game_id: str,
    ) -> Future | None:
        if purpose not in ("lesson_index", "lesson_search"):
            raise ValueError("unknown embedding purpose")
        if not inputs or any(not isinstance(value, str) or not value.strip() for value in inputs):
            raise ValueError("embeddings require non-empty text")
        payload = {
            "model": EMBEDDING_MODEL, "input": list(inputs),
            "dimensions": EMBEDDING_DIMENSIONS, "encoding_format": "float",
        }
        return self._start_request(
            purpose=purpose, deadline=deadline,
            request=lambda: self._request_embedding(
                purpose=purpose, game_id=game_id, payload=payload, deadline=deadline,
            ),
        )

    def _start_request(
        self, *, purpose: str, deadline: float, request: Callable[[], dict | list[list[float]]],
    ) -> Future | None:
        minimum = MIN_PLAY_REQUEST_SECONDS if purpose == "play" else MIN_REQUEST_SECONDS
        now = time.monotonic()
        spare = deadline - now - minimum
        if spare <= 0 or now < self.retry_ready_at[purpose]:
            return None
        gate = self.gates[purpose]
        if not gate.acquire(blocking=False):
            if purpose != "play" or not gate.acquire(timeout=min(MAX_PLAY_ADMISSION_SECONDS, spare)):
                return None
        if deadline - time.monotonic() <= minimum:
            gate.release()
            return None
        future: Future = Future()

        def run():
            try:
                if future.set_running_or_notify_cancel():
                    future.set_result(request())
            except Exception as exc:
                future.set_exception(exc)
            finally:
                gate.release()

        try:
            threading.Thread(target=run, name=f"provider-{purpose}", daemon=True).start()
        except RuntimeError:
            gate.release()
            LOG.warning("provider worker unavailable purpose=%s", purpose)
            return None
        return future

    def _request(
        self, *, purpose: str, game_id: str, instructions: str, context: dict,
        schema: dict, deadline: float, effort: str, max_tokens: int,
    ) -> dict:
        payload = {
            "model": self.model,
            "store": False,
            "input": [
                {"role": "developer", "content": instructions},
                {"role": "user", "content": json.dumps(context, separators=(",", ":"), ensure_ascii=True)},
            ],
            "service_tier": "default",
            "reasoning": {"effort": effort},
            "max_output_tokens": max_tokens,
            "text": {"format": {"type": "json_schema", "name": f"kriegspiel_{purpose}", "strict": True, "schema": schema}},
        }
        now = time.monotonic()
        if now < self.retry_ready_at[purpose]:
            raise ModelUnavailable("provider backoff")
        if deadline - now <= MIN_REQUEST_SECONDS:
            raise ModelUnavailable("deadline elapsed before budget reservation")
        maximum_cost = estimate_request_cost_upper_bound_usd(
            payload, input_usd_per_million_tokens=CACHE_WRITE_PRICE,
            output_usd_per_million_tokens=OUTPUT_PRICE, maximum_output_tokens=max_tokens,
        )
        reservation = self.ledger.reserve(maximum_cost)
        if reservation is None:
            raise ModelUnavailable("shared budget exhausted")
        usage = None
        charged = 0.0
        started, latency = None, None
        timed_out, valid_response = False, False
        try:
            seconds = deadline - time.monotonic()
            if seconds <= MIN_REQUEST_SECONDS:
                raise ModelUnavailable("deadline elapsed before provider request")
            charged = None  # A transport failure can still incur provider charges.
            connect_timeout = min(2.0, seconds / 3)
            read_timeout = seconds - connect_timeout if purpose == "play" else seconds * 2 / 3
            started = time.monotonic()
            response = requests.post(
                f"{BASE}/responses", headers=self.headers, json=payload,
                timeout=(connect_timeout, read_timeout), allow_redirects=False,
            )
            if response.status_code == 429:
                self.retry_ready_at[purpose] = time.monotonic() + 60
            if 400 <= response.status_code < 500:
                charged = 0.0
            response.raise_for_status()
            if response.status_code != 200:
                raise ModelUnavailable(f"unexpected provider status {response.status_code}")
            latency = time.monotonic() - started
            data = response.json()
            usage = usage_record(data, self.model)
            if usage is not None:
                charged = usage["cost_usd"]
            plan = response_text(data)
            valid_response = True
            return plan
        except requests.Timeout:
            if started is not None:
                latency, timed_out = time.monotonic() - started, True
            raise
        finally:
            if purpose == "play" and (valid_response or timed_out) and self.record_timing is not None:
                try:
                    self.record_timing(self.model, effort, latency, timed_out=timed_out)
                except (OSError, SQLiteError, ValueError):
                    LOG.warning("could not persist provider timing model=%s effort=%s", self.model, effort)
            self.ledger.settle(reservation, charged)
            LOG.info(
                "usage purpose=%s game=%s model=%s effort=%s latency_ms=%s cost_usd=%.6f accounting=%s input_tokens=%s output_tokens=%s",
                purpose, game_id, self.model, effort, round(latency * 1000) if latency is not None else None, charged if charged is not None else reservation.amount_microusd / 1_000_000,
                "reported" if usage else "reservation" if charged is None else "unbilled",
                usage["input_tokens"] if usage else 0, usage["output_tokens"] if usage else 0,
            )
            if purpose == "play" and usage is not None:
                self.report_live({"game_id": game_id, **usage})

    def _request_embedding(
        self, *, purpose: str, game_id: str, payload: dict, deadline: float,
    ) -> list[list[float]]:
        now = time.monotonic()
        if now < self.retry_ready_at[purpose]:
            raise ModelUnavailable("provider backoff")
        if deadline - now <= MIN_REQUEST_SECONDS:
            raise ModelUnavailable("deadline elapsed before budget reservation")
        maximum_cost = estimate_request_cost_upper_bound_usd(
            payload, input_usd_per_million_tokens=EMBEDDING_INPUT_PRICE,
            output_usd_per_million_tokens=0.0, maximum_output_tokens=0,
        )
        reservation = self.ledger.reserve(maximum_cost)
        if reservation is None:
            raise ModelUnavailable("shared budget exhausted")

        inputs = None
        charged = 0.0
        started = None
        try:
            seconds = deadline - time.monotonic()
            if seconds <= MIN_REQUEST_SECONDS:
                raise ModelUnavailable("deadline elapsed before provider request")
            charged = None
            connect_timeout = min(2.0, seconds / 3)
            started = time.monotonic()
            response = requests.post(
                f"{BASE}/embeddings", headers=self.headers, json=payload,
                timeout=(connect_timeout, seconds - connect_timeout), allow_redirects=False,
            )
            if response.status_code == 429:
                self.retry_ready_at[purpose] = time.monotonic() + 60
            if 400 <= response.status_code < 500:
                charged = 0.0
            response.raise_for_status()
            if response.status_code != 200:
                raise ModelUnavailable(f"unexpected provider status {response.status_code}")
            data = response.json()
            usage = data.get("usage") if isinstance(data, dict) else None
            if isinstance(usage, dict) and type(usage.get("prompt_tokens")) is int and usage["prompt_tokens"] >= 0:
                inputs = usage["prompt_tokens"]
                charged = inputs * EMBEDDING_INPUT_PRICE / 1_000_000
            return _embedding_vectors(data, len(payload["input"]))
        finally:
            latency_ms = round((time.monotonic() - started) * 1000) if started is not None else None
            self.ledger.settle(reservation, charged)
            LOG.info(
                "usage purpose=%s game=%s model=%s latency_ms=%s cost_usd=%.8f accounting=%s input_tokens=%s output_tokens=0",
                purpose, game_id, EMBEDDING_MODEL, latency_ms,
                charged if charged is not None else reservation.amount_microusd / 1_000_000,
                "reported" if inputs is not None else "reservation" if charged is None else "unbilled",
                inputs if inputs is not None else 0,
            )
