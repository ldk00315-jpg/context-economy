"""Provider-neutral paired evaluation. A model runner supplies ALL call usage."""
from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

from . import jsoncodec as jc


@dataclass(frozen=True)
class CallUsage:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0

    def __post_init__(self):
        values = (self.input_tokens, self.output_tokens, self.cached_input_tokens, self.cache_write_tokens)
        if any(type(v) is not int or v < 0 for v in values):
            raise ValueError("Usage counts must be nonnegative integers")
        if self.cached_input_tokens + self.cache_write_tokens > self.input_tokens:
            raise ValueError("Cache reads/writes must be disjoint subsets of input tokens")

    def cost(self, *, input_per_million: Decimal, output_per_million: Decimal,
             read_multiplier: Decimal, write_multiplier: Decimal) -> Decimal:
        if min(input_per_million, output_per_million, read_multiplier, write_multiplier) < 0:
            raise ValueError("Rates must be nonnegative")
        normal = self.input_tokens - self.cached_input_tokens - self.cache_write_tokens
        weighted = normal + self.cached_input_tokens * read_multiplier + self.cache_write_tokens * write_multiplier
        return (weighted * input_per_million + self.output_tokens * output_per_million) / Decimal(1_000_000)


@dataclass(frozen=True)
class Task:
    id: str
    question: str
    context: str
    tools: dict[str, Callable]


@dataclass(frozen=True)
class ModelRun:
    output: str
    calls: tuple[CallUsage, ...]
    completed: bool = True
    error: str | None = None


@dataclass(frozen=True)
class Pair:
    baseline: Task
    optimized: Task
    expected_json: str


def strict_json_grade(output: str, expected: str) -> bool:
    """Complete JSON answer only: no partial credit, substrings, extra keys or missing rows."""
    try:
        return jc.equal(jc.loads(output), jc.loads(expected))
    except (ValueError, TypeError, KeyError):
        return False


def evaluate_pairs(pairs: list[Pair], runner: Callable[[Task], ModelRun], *,
                   repetitions: int = 1, seed: int = 20260908) -> dict:
    """Adapter owns provider calls and must report retries/retrieval/hidden reasoning usage.

    Only initial context differs. Both arms must otherwise use identical model settings.
    Exceptions have UNKNOWN usage, never zero-cost success. No LLM is built into this library.
    """
    if type(repetitions) is not int or repetitions <= 0 or not pairs:
        raise ValueError("Provide tasks and positive repetitions")
    rng = random.Random(seed)
    records = []
    for pair in pairs:
        if pair.baseline.id != pair.optimized.id or pair.baseline.question != pair.optimized.question:
            raise ValueError("Paired tasks must have the same id and question")
        jc.loads(pair.expected_json)
        for repetition in range(repetitions):
            arms = [("baseline", pair.baseline), ("optimized", pair.optimized)]
            rng.shuffle(arms)
            record = {"id": pair.baseline.id, "repetition": repetition, "order": [a for a, _ in arms]}
            for arm, task in arms:
                try:
                    result = runner(task)
                    if not isinstance(result, ModelRun):
                        raise TypeError("runner must return ModelRun")
                    record[arm] = {
                        "correct": result.completed and not result.error and strict_json_grade(result.output, pair.expected_json),
                        "output": result.output, "error": result.error,
                        "usage_known": bool(result.calls),
                        "input_tokens": sum(c.input_tokens for c in result.calls) if result.calls else None,
                        "output_tokens": sum(c.output_tokens for c in result.calls) if result.calls else None,
                        "cached_input_tokens": sum(c.cached_input_tokens for c in result.calls) if result.calls else None,
                        "cache_write_tokens": sum(c.cache_write_tokens for c in result.calls) if result.calls else None,
                        "call_count": len(result.calls),
                    }
                except Exception as exc:
                    record[arm] = {"correct": False, "error": type(exc).__name__, "usage_known": False,
                                   "input_tokens": None, "output_tokens": None, "call_count": None}
            records.append(record)
    n = len(records)
    baseline_correct = sum(bool(r["baseline"]["correct"]) for r in records)
    optimized_correct = sum(bool(r["optimized"]["correct"]) for r in records)
    regressions = sum(bool(r["baseline"]["correct"]) and not r["optimized"]["correct"] for r in records)
    all_known = all(r[a]["usage_known"] for r in records for a in ("baseline", "optimized"))
    totals = {a: sum(r[a]["input_tokens"] + r[a]["output_tokens"] for r in records)
              for a in ("baseline", "optimized")} if all_known else None
    return {"seed": seed, "pairs": n, "unique_tasks": len(pairs),
            "baseline_correct": baseline_correct, "optimized_correct": optimized_correct,
            "regressions": regressions, "usage_complete": all_known, "total_tokens": totals,
            "token_savings_pct": 100*(1-totals["optimized"]/totals["baseline"]) if totals and totals["baseline"] else None,
            # A descriptive score is NOT a non-inferiority test, especially for repeated tasks.
            "quality_claim": "observed_only_not_noninferiority", "records": records}
