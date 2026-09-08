from __future__ import annotations

import json
from dataclasses import dataclass, replace
from decimal import Decimal, localcontext
from typing import Callable, Literal

from . import jsoncodec as jc
from .store import Store


@dataclass(frozen=True)
class View:
    text: str
    format: str
    coverage: str
    original_id: str | None
    original_tokens: int
    tokens: int
    within_budget: bool
    reason: str = ""

    @property
    def saved_tokens(self) -> int:
        return self.original_tokens - self.tokens

    def restore_semantic(self) -> str:
        if self.format == "table-v1":
            return jc.table_decode(self.text.split("\n", 1)[1])
        if self.format == "runs-v1":
            runs = json.loads(self.text.split("\n", 1)[1])
            return "".join(line * count for count, line in runs)
        if self.format in ("raw", "json"):
            return self.text
        raise ValueError("Partial/query views cannot reconstruct originals; use retrieve")


class Economy:
    def __init__(self, store: Store, count: Callable[[str], int]):
        self.store = store
        self.count = count

    def _view(self, text, fmt, coverage, ref, original, budget, reason=""):
        tokens = self.count(text)
        return View(text, fmt, coverage, ref, self.count(original), tokens,
                    budget is None or tokens <= budget, reason)

    @staticmethod
    def _budget(budget):
        if budget is not None and (type(budget) is not int or budget <= 0):
            raise ValueError("budget must be a positive integer")

    def pack(self, text: str, *, kind: Literal["json", "text", "code"] = "text",
             budget: int | None = None, byte_exact: bool = False) -> View:
        """Smallest verified complete representation. A budget never authorizes omission."""
        self._budget(budget)
        if kind not in ("json", "text", "code"):
            raise ValueError("kind must be json, text or code")
        original = self._view(text, "raw", "all", None, text, budget)
        try:
            ref = self.store.put(text)
        except Exception as exc:
            return replace(original, reason=f"store_failed:{type(exc).__name__}")
        original = replace(original, original_id=ref)
        if byte_exact or kind == "code":
            return replace(original, reason="byte_exact")
        candidates = [original]
        try:
            if kind == "json":
                value = jc.loads(text)
                normalized = jc.dumps(value)
                candidates.append(self._view(normalized, "json", "all", ref, text, budget))
                table = jc.table_encode(value)
                if table is not None:
                    # Count the complete model-visible schema explanation.
                    rendered = 'TABLE-V1 (all rows, in order; zip columns with each row to recover JSON objects):\n' + table
                    candidate = self._view(rendered, "table-v1", "all", ref, text, budget)
                    if candidate.restore_semantic() == normalized:
                        candidates.append(candidate)
            else:
                runs = []
                for line in text.splitlines(keepends=True):
                    if runs and runs[-1][1] == line:
                        runs[-1][0] += 1
                    else:
                        runs.append([1, line])
                rendered = 'RUNS-V1 (all lines, exact order/endings; each [count,text] repeats text count times):\n' + json.dumps(runs, ensure_ascii=False, separators=(",", ":"))
                candidate = self._view(rendered, "runs-v1", "all", ref, text, budget)
                if candidate.restore_semantic() == text:
                    candidates.append(candidate)
            winner = min(candidates, key=lambda c: c.tokens)
            if not winner.within_budget:
                winner = replace(winner, reason="budget_exceeded_complete_data_preserved")
            return winner
        except Exception as exc:
            return replace(original, reason=f"normalization_skipped:{type(exc).__name__}")

    def retrieve(self, original_id: str) -> str:
        """Byte-exact UTF-8 original, checked on every read."""
        return self.store.get(original_id)

    def _reply(self, ref, original, body, *, coverage, budget=None, reason=""):
        header = {"source": ref, "coverage": coverage, **body}
        return self._view(jc.dumps(header), "query-v1", coverage, ref, original, budget, reason)

    def aggregate(self, ref: str, *, op: Literal["count", "sum", "min", "max"],
                  path: str = "", field: str | None = None) -> View:
        """Exact declared operation over ALL entries. Reject missing/non-numeric values."""
        if op not in ("count", "sum", "min", "max"):
            raise ValueError("Unsupported aggregate")
        original = self.retrieve(ref)
        values = jc.pointer(jc.loads(original), path)
        if not isinstance(values, list):
            raise ValueError("Aggregate target must be a JSON array")
        if op == "count":
            if field is not None:
                raise ValueError("count does not accept field; counts all array entries")
            result = len(values)
        else:
            selected = [jc.pointer(v, field) for v in values] if field is not None else values
            if any(not isinstance(v, jc.Number) for v in selected):
                raise ValueError("Every selected value must be a JSON number; nothing is skipped")
            nums = [v.decimal() for v in selected]
            if op in ("min", "max") and not nums:
                raise ValueError("min/max of empty array is undefined")
            if op == "sum":
                if nums:
                    precision = max(n.adjusted() for n in nums) - min(n.as_tuple().exponent for n in nums) + len(str(len(nums))) + 4
                    if precision > 1_000_000 or any(abs(n.as_tuple().exponent) > 1_000_000 for n in nums):
                        raise ValueError("Numeric range exceeds exact aggregation resource limit")
                    with localcontext() as ctx:
                        ctx.prec = max(28, precision)
                        total = sum(nums, Decimal(0))
                else:
                    total = Decimal(0)
            else:
                total = (min if op == "min" else max)(nums)
            result = jc.Number(str(total))
        return self._reply(ref, original, {"operation": op, "path": path, "field": field,
                                          "source_items": len(values), "result": result}, coverage="all_for_operation")

    def select(self, ref: str, *, path: str = "", field: str, equals_json: str) -> View:
        """Exact field equality on the entire array, returning every matching row."""
        original = self.retrieve(ref)
        values = jc.pointer(jc.loads(original), path)
        if not isinstance(values, list):
            raise ValueError("Select target must be an array")
        wanted = jc.loads(equals_json)
        matches = []
        for idx, row in enumerate(values):
            # Missing fields are explicit errors instead of silent exclusion.
            actual = jc.pointer(row, field)
            if jc.equal(actual, wanted):
                matches.append({"index": idx, "value": row})
        return self._reply(ref, original, {"operation": "select_equal", "path": path,
                                          "field": field, "equals": wanted, "source_items": len(values),
                                          "matches": len(matches), "results": matches}, coverage="all_matches")

    def read(self, ref: str, *, start: int = 0, limit: int = 100, unit: str = "lines",
             path: str = "") -> View:
        if type(start) is not int or type(limit) is not int or start < 0 or limit <= 0:
            raise ValueError("start >= 0 and limit > 0 required")
        original = self.retrieve(ref)
        if unit == "lines":
            if path:
                raise ValueError("path is only supported for items")
            values = original.splitlines(keepends=True)
        elif unit == "items":
            values = jc.pointer(jc.loads(original), path)
            if not isinstance(values, list):
                raise ValueError("Items target must be an array")
        else:
            raise ValueError("unit must be lines or items")
        if start > len(values):
            raise ValueError("start exceeds source length")
        end = min(start + limit, len(values))
        return self._reply(ref, original, {"unit": unit, "path": path, "start": start, "end": end,
                                          "total": len(values), "omitted": len(values)-(end-start),
                                          "next_start": end if end < len(values) else None,
                                          "data": values[start:end]},
                           coverage="all" if start == 0 and end == len(values) else "partial")

    def focus(self, ref: str, *, terms: list[str], budget: int = 800, radius: int = 2) -> View:
        """Experimental low-level excerpt; bypasses Router's no-partial policy.

        Never a claim that selected lines suffice or that retrieval saves tokens.
        """
        self._budget(budget)
        if not terms or any(not isinstance(t, str) or not t for t in terms):
            raise ValueError("Provide nonempty literal search terms")
        if type(radius) is not int or radius < 0:
            raise ValueError("radius must be nonnegative")
        original = self.retrieve(ref)
        lines = original.splitlines(keepends=True)
        needle = [t.casefold() for t in terms]
        hits = [i for i, line in enumerate(lines) if any(t in line.casefold() for t in needle)]
        hit_set = set(hits)
        original_tokens = self.count(original)
        selected = set()
        def render(indices):
            coverage = "all" if len(indices) == len(lines) else "partial"
            text = jc.dumps({"source": ref, "coverage": coverage, "operation": "focus", "terms": terms, "total_lines": len(lines),
                "matching_lines": len(hits), "shown_matching_lines": len(indices & hit_set),
                "omitted_lines": len(lines)-len(indices),
                "next_action": "read(source,start,limit) for omitted evidence; retrieve(source) for exact full text",
                "lines": [{"index": i, "text": lines[i]} for i in sorted(indices)]})
            tokens = self.count(text)
            return View(text, "query-v1", coverage, ref, original_tokens, tokens, tokens <= budget)
        # Select complete neighborhoods only; never cut a line or error trace mid-byte.
        for hit in hits:
            candidate = selected | set(range(max(0, hit-radius), min(len(lines), hit+radius+1)))
            if render(candidate).within_budget:
                selected = candidate
        result = render(selected)
        if not result.within_budget:
            return replace(result, reason="budget_too_small_for_retrieval_metadata")
        return result
