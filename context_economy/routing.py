"""Explicit query contracts and measured context routing, without guessing intent."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from . import jsoncodec as jc
from .engine import Economy, View


@dataclass(frozen=True)
class ExplorationCost:
    """Legacy estimation parameters, retained for source compatibility only.

    Router never authorizes partial exploration from these parameters.

    continuation_overhead includes repeated harness, request, and framing tokens.
    Initial common prompt cost cancels out of the route comparison.
    """
    full_retrieval_probability: float = 1.0
    continuation_overhead: int = 0
    minimum_saving: int = 1

    def __post_init__(self):
        p = self.full_retrieval_probability
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('Retrieval probability must be finite and between 0 and 1')
        if any(type(x) is not int or x < 0 for x in (self.continuation_overhead, self.minimum_saving)):
            raise ValueError('Cost bounds must be nonnegative integers')


@dataclass(frozen=True)
class Decision:
    view: View
    route: str
    reason: str
    complete_tokens: int
    estimated_context_tokens: float
    candidates: tuple[tuple[str, float], ...]


class Router:
    """Default keeps complete evidence; exact reductions require an explicit operation.

    Caller maps task semantics to operations. This is not an NL intent classifier.
    Model/tool costs of performing that mapping must be counted by the caller.
    """

    def __init__(self, economy: Economy):
        self.economy = economy

    def route(self, text: str, *, kind: str = 'text', operation: str = 'complete',
              arguments: dict | None = None, budget: int | None = None,
              exploration_cost: ExplorationCost | None = None) -> Decision:
        e = self.economy
        args = dict(arguments or {})
        if operation not in ('complete', 'aggregate', 'select', 'lookup', 'explore'):
            raise ValueError('Unknown operation')
        if operation == 'complete' and args:
            raise ValueError('Complete operation takes no arguments')
        if operation in ('aggregate', 'select', 'lookup') and kind != 'json':
            raise ValueError('Exact JSON operations require kind=json')
        complete = e.pack(text, kind=kind, budget=budget)
        candidates = [('complete', float(complete.tokens))]

        def keep(reason):
            route = 'raw' if complete.format == 'raw' else 'complete_pack'
            return Decision(complete, route, reason, complete.tokens, float(complete.tokens), tuple(candidates))

        if operation == 'complete':
            return keep('complete_evidence_required')
        if complete.original_id is None:
            return keep('original_store_unavailable')
        ref = complete.original_id
        if operation == 'explore':
            # Full recovery costs at least the complete source plus an extra call.
            # No probability estimate can bound the cost of that individual case.
            return keep('partial_exploration_blocked_unbounded_recovery')
        try:
            if operation == 'aggregate':
                view = e.aggregate(ref, **args)
            elif operation == 'select':
                view = e.select(ref, **args)
            else:
                if set(args) != {'path'}:
                    raise TypeError('lookup requires exactly path')
                value = jc.pointer(jc.loads(e.retrieve(ref)), args['path'])
                view = e._reply(ref, text, {'operation': 'lookup', 'path': args['path'], 'value': value},
                                coverage='all_at_path', budget=budget)
        except (ValueError, KeyError, IndexError) as exc:
            # No synthetic result on missing evidence or unsupported numeric range.
            return keep(f'exact_operation_unavailable:{type(exc).__name__}')
        candidates.append((operation, float(view.tokens)))
        if view.tokens >= complete.tokens:
            return keep('exact_result_overhead_erases_saving')
        if budget is not None:
            view = replace(view, within_budget=view.tokens <= budget)
        return Decision(view, operation, 'explicit_query_scope_complete', complete.tokens,
                        float(view.tokens), tuple(candidates))
