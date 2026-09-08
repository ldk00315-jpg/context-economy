"""Adapter-free input preparation. No model calls, tools, or output restrictions."""
from dataclasses import dataclass
import json
from typing import Callable

from .routing import Router


@dataclass(frozen=True)
class PreparedInput:
    text: str
    input_tokens: int
    original_input_tokens: int
    route: str
    coverage: str
    reason: str
    original_id: str | None
    within_budget: bool

    @property
    def saved_tokens(self) -> int:
        return self.original_input_tokens - self.input_tokens


def prepare_input(router: Router, text: str, question: str = '', *,
                  render: Callable[[str], str] | None = None,
                  count: Callable[[str], int] | None = None,
                  budget: int | None = None, **route_args) -> PreparedInput:
    """Return ready-to-send text no larger than the same rendered original.

    Uses the supplied tokenizer (default: router counter). This is a comparison
    of visible input under that tokenizer, not hidden provider usage or output.
    Optional render(context) must be a pure function and includes the caller's
    question, instructions, history, and any other text to compare. No rendering
    or metadata should be added AFTER the measurement if relying on this bound.
    A small budget only sets within_budget=False; evidence is never truncated.
    """
    if budget is not None and (type(budget) is not int or budget <= 0):
        raise ValueError('budget must be a positive integer')
    if not isinstance(text, str) or not isinstance(question, str):
        raise TypeError('text and question must be strings')
    counter = count if count is not None else router.economy.count
    if render is None:
        def render(context):
            return json.dumps({'question': question, 'context': context}, ensure_ascii=False)

    d = router.route(text, **route_args)
    if d.view.coverage not in ('all', 'all_matches', 'all_for_operation', 'all_at_path'):
        raise ValueError('Partial evidence is not allowed in standard preparation')
    original = render(text)
    candidate = render(d.view.text)
    if not isinstance(original, str) or not isinstance(candidate, str):
        raise TypeError('render must return a string')
    before, after = counter(original), counter(candidate)
    if any(type(n) is not int or n < 0 for n in (before, after)):
        raise ValueError('Counter must return nonnegative integer token counts')
    if after >= before:
        chosen, after, route, coverage = original, before, 'raw', 'all'
        reason = d.reason if d.view.text == text else 'rendered_input_has_no_saving'
    else:
        chosen, route, coverage, reason = candidate, d.route, d.view.coverage, d.reason
    return PreparedInput(chosen, after, before, route, coverage, reason,
                         d.view.original_id, budget is None or after <= budget)
