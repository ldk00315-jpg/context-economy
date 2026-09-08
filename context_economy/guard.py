"""Provider-neutral one-attempt gate. No retries, tools, or retrieval loop.

The adapter must make one physical call, disable SDK retries/tools and enforce
the supplied output limit including reasoning. Accounting cannot enforce those
provider-side properties; do not connect an adapter that cannot support them.
"""
from dataclasses import dataclass
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

from .evaluation import ModelRun
from .routing import Router


class BudgetRefused(ValueError):
    pass


class AttemptAlreadyUsed(RuntimeError):
    pass


@dataclass(frozen=True)
class SingleCallPlan:
    prompt: str
    input_upper_bound: int
    max_output_tokens: int
    baseline_upper_bound: int
    route: str

    @property
    def total_upper_bound(self):
        return self.input_upper_bound + self.max_output_tokens


def plan_call(router: Router, text: str, question: str, *, count_input_upper_bound,
              max_output_tokens: int, total_limit: int, **route_args) -> SingleCallPlan:
    """Count FINAL rendered requests, including provider framing via caller counter.

    count_input_upper_bound must include all hidden instructions/tool/schema tokens
    and use the provider tokenizer or a proven upper bound. Unknown means refuse.
    This bounds against the baseline's SAME output ceiling, not its actual output.
    """
    if any(type(x) is not int or x <= 0 for x in (max_output_tokens, total_limit)):
        raise ValueError('Positive integer output and total limits required')
    d = router.route(text, **route_args)
    if d.view.coverage not in ('all', 'all_matches', 'all_for_operation', 'all_at_path'):
        raise BudgetRefused('Partial evidence is not allowed')

    def render(context):
        return ('Answer using only the supplied evidence. No tools or extra retrieval are available. '
                'If evidence is insufficient, state that; do not invent missing facts.\n'
                + json.dumps({'question': question, 'evidence': context}, ensure_ascii=False))

    original = render(text)
    candidate = render(d.view.text)
    base_n = count_input_upper_bound(original)
    candidate_n = count_input_upper_bound(candidate)
    if any(type(x) is not int or x < 0 for x in (base_n, candidate_n)):
        raise BudgetRefused('Unknown or invalid final input upper bound')
    # Escaping/framing can erase a payload saving. Compare complete wire prompt.
    if candidate_n > base_n:
        candidate, candidate_n, route = original, base_n, 'raw'
    else:
        route = d.route
    if candidate_n + max_output_tokens > total_limit:
        raise BudgetRefused('Call cannot fit; refuse before sending, never truncate evidence')
    return SingleCallPlan(candidate, candidate_n, max_output_tokens, base_n + max_output_tokens, route)


class SingleCallGate:
    """Durable, fail-closed at-most-once admission for a logical request ID.

    Pending claims survive crashes. No automatic retry even on an adapter error or
    missing usage; the previous call may already have consumed budget. Separate
    IDs are separate user-authorized jobs, not a mechanism to retry automatically.
    """
    def __init__(self, path):
        self.path = str(Path(path))
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, status TEXT NOT NULL)')

    def run(self, request_id: str, plan: SingleCallPlan, send) -> ModelRun:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError('Stable nonempty request ID required')
        if (type(plan.input_upper_bound) is not int or plan.input_upper_bound < 0 or
            type(plan.max_output_tokens) is not int or plan.max_output_tokens <= 0 or
            type(plan.baseline_upper_bound) is not int or plan.total_upper_bound > plan.baseline_upper_bound):
            raise BudgetRefused('Invalid plan bounds')
        fingerprint = hashlib.sha256(json.dumps([plan.prompt, plan.input_upper_bound,
            plan.max_output_tokens, plan.baseline_upper_bound]).encode()).hexdigest()
        # Commit before any model side effect. SQLite serializes competing claims.
        try:
            with closing(sqlite3.connect(self.path)) as db, db:
                db.execute('INSERT INTO attempts VALUES (?, ?, ?)', (request_id, fingerprint, 'pending'))
        except sqlite3.IntegrityError as exc:
            raise AttemptAlreadyUsed('Attempt already admitted; retry is blocked') from exc
        status = 'failed_or_unknown'
        try:
            result = send(plan)  # exactly one invocation; no fallback branch
            if not isinstance(result, ModelRun) or len(result.calls) != 1:
                raise BudgetRefused('Adapter must report exactly one physical call; usage unknown/invalid')
            u = result.calls[0]
            if u.input_tokens > plan.input_upper_bound or u.output_tokens > plan.max_output_tokens:
                raise BudgetRefused('Adapter exceeded declared bounds; recorded failure, no retry')
            status = 'completed' if result.completed and not result.error else 'failed_or_unknown'
            return result
        finally:
            with closing(sqlite3.connect(self.path)) as db, db:
                db.execute('UPDATE attempts SET status=? WHERE id=?', (status, request_id))

    def status(self, request_id):
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute('SELECT status FROM attempts WHERE id=?', (request_id,)).fetchone()
        return row[0] if row else None
