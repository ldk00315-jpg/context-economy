from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
import sqlite3

import pytest

from context_economy import (Economy, Store, Router, plan_call, SingleCallGate,
                             BudgetRefused, AttemptAlreadyUsed, ExplorationCost)
from context_economy.evaluation import ModelRun, CallUsage


@pytest.fixture
def router(tmp_path):
    return Router(Economy(Store(tmp_path / 'originals.sqlite'), len))


def plan(router, **kwargs):
    return plan_call(router, 'original evidence', 'Summarize', count_input_upper_bound=len,
                     max_output_tokens=100, total_limit=1000, **kwargs)


def test_unknown_input_and_insufficient_budget_fail_before_send(router):
    for counter in (lambda _: None, lambda _: -1, lambda _: True):
        with pytest.raises(BudgetRefused):
            plan_call(router, 'x', 'q', count_input_upper_bound=counter,
                      max_output_tokens=100, total_limit=1000)
    with pytest.raises(BudgetRefused):
        plan_call(router, 'x' * 500, 'q', count_input_upper_bound=len,
                  max_output_tokens=100, total_limit=10)


def test_final_rendering_and_output_reservation_are_counted(router):
    p = plan(router)
    assert p.input_upper_bound == len(p.prompt)
    assert p.total_upper_bound == len(p.prompt) + 100
    assert p.total_upper_bound <= p.baseline_upper_bound


def test_provider_rendering_can_reject_payload_saving(router):
    rows = [{'long_field_name': i} for i in range(200)]
    raw = json.dumps(rows, indent=2)
    assert router.route(raw, kind='json').route == 'complete_pack'
    p = plan_call(router, raw, 'All rows', kind='json',
        count_input_upper_bound=lambda s: len(s) + (100000 if 'TABLE-V1' in s else 0),
        max_output_tokens=100, total_limit=1000000)
    assert p.route == 'raw'
    assert json.loads(p.prompt.split('\n', 1)[1])['evidence'] == raw


def test_previously_profitable_estimate_cannot_open_retrieval_loop(router):
    raw = 'target reference R-9\n' + ''.join(f'line {i}: a different record\n' for i in range(200)) + 'R-9: publication forbidden\n'
    p = plan_call(router, raw, 'May target publish?', operation='explore',
        arguments={'terms': ['target'], 'budget': 500, 'radius': 0},
        exploration_cost=ExplorationCost(0, 0), count_input_upper_bound=len,
        max_output_tokens=100, total_limit=100000)
    assert 'publication forbidden' in p.prompt
    assert p.total_upper_bound <= p.baseline_upper_bound


def test_exact_scope_still_saves_without_second_call(router, tmp_path):
    p = plan_call(router, json.dumps(list(range(200))), 'Sum', kind='json',
        operation='aggregate', arguments={'op': 'sum'}, count_input_upper_bound=len,
        max_output_tokens=100, total_limit=100000)
    assert p.route == 'aggregate' and p.total_upper_bound < p.baseline_upper_bound
    sent = []
    def send(p):
        sent.append(p)
        return ModelRun('19900', (CallUsage(p.input_upper_bound, 5),))
    g = SingleCallGate(tmp_path / 'attempts.sqlite')
    assert g.run('sum-order-1', p, send).output == '19900'
    with pytest.raises(AttemptAlreadyUsed):
        SingleCallGate(g.path).run('sum-order-1', p, send)
    assert len(sent) == 1 and g.status('sum-order-1') == 'completed'


@pytest.mark.parametrize('outcome', ['exception', 'unknown', 'two_calls', 'over_limit', 'incomplete'])
def test_failure_never_triggers_retry_even_after_restart(router, tmp_path, outcome):
    p = plan(router)
    sent = []
    def send(p):
        sent.append(p)
        if outcome == 'exception':
            raise TimeoutError('may already have spent tokens')
        if outcome == 'unknown':
            return ModelRun('?', ())
        if outcome == 'two_calls':
            return ModelRun('?', (CallUsage(1, 1), CallUsage(1, 1)))
        if outcome == 'over_limit':
            return ModelRun('?', (CallUsage(p.input_upper_bound, 101),))
        return ModelRun('insufficient evidence', (CallUsage(1, 1),), completed=False)
    g = SingleCallGate(tmp_path / 'attempts.sqlite')
    if outcome == 'incomplete':
        assert not g.run('job', p, send).completed
    else:
        with pytest.raises((TimeoutError, BudgetRefused)):
            g.run('job', p, send)
    assert g.status('job') == 'failed_or_unknown'
    with pytest.raises(AttemptAlreadyUsed):
        SingleCallGate(g.path).run('job', p, send)
    assert len(sent) == 1


def test_pending_attempt_survives_crash_and_blocks_reuse(router, tmp_path):
    g = SingleCallGate(tmp_path / 'attempts.sqlite')
    with closing(sqlite3.connect(g.path)) as db, db:
        db.execute('INSERT INTO attempts VALUES (?, ?, ?)', ('crashed', 'hash', 'pending'))
    with pytest.raises(AttemptAlreadyUsed):
        SingleCallGate(g.path).run('crashed', plan(router), lambda _: pytest.fail('must not send'))


def test_concurrent_same_id_admits_only_one_call(router, tmp_path):
    p = plan(router)
    g = SingleCallGate(tmp_path / 'attempts.sqlite')
    sent = []
    def worker(_):
        try:
            return g.run('shared', p, lambda p: (sent.append(p) or ModelRun('ok', (CallUsage(1, 1),))))
        except AttemptAlreadyUsed:
            return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(worker, range(4)))
    assert len(sent) == 1 and sum(x is not None for x in outcomes) == 1


def test_ledger_failure_prevents_model_side_effect(router, tmp_path):
    g = SingleCallGate(tmp_path / 'attempts.sqlite')
    g.path = str(tmp_path / 'missing' / 'cannot-open.sqlite')
    with pytest.raises(sqlite3.OperationalError):
        g.run('job', plan(router), lambda _: pytest.fail('must not send'))
