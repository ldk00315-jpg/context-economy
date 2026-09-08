import json

import pytest

from context_economy import Economy, Store, Router, ExplorationCost


@pytest.fixture
def router(tmp_path):
    return Router(Economy(Store(tmp_path / 'data.sqlite'), len))


def test_complete_enumeration_never_becomes_excerpt(router):
    data = [f'/module/{i}' for i in range(180)]
    d = router.route(json.dumps(data), kind='json', budget=5)
    assert json.loads(d.view.restore_semantic()) == data
    assert d.view.coverage == 'all' and not d.view.within_budget


def test_exact_sum_short_falls_back_long_saves(router):
    short = router.route('[0.1,0.2]', kind='json', operation='aggregate', arguments={'op': 'sum'})
    assert short.route == 'raw'
    long = router.route(json.dumps(list(range(400))), kind='json', operation='aggregate', arguments={'op': 'sum'})
    assert long.route == 'aggregate'
    assert json.loads(long.view.text)['result'] == sum(range(400))


def test_select_preserves_late_duplicates(router):
    rows = [{'id': i % 60, 'value': i} for i in range(180)]
    d = router.route(json.dumps(rows), kind='json', operation='select',
                     arguments={'field': '/id', 'equals_json': '17'})
    assert d.route == 'select'
    assert [r['value'] for r in json.loads(d.view.text)['results']] == [rows[i] for i in (17, 77, 137)]


def test_lookup_escaped_pointer_and_missing_evidence(router):
    raw = json.dumps({'a/b': {'~key': None}, 'filler': list(range(200))})
    d = router.route(raw, kind='json', operation='lookup', arguments={'path': '/a~1b/~0key'})
    assert d.route == 'lookup' and json.loads(d.view.text)['value'] is None
    missing = router.route(raw, kind='json', operation='lookup', arguments={'path': '/missing'})
    assert missing.view.coverage == 'all'
    assert missing.reason == 'exact_operation_unavailable:KeyError'


def test_exploration_cannot_be_authorized_by_optimistic_cost(router):
    text = 'target needs an exception check\n' + ''.join(f'line {i}: unique reference\n' for i in range(200))
    args = {'terms': ['target'], 'budget': 600, 'radius': 0}
    default = router.route(text, operation='explore', arguments=args)
    assert default.view.coverage == 'all'
    d = router.route(text, operation='explore', arguments=args,
                     exploration_cost=ExplorationCost(0.1, 1000))
    assert d.view.coverage == 'all'
    assert d.reason == 'partial_exploration_blocked_unbounded_recovery'
    too_small = router.route(text, operation='explore', arguments=args, budget=1,
                             exploration_cost=ExplorationCost(0.1, 1000))
    assert too_small.view.coverage == 'all' and not too_small.view.within_budget
    zero_risk_claim = router.route(text, operation='explore', arguments=args,
                                   exploration_cost=ExplorationCost(0, 0))
    assert zero_risk_claim.view.restore_semantic() == text
    expensive = router.route(text, operation='explore', arguments=args,
                             exploration_cost=ExplorationCost(0.1, 100000))
    assert expensive.view.coverage == 'all'


def test_no_loss_when_store_unavailable(tmp_path):
    class Broken:
        def put(self, text):
            raise OSError('offline')
    d = Router(Economy(Broken(), len)).route('[1,2]', kind='json', operation='aggregate', arguments={'op': 'sum'})
    assert d.view.text == '[1,2]' and d.reason == 'original_store_unavailable'


@pytest.mark.parametrize('cost', [float('nan'), -0.1, 1.1, True])
def test_cost_rejects_invalid_probabilities(cost):
    with pytest.raises(ValueError):
        ExplorationCost(cost)


def test_invalid_contract_is_explicit(router):
    with pytest.raises(ValueError):
        router.route('x', operation='magic')
    code = router.route('print(1)', kind='code', operation='explore', arguments={'terms': ['print']})
    assert code.view.text == 'print(1)' and code.view.coverage == 'all'
