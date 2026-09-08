from dataclasses import replace
import json
import subprocess
import sys

import pytest

from context_economy import Economy, Store, Router, prepare_input, ExplorationCost


@pytest.fixture
def router(tmp_path):
    return Router(Economy(Store(tmp_path / 'originals.sqlite'), len))


def test_ready_input_contains_question_and_measured_full_text(router):
    raw = json.dumps([{'long_name': i, 'other_long_name': i * 2} for i in range(150)], indent=2)
    p = prepare_input(router, raw, 'Find all records', kind='json')
    assert p.input_tokens == len(p.text) < p.original_input_tokens
    assert p.original_input_tokens == len(json.dumps({'question': 'Find all records', 'context': raw}, ensure_ascii=False))
    assert json.loads(p.text)['question'] == 'Find all records'
    assert p.coverage == 'all'


def test_custom_renderer_counts_history_and_escaping_then_falls_back(router):
    raw = json.dumps([{'long_name': i} for i in range(150)], indent=2)
    render = lambda context: 'existing instructions\n' + context + ('x' * 20000 if 'TABLE-V1' in context else '')
    p = prepare_input(router, raw, kind='json', render=render)
    assert p.route == 'raw' and p.text == render(raw)
    assert p.input_tokens == p.original_input_tokens


def test_explicit_scope_and_small_budget_do_not_add_model_step(router):
    raw = json.dumps(list(range(200)))
    p = prepare_input(router, raw, 'Exact sum', kind='json', operation='aggregate', arguments={'op': 'sum'}, budget=1)
    assert p.route == 'aggregate' and p.coverage == 'all_for_operation'
    assert not p.within_budget
    assert json.loads(json.loads(p.text)['context'])['result'] == sum(range(200))


def test_short_code_and_optimistic_exploration_keep_complete_input(router):
    raw = 'target\n' + ''.join(f'line {i}\n' for i in range(100)) + 'exception forbids action\n'
    for args in ({'kind': 'code'}, {'operation': 'explore', 'exploration_cost': ExplorationCost(0, 0)}):
        p = prepare_input(router, raw, 'Can it run?', **args)
        assert json.loads(p.text)['context'] == raw and p.coverage == 'all'
    assert prepare_input(router, '[0.1,0.2]', kind='json', operation='aggregate', arguments={'op': 'sum'}).route == 'raw'


def test_storage_failure_returns_complete_input():
    class Broken:
        def put(self, text):
            raise OSError('unavailable')
    p = prepare_input(Router(Economy(Broken(), len)), 'facts', 'question')
    assert json.loads(p.text)['context'] == 'facts' and p.original_id is None


@pytest.mark.parametrize('counter', [lambda _: None, lambda _: -1, lambda _: True])
def test_invalid_measurement_is_not_reported_as_saving(router, counter):
    with pytest.raises(ValueError):
        prepare_input(router, 'text', count=counter)


def test_partial_output_from_custom_router_is_rejected(router):
    d = router.route('text')
    class Partial:
        economy = router.economy
        def route(self, *args, **kwargs):
            return replace(d, view=replace(d.view, coverage='partial'))
    with pytest.raises(ValueError, match='Partial'):
        prepare_input(Partial(), 'text')


def test_cli_emits_exact_prepared_text_and_no_model_adapter(tmp_path):
    pytest.importorskip('tiktoken')
    path = tmp_path / 'source.json'
    path.write_text(json.dumps([{'long_name': i} for i in range(50)], indent=2), encoding='utf-8')
    cmd = [sys.executable, '-m', 'context_economy', '--db', str(tmp_path / 'cli.sqlite'),
           'prepare', str(path), '--kind', 'json', '--question', 'All records']
    a = subprocess.run(cmd, capture_output=True, check=True)
    b = subprocess.run(cmd + ['--metadata'], capture_output=True, check=True)
    meta = json.loads(b.stdout)
    assert a.stdout == meta['text'].encode('utf-8')
    assert meta['input_tokens'] <= meta['original_input_tokens']
    refused = subprocess.run(cmd + ['--budget', '1'], capture_output=True)
    assert refused.returncode == 2 and refused.stdout == b''
