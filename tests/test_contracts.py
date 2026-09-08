import json
import random
import sqlite3
from decimal import Decimal

import pytest

from context_economy import Economy, Store, IntegrityError
from context_economy import jsoncodec as jc


@pytest.fixture
def economy(tmp_path):
    # Unit tests inject deterministic length counting; benchmark uses actual tiktoken.
    return Economy(Store(tmp_path / "originals.sqlite"), len)


def test_table_keeps_all_rows_types_and_numeric_lexemes(economy):
    raw = '[' + ','.join('{"long_identifier":%d,"long_numeric_value":0.12345678901234567890123456789,"status":null,"flag":false}' % i for i in range(100)) + ']'
    view = economy.pack(raw, kind="json")
    assert view.format == "table-v1"
    assert view.restore_semantic() == jc.dumps(jc.loads(raw))
    assert economy.retrieve(view.original_id) == raw


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '[NaN,Infinity]', '{broken', '1e99999999999999999999'])
def test_nonstandard_or_unsupported_normalization_is_safe(economy, raw):
    view = economy.pack(raw, kind="json")
    assert view.restore_semantic() == raw


@pytest.mark.parametrize("raw", ['[{"a":null},{"b":1}]', '[{"a":1},{"a":2,"b":null}]', '[{"a":1,"b":2},{"b":2,"a":1}]'])
def test_missing_null_and_order_never_coerced(economy, raw):
    view = economy.pack(raw, kind="json")
    assert view.format != "table-v1"
    assert jc.dumps(jc.loads(view.restore_semantic())) == jc.dumps(jc.loads(raw))


def test_arrays_not_sampled_even_with_tiny_budget(economy):
    for data in [list(range(500)), [f'/src/file_{i}.py' for i in range(300)]]:
        raw = json.dumps(data)
        view = economy.pack(raw, kind="json", budget=10)
        assert json.loads(view.restore_semantic()) == data
        assert not view.within_budget
        assert view.coverage == "all"
        assert "budget_exceeded" in view.reason


def test_runs_preserve_counts_order_crlf_and_final_newline(economy):
    raw = "ok\r\n"*100 + "FATAL no inventory\n" + "ok\n"*100 + "last"
    view = economy.pack(raw)
    assert view.format == "runs-v1"
    assert view.restore_semantic() == raw


def test_source_code_exact_and_no_inflation(economy):
    code = '# UTF-8 日本語\r\ndef f():\r\n    return "hello  world"\r\n' * 30
    assert economy.pack(code, kind="code").text == code
    assert economy.pack(code, byte_exact=True).text == code
    assert economy.pack("ok").text == "ok"


def test_store_failure_returns_full_input(economy, monkeypatch):
    def fail(_):
        raise OSError("full")
    monkeypatch.setattr(economy.store, "put", fail)
    raw = json.dumps(list(range(500)))
    view = economy.pack(raw, kind="json")
    assert view.text == raw and view.original_id is None
    assert view.reason == "store_failed:OSError"


def test_store_restart_idempotence_and_corruption(economy):
    original = '原文\r\nwith\nnewlines'
    ref = economy.store.put(original)
    assert economy.store.put(original) == ref
    reopened = Store(economy.store.path)
    assert reopened.get(ref) == original
    with sqlite3.connect(economy.store.path) as db:
        db.execute("UPDATE originals SET body=? WHERE id=?", (b'bad', ref))
    with pytest.raises(IntegrityError):
        reopened.get(ref)
    with pytest.raises(IntegrityError):
        reopened.put(original)
    with pytest.raises(KeyError):
        reopened.get('missing')


def test_exact_decimal_aggregate_not_float(economy):
    ref = economy.store.put('[0.1,0.2,9007199254740993,0.000000000000000000000000000000000001]')
    result = jc.loads(economy.aggregate(ref, op="sum").text)
    assert result['result'].text == '9007199254740993.300000000000000000000000000000000001'
    assert result['source_items'].text == '4'


def test_exact_aggregation_uses_every_item(economy):
    values = list(range(500))
    ref = economy.store.put(json.dumps(values))
    assert json.loads(economy.aggregate(ref, op="sum").text)['result'] == 124750
    assert json.loads(economy.aggregate(ref, op="count").text)['result'] == 500
    assert json.loads(economy.aggregate(ref, op="min").text)['result'] == 0
    assert json.loads(economy.aggregate(ref, op="max").text)['result'] == 499


@pytest.mark.parametrize("data", [[1, None], [1, True], [1, "2"]])
def test_aggregate_rejects_skipped_values(economy, data):
    ref = economy.store.put(json.dumps(data))
    with pytest.raises(ValueError):
        economy.aggregate(ref, op="sum")


def test_missing_field_rejected_and_pointer_escaped(economy):
    raw = '{"a/b":[{"v~x":0.1},{"v~x":0.2}]}'
    ref = economy.store.put(raw)
    assert jc.loads(economy.aggregate(ref, path='/a~1b', field='/v~0x', op='sum').text)['result'].text == '0.3'
    with pytest.raises(KeyError):
        economy.aggregate(ref, path='/a~1b', field='/missing', op='sum')
    with pytest.raises(ValueError):
        jc.pointer(jc.loads('[1]'), '/01')


def test_select_finds_middle_row_and_preserves_full_record(economy):
    rows = [{"id": i, "name": "ordinary", "value": i*7} for i in range(500)]
    ref = economy.store.put(json.dumps(rows))
    answer = json.loads(economy.select(ref, field='/id', equals_json='267.0').text)
    assert answer['matches'] == 1
    assert answer['results'] == [{'index': 267, 'value': rows[267]}]
    assert answer['coverage'] == 'all_matches'


def test_select_nested_equality_respects_types_and_numeric_precision(economy):
    ref = economy.store.put('[{"v":{"x":1,"y":false}},{"v":{"x":true,"y":false}}]')
    result = json.loads(economy.select(ref, field='/v', equals_json='{"y":false,"x":1.0}').text)
    assert result['matches'] == 1 and result['results'][0]['index'] == 0


def test_pagination_restores_every_item_without_gaps(economy):
    rows = [f'/src/module_{i}.py' for i in range(307)]
    ref = economy.store.put(json.dumps(rows))
    restored, start = [], 0
    while start is not None:
        result = json.loads(economy.read(ref, start=start, limit=37, unit='items').text)
        restored.extend(result['data'])
        start = result['next_start']
    assert restored == rows
    assert economy.retrieve(ref) == json.dumps(rows)


def test_focus_discloses_missing_evidence_and_retrieves(economy):
    raw = ''.join(f'line {i}\n' for i in range(200)) + 'FATAL inventory lost\ntraceback here\n'
    ref = economy.store.put(raw)
    view = economy.focus(ref, terms=['fatal'], radius=1, budget=1000)
    result = json.loads(view.text)
    assert result['coverage'] == 'partial'
    assert result['matching_lines'] == 1
    assert result['omitted_lines'] > 0
    assert 'FATAL inventory lost' in view.text
    assert economy.retrieve(ref) == raw
    none = json.loads(economy.focus(ref, terms=['absent'], budget=1000).text)
    assert none['matching_lines'] == 0 and none['coverage'] == 'partial'
    tiny = economy.focus(ref, terms=['fatal'], budget=1)
    assert not tiny.within_budget and tiny.reason


def test_deterministic_pack_and_generative_roundtrip(economy):
    rng = random.Random(731)
    pool = [None, True, False, '日本語\n"x"', -1, 99999999999999999, {'nested': [1, None]}]
    for _ in range(50):
        value = [{'long_key': rng.choice(pool), 'other_key': rng.choice(pool)} for _ in range(rng.randrange(1, 40))]
        text = json.dumps(value, ensure_ascii=False, indent=2)
        packed = economy.pack(text, kind='json')
        assert jc.dumps(jc.loads(packed.restore_semantic())) == jc.dumps(jc.loads(text))
        assert packed.tokens <= len(text)
        assert economy.pack(text, kind='json') == packed
