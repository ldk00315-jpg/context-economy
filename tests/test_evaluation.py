from decimal import Decimal

import pytest

from context_economy.evaluation import CallUsage, ModelRun, Task, Pair, evaluate_pairs, strict_json_grade


@pytest.mark.parametrize('answer', ['[1,2]', '[1,2,3,4]', 'correct answer [1,2,3]', '[true,2,3]'])
def test_no_partial_credit(answer):
    assert not strict_json_grade(answer, '[1,2,3]')


def test_numeric_equivalence_but_no_float_rounding():
    assert strict_json_grade('0.30', '0.3')
    assert not strict_json_grade('9007199254740992', '9007199254740993')
    assert not strict_json_grade('{"a":1,"extra":0}', '{"a":1}')


def test_counts_retrieval_and_retries_not_just_first_call():
    pair = Pair(Task('sum', 'sum?', 'full', {}), Task('sum', 'sum?', 'short', {}), '3')
    def runner(task):
        calls = (CallUsage(100, 10),) if task.context == 'full' else (CallUsage(30, 10), CallUsage(140, 10))
        return ModelRun('3', calls)
    report = evaluate_pairs([pair], runner)
    assert report['optimized_correct'] == 1
    assert report['token_savings_pct'] < 0
    assert report['total_tokens']['optimized'] == 190


def test_failed_calls_are_not_zero_cost_success():
    pair = Pair(Task('x', 'q', 'full', {}), Task('x', 'q', 'short', {}), '3')
    def runner(task):
        if task.context == 'short':
            raise TimeoutError()
        return ModelRun('3', (CallUsage(100, 3),))
    report = evaluate_pairs([pair], runner)
    assert report['regressions'] == 1
    assert not report['usage_complete']
    assert report['token_savings_pct'] is None


def test_usage_validated_and_cost_separate():
    with pytest.raises(ValueError):
        CallUsage(100, 1, 90, 20)
    u = CallUsage(100, 20, 50, 10)
    assert u.cost(input_per_million=Decimal(2), output_per_million=Decimal(10),
                  read_multiplier=Decimal('.1'), write_multiplier=Decimal('1.25')) == Decimal('.000315')
