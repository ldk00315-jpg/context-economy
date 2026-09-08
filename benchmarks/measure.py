"""Deterministic integrity + visible-token benchmark; NOT an LLM quality evaluation."""
import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from context_economy import Economy, Store, TiktokenCounter
from context_economy import jsoncodec as jc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='results')
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260908)
    count = TiktokenCounter('o200k_base')
    rows = [{'event_id': i, 'service_name': 'checkout', 'region_name': 'ap-northeast-1',
             'status': 'FATAL' if i == 167 else 'ok', 'elapsed_ms': rng.randrange(20, 100)} for i in range(300)]
    paths = [f'/project/src/module_{i:04d}.py' for i in range(300)]
    numbers = [rng.randrange(1, 100000)/100 for _ in range(500)]
    cases = [
        ('dictionary_json', json.dumps(rows, indent=2), 'json'),
        ('paths_all_300', json.dumps(paths), 'json'),
        ('numbers_all_500', json.dumps(numbers), 'json'),
        ('repeated_logs', 'healthcheck succeeded\n'*200 + 'FATAL inventory unavailable\ntraceback: checkout.py:47\n' + 'healthcheck succeeded\n'*200, 'text'),
        ('distinct_logs', ''.join(f'2026-09-08 INFO request={i} duration={rng.randrange(10,1000)}ms\n' for i in range(400)), 'text'),
        ('japanese_repeat', '注文を受け付け、在庫を確認し、支払い成功後に発送します。\n'*100, 'text'),
        ('source_code', '\n\n'.join(f'def compute_{i}(x):\n    value = x + {i}\n    return value * 2' for i in range(40)), 'code'),
        ('short_message', 'All 12 tests passed.', 'text'),
    ]
    records = []
    with tempfile.TemporaryDirectory(prefix='context-economy-bench-') as directory:
        engine = Economy(Store(Path(directory)/'originals.sqlite'), count)
        for name, raw, kind in cases:
            view = engine.pack(raw, kind=kind)
            restored = view.restore_semantic()
            integrity = jc.dumps(jc.loads(restored)) == jc.dumps(jc.loads(raw)) if kind=='json' else restored==raw
            assert integrity and engine.retrieve(view.original_id)==raw
            record = {'case':name, 'before':count(raw), 'after':view.tokens,
                      'saved_pct':round((1-view.tokens/count(raw))*100, 2),
                      'format':view.format, 'all_data_restored':integrity,
                      'byte_exact_original_retrieved':True,
                      'simple_minified_json_tokens':count(jc.dumps(jc.loads(raw))) if kind=='json' else None}
            records.append(record)
            (out/(name+'.txt')).write_text(view.text, encoding='utf-8')

        raw = json.dumps(list(range(500)))
        ref = engine.store.put(raw)
        result = engine.aggregate(ref, op='sum')
        assert json.loads(result.text)['result'] == 124750
        query_records = [{'case':'exact_sum_all_500','source_tokens':count(raw),
                          'result_tokens':result.tokens, 'saved_pct':round(100*(1-result.tokens/count(raw)),2),
                          'answer_exact':True, 'evaluated_items':500}]
        raw = json.dumps(rows)
        ref = engine.store.put(raw)
        result = engine.select(ref, field='/event_id', equals_json='267')
        assert json.loads(result.text)['results'][0]['value']==rows[267]
        query_records.append({'case':'exact_find_middle_row','source_tokens':count(raw),
                              'result_tokens':result.tokens, 'saved_pct':round(100*(1-result.tokens/count(raw)),2),
                              'answer_exact':True, 'evaluated_items':300})

        # Explicitly model repeated prompt exposure, including metadata + retrieval request.
        # This is a scripted text replay, not provider usage or model behavior.
        raw = cases[4][1]
        ref = engine.store.put(raw)
        question = 'Return the exact duration for request=237; inspect the original if the excerpt is insufficient.'
        focus = engine.focus(ref, terms=['request=237 '], budget=350, radius=1)
        initial_baseline = json.dumps([{'role':'user','content':question},{'role':'tool','content':raw}], ensure_ascii=False)
        initial_optimized = json.dumps([{'role':'user','content':question},{'role':'tool','content':focus.text}], ensure_ascii=False)
        retrieve_call = json.dumps({'tool':'retrieve','source':ref})
        second = json.dumps([{'role':'user','content':question},{'role':'tool','content':focus.text},
                             {'role':'assistant','content':retrieve_call},{'role':'tool','content':engine.retrieve(ref)}], ensure_ascii=False)
        replay = {
            'kind':'scripted_visible_text_not_model_usage',
            'baseline_one_request_tokens':count(initial_baseline),
            'focus_sufficient_one_request_tokens':count(initial_optimized),
            'focus_then_full_retrieve_total_tokens':count(initial_optimized)+count(retrieve_call)+count(second),
            'retrieval_calls':1,
            'final_answer_tokens':'unmeasured; no LLM called',
            'note':'Retrieval request text is counted as generated output and again in the next input.'}
        replay['focus_saved_pct'] = round(100*(1-replay['focus_sufficient_one_request_tokens']/replay['baseline_one_request_tokens']),2)
        replay['retrieve_saved_pct'] = round(100*(1-replay['focus_then_full_retrieve_total_tokens']/replay['baseline_one_request_tokens']),2)
        assert replay['retrieve_saved_pct'] < 0

    report = {'seed':20260908,'tokenizer':count.name,'scope':'model-visible payload and explicitly scripted replay only',
              'llm_calls':0,'llm_quality':'not_evaluated','lossless_cases':records,'exact_operations':query_records,'replay':replay}
    (out/'benchmark.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines = ['# 初版実測（2026-09-08）','',
        '**実LLMの品質比較ではありません。** 固定seedの全件復号・正確な計算と、tiktoken:o200k_baseによる表示本文の計数。説明ヘッダー・原文参照等も含む。API課金額ではない。', '',
        '| 入力 | 元 | 単純JSON空白除去 | 本実装 | 削減率 | 全件復号 |','|---|---:|---:|---:|---:|---|']
    for r in records:
        lines.append(f"| {r['case']} | {r['before']} | {r['simple_minified_json_tokens'] if r['simple_minified_json_tokens'] is not None else '—'} | {r['after']} | {r['saved_pct']}% | OK |")
    lines += ['', '## 呼び出し側が明示した正確な操作', '',
              '全データをモデルへ渡す代わりに、ローカルで全件処理した結果を渡す。任意の質問へ自動的に適用するものではない。', '',
              '| 操作 | 原文 | 結果 | 削減率 | 全件処理・正確性 |','|---|---:|---:|---:|---|']
    for r in query_records:
        lines.append(f"| {r['case']} | {r['source_tokens']} | {r['result_tokens']} | {r['saved_pct']}% | OK |")
    lines += ['', '## 追加取得で損をするケースも計上', '',
              f"- 原文を1回読む本文トークン：{replay['baseline_one_request_tokens']}",
              f"- 抜粋で足りる場合：{replay['focus_sufficient_one_request_tokens']}（{replay['focus_saved_pct']}%削減）",
              f"- 抜粋後に原文を全取得する場合の累積：{replay['focus_then_full_retrieve_total_tokens']}（{replay['retrieve_saved_pct']}%削減、つまり増加）", '',
              'この再生では質問・ツール結果・再取得要求・次回の履歴再入力を計数する。再取得するかどうかは手順で指定し、モデルが適切に判断できるかは測っていない。ツール定義、最終回答、provider固有のhidden framingは未計数。実LLMの評価にはevaluation.pyの全呼出しusageを使う。', '',
              '数字・パス・コードを保持するため、節約率が0%のケースも正常な結果。全件保持と厳密操作の成功は、モデルの読み取り品質の保証ではない。']
    (out/'BENCHMARK.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
