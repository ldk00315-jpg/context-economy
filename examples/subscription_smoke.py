"""Two fresh Codex subscription calls; run one arm at a time to check quota between calls.

No API keys, login changes, or retries. Synthetic six-question batch; not a
statistical quality evaluation. Tool activity invalidates the comparison.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from context_economy import Economy, Store, TiktokenCounter
from context_economy.evaluation import strict_json_grade

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'subscription'


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    counter = TiktokenCounter()
    engine = Economy(Store(OUT / 'originals.sqlite'), counter)
    rows = [{'id': i, 'status': 'active', 'region': 'west', 'value': i * 17}
            for i in range(80)]
    rows[43].update(status='paused', region='east', value=-731)
    paths = [f'/src/module_{i}.py' for i in range(16)]
    code = 'def allowed(active, banned):\n    return active and not banned\n'
    cases = [
        ('table', json.dumps(rows, indent=2), 'json',
         'Return the complete object with id 43.', rows[43]),
        ('logs', 'INFO ready\n' * 90 + 'ERROR timeout\n' + 'INFO ready\n' * 30,
         'text', 'Return {"ready_count":integer,"error_line_1based":integer,"total_lines":integer}.',
         {'ready_count': 120, 'error_line_1based': 91, 'total_lines': 121}),
        ('negation', '通常会員は予約できる。ただし停止中の会員は予約できない。管理者も停止中は例外扱いしない。',
         'text', '停止中の管理者は予約できるか。Return {"allowed":boolean}.', {'allowed': False}),
        ('paths', json.dumps(paths, indent=2), 'json', 'Return ALL paths in original order.', paths),
        ('sum', '[0.1,0.2,0.3,9007199254740993]', 'json',
         'Return the exact decimal sum as a JSON STRING (not a number).', '9007199254740993.6'),
        ('code', code, 'code', 'Return allowed(True, True) as a JSON boolean.', False),
    ]
    expected = {name: answer for name, _, _, _, answer in cases}
    prefix = ('Answer these six independent data questions. Use ONLY the supplied data. '
              'Do not call any tools or read files. Encodings include their decoding instructions. '
              'Return ONE JSON object keyed by question id, with no prose or markdown.\n')
    manifest = {'model': 'gpt-6-astra', 'reasoning_effort': 'medium', 'questions': 6,
                'repetitions': 1, 'max_model_calls': 2, 'expected': expected,
                'quota_baseline_percent': 9, 'quota_before_calls_percent': 10,
                'quota_warning': 'Shared account, integer percentages; no exact 2% guarantee.',
                'quality_claim': 'synthetic_batched_smoke_only', 'cases': []}
    prompts = {'baseline': prefix, 'optimized': prefix}
    for name, raw, kind, question, _ in cases:
        view = engine.pack(raw, kind=kind)
        # Explicit sum request authorizes a full-source exact local operation.
        if name == 'sum':
            view = engine.aggregate(view.original_id, op='sum')
        manifest['cases'].append({'id': name, 'format': view.format,
                                  'original_tokens': counter(raw), 'optimized_tokens': view.tokens,
                                  'coverage': view.coverage})
        for arm, context in [('baseline', raw), ('optimized', view.text)]:
            prompts[arm] += f'\nQUESTION {name}: {question}\nDATA:\n{context}\nEND DATA\n'
    manifest['prompt_tokens_o200k_base'] = {arm: counter(p) for arm, p in prompts.items()}
    for arm, prompt in prompts.items():
        (OUT / f'{arm}.prompt.txt').write_text(prompt, encoding='utf-8')
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False))


def run(arm, executable):
    target = OUT / f'{arm}.jsonl'
    if target.exists():
        raise SystemExit('Existing attempt: refusing automatic rerun or overwrite.')
    prompt = (OUT / f'{arm}.prompt.txt').read_text(encoding='utf-8')
    # Fresh empty cwd, no developer repository or answer files in the prompt.
    work = Path(tempfile.mkdtemp(prefix='context-economy-eval-'))
    cmd = [executable, 'exec', '--ignore-user-config', '--ephemeral', '--json',
           '--skip-git-repo-check', '--sandbox', 'read-only', '-C', str(work),
           '-m', 'gpt-6-astra', '-c', 'model_reasoning_effort="medium"',
           '-c', 'project_doc_max_bytes=0', '-c', 'web_search="disabled"',
           '--disable', 'shell_tool', '--disable', 'multi_agent', '--disable', 'hooks',
           '--disable', 'plugins', '--disable', 'apps', '-']
    env = os.environ.copy()
    # Prevent incidental API-key environment overrides; use existing ChatGPT auth.
    for key in ('CODEX_API_KEY', 'OPENAI_API_KEY'):
        env.pop(key, None)
    (OUT / f'{arm}.invocation.json').write_text(json.dumps(cmd, indent=2), encoding='utf-8')
    with target.open('wb') as stdout, (OUT / f'{arm}.stderr.txt').open('wb') as stderr:
        try:
            result = subprocess.run(cmd, input=prompt.encode('utf-8'), stdout=stdout,
                                    stderr=stderr, env=env, timeout=150)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 'timeout_usage_unknown'
    (OUT / f'{arm}.exit.json').write_text(json.dumps({'returncode': code}), encoding='utf-8')
    summarize()


def summarize():
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    report = {'model_requested': manifest['model'], 'reasoning_effort_requested': 'medium',
              'quality_claim': manifest['quality_claim'], 'arms': {}}
    for arm in ('baseline', 'optimized'):
        path = OUT / f'{arm}.jsonl'
        if not path.exists():
            continue
        events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        usages = [e['usage'] for e in events if e.get('type') == 'turn.completed']
        items = [e['item'] for e in events if e.get('type') == 'item.completed']
        activity = [e.get('item', {}).get('type') for e in events
                    if e.get('type', '').startswith('item.') and
                    e.get('item', {}).get('type') not in ('agent_message', 'reasoning')]
        messages = [x['text'] for x in items if x['type'] == 'agent_message']
        try:
            answer = json.loads(messages[-1])
        except (ValueError, IndexError):
            answer = None
        grades = {key: type(answer) is dict and key in answer and
                  strict_json_grade(json.dumps(answer[key]), json.dumps(value))
                  for key, value in manifest['expected'].items()}
        report['arms'][arm] = {'usage': usages, 'tool_activity': activity,
            'valid': len(usages) == 1 and not activity and
                     json.loads((OUT / f'{arm}.exit.json').read_text())['returncode'] == 0,
            'grades': grades, 'answer': answer,
            'exact_batch_correct': strict_json_grade(json.dumps(answer), json.dumps(manifest['expected']))}
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'baseline', 'optimized', 'summarize'])
    parser.add_argument('--codex', default='codex')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'summarize':
        summarize()
    else:
        run(args.action, args.codex)
