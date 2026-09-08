"""Bounded subscription retrieval evaluation. Each `step` performs ONE model call.

Application-mediated JSON actions use the real Economy store/read/retrieve.
Continuation replays the full transcript in a fresh CLI call; all usage counts.
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

OUT = Path(__file__).resolve().parents[1] / 'results' / 'retrieval'


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def load(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def engine():
    return Economy(Store(OUT / 'originals.sqlite'), TiktokenCounter())


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'manifest.json').exists():
        raise SystemExit('Fixture already exists; refusing overwrite.')
    e = engine()
    alpha = ['対象: Kestrel。通常の公開日は水曜日。例外規定も確認すること。\n']
    alpha += [f'参考メモ {i}: 定例清掃は資料室で実施。\n' for i in range(100)]
    alpha += ['特則: 承認状態が保留の案件は曜日に関係なく公開禁止。\n',
              '最新台帳: 対象の承認状態は保留。\n']
    beta = ['案件 Bracken の適用規則は R-58。担当者と保管日数は規則表を参照。\n']
    beta += [f'補足 {i}: 備品の棚卸し記録。\n' for i in range(100)]
    beta += ['規則表 R-58: 担当者は三浦。保管日数は37日。\n',
             '規則表 R-59: 担当者は加藤。保管日数は14日。\n']
    gamma = [f'event {i:03d}: OK\n' for i in range(120)]
    gamma[9] = 'event 009: WARN transient\n'
    gamma[97] = 'event 097: WARN persistent\n'
    gamma[118] = 'event 118: WARN delayed\n'
    cases = [
        ('permission', ''.join(alpha), ['Kestrel'],
         '水曜日にKestrelを公開できるか。{"allowed":boolean,"approval_status":string}で答える。',
         {'allowed': False, 'approval_status': '保留'}),
        ('reference', ''.join(beta), ['Bracken'],
         'Brackenの担当者と保管日数を{"owner":string,"days":integer}で答える。',
         {'owner': '三浦', 'days': 37}),
        ('all_events', ''.join(gamma), ['WARN'],
         '全ログ中のWARN件数と全WARNのevent番号を{"count":integer,"event_ids":integer[]}で時系列順に答える。',
         {'count': 3, 'event_ids': [9, 97, 118]}),
        ('control', '案件 Delta の確定担当者: 山本。\n', ['Delta'],
         'Deltaの担当者名をJSON文字列で答える。', '山本'),
    ]
    prefix = ('Solve the independent questions using only supplied source evidence. '
              'A source may be complete or partial; judge whether evidence suffices. '
              'Do not guess missing facts. Do not use native tools or read files. '
              'Respond with exactly one JSON object: '
              '{"action":"final","answers":{question_id:answer,...}} OR '
              '{"action":"request","requests":[{"op":"retrieve","source":"sha256 id"},...]}. '
              'You may instead request {"op":"read","source":"sha256 id","start":0,"limit":100}. '
              'read uses zero-based line offsets, returns a page with coverage and next_start; '
              'retrieve returns exact complete source text. Batch needed requests (maximum 4). '
              'The application executes requests and returns results in the transcript. '
              'Only listed source IDs are available. Max 2 request rounds; answer without requests when sufficient.\n')
    prompts = {'baseline': prefix, 'optimized': prefix}
    manifest = {'model': 'gpt-6-astra', 'effort': 'medium', 'quota_start': 11,
                'max_calls': 4, 'max_calls_per_arm': {'baseline': 1, 'optimized': 3},
                'expected': {}, 'sources': [], 'cases': [], 'protocol': 'application_json_actions'}
    for name, raw, terms, question, answer in cases:
        packed = e.pack(raw)
        view = e.focus(packed.original_id, terms=terms, budget=300, radius=0)
        # Keep only the first WARN match to test coverage-aware complete enumeration.
        # Select a budget from source metadata alone, never from a model response.
        if name == 'all_events':
            for budget in range(80, 301):
                candidate = e.focus(packed.original_id, terms=terms, budget=budget, radius=0)
                if len(json.loads(candidate.text)['lines']) == 1:
                    view = candidate
                    break
            assert json.loads(view.text)['shown_matching_lines'] == 1
        manifest['expected'][name] = answer
        manifest['sources'].append(packed.original_id)
        manifest['cases'].append({'id': name, 'source': packed.original_id,
                                  'original_tokens': e.count(raw), 'focus_tokens': view.tokens,
                                  'coverage': view.coverage})
        for arm, context in [('baseline', raw), ('optimized', view.text)]:
            prompts[arm] += f'\nQUESTION {name}: {question}\nSOURCE {packed.original_id}:\n{context}\nEND SOURCE\n'
    for arm, prompt in prompts.items():
        save(f'{arm}.state.json', {'transcript': prompt, 'calls': 0, 'done': False, 'requests': []})
        (OUT / f'{arm}.initial.txt').write_text(prompt, encoding='utf-8')
    manifest['local_prompt_tokens'] = {a: e.count(p) for a, p in prompts.items()}
    save('manifest.json', manifest)
    print(json.dumps({'cases': manifest['cases'], 'prompts': manifest['local_prompt_tokens']}))


def dispatch(response, manifest, e):
    if not isinstance(response, dict) or response.get('action') != 'request':
        raise ValueError('Expected request action')
    requests = response.get('requests')
    if not isinstance(requests, list) or not 1 <= len(requests) <= 4:
        raise ValueError('Expected 1..4 requests')
    results = []
    for req in requests:
        if not isinstance(req, dict) or req.get('source') not in manifest['sources']:
            raise ValueError('Unknown source')
        if req.get('op') == 'retrieve' and set(req) == {'op', 'source'}:
            data = e.retrieve(req['source'])
        elif req.get('op') == 'read' and set(req) == {'op', 'source', 'start', 'limit'}:
            if type(req['start']) is not int or type(req['limit']) is not int or not 1 <= req['limit'] <= 500:
                raise ValueError('Invalid page')
            data = e.read(req['source'], start=req['start'], limit=req['limit'], unit='lines').text
        else:
            raise ValueError('Invalid operation')
        results.append({'request': req, 'result': data})
    return results


def step(arm, executable):
    manifest = load('manifest.json')
    state = load(f'{arm}.state.json')
    if state['done'] or state['calls'] >= manifest['max_calls_per_arm'][arm]:
        raise SystemExit('Arm finished or call cap reached')
    n = state['calls'] + 1
    stem = f'{arm}.{n}'
    path = OUT / f'{stem}.jsonl'
    if path.exists():
        raise SystemExit('Attempt already exists; no automatic retry')
    work = tempfile.mkdtemp(prefix='economy-retrieval-')
    cmd = [executable, 'exec', '--ignore-user-config', '--ephemeral', '--json',
           '--skip-git-repo-check', '--sandbox', 'read-only', '-C', work,
           '-m', manifest['model'], '-c', 'model_reasoning_effort="medium"',
           '-c', 'project_doc_max_bytes=0', '-c', 'web_search="disabled"',
           '--disable', 'shell_tool', '--disable', 'multi_agent', '--disable', 'hooks',
           '--disable', 'plugins', '--disable', 'apps', '-']
    save(f'{stem}.invocation.json', cmd)
    (OUT / f'{stem}.prompt.txt').write_text(state['transcript'], encoding='utf-8')
    env = os.environ.copy()
    for key in ('OPENAI_API_KEY', 'CODEX_API_KEY'):
        env.pop(key, None)
    with path.open('wb') as stdout, (OUT / f'{stem}.stderr.txt').open('wb') as stderr:
        try:
            rc = subprocess.run(cmd, input=state['transcript'].encode('utf-8'), stdout=stdout,
                                stderr=stderr, env=env, timeout=150).returncode
        except subprocess.TimeoutExpired:
            rc = 'timeout_usage_unknown'
    save(f'{stem}.exit.json', {'returncode': rc})
    events = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    usage = [ev['usage'] for ev in events if ev.get('type') == 'turn.completed']
    items = [ev['item'] for ev in events if ev.get('type') == 'item.completed']
    native = [ev for ev in events if ev.get('type', '').startswith('item.') and
              ev.get('item', {}).get('type') not in ('reasoning', 'agent_message')]
    if rc != 0 or len(usage) != 1 or native:
        raise SystemExit('Invalid run or unknown usage; retained raw logs, stop')
    response = json.loads([it['text'] for it in items if it['type'] == 'agent_message'][-1])
    state['calls'] = n
    state.setdefault('usage', []).extend(usage)
    if response.get('action') == 'final' and set(response) == {'action', 'answers'}:
        state['done'] = True
        state['answer'] = response['answers']
        state['correct'] = strict_json_grade(json.dumps(response['answers']), json.dumps(manifest['expected']))
    else:
        results = dispatch(response, manifest, engine())
        state['requests'].append(response['requests'])
        save(f'{stem}.retrieval.json', results)
        state['transcript'] += '\nASSISTANT ACTION:\n' + json.dumps(response, ensure_ascii=False)
        state['transcript'] += '\nAPPLICATION RESULTS:\n' + json.dumps(results, ensure_ascii=False)
        state['transcript'] += '\nContinue using the same response protocol.\n'
    save(f'{arm}.state.json', state)
    print(json.dumps({k: v for k, v in state.items() if k != 'transcript'}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'baseline', 'optimized'])
    p.add_argument('--codex', default='codex')
    args = p.parse_args()
    prepare() if args.action == 'prepare' else step(args.action, args.codex)
