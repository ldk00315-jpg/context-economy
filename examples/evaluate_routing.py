"""Frozen mixed fixtures: public config/code plus explicitly synthetic API/log tasks."""
import argparse
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from context_economy import Economy, Store, TiktokenCounter, Router
from context_economy.evaluation import strict_json_grade
from examples import subscription_smoke as cli

OUT = Path(__file__).resolve().parents[1] / 'results' / 'routing'
PUBLIC = Path('I:/Workspace/headroom-audit-20260908')


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'manifest.json').exists():
        raise SystemExit('Fixtures already frozen')
    e = Economy(Store(OUT / 'originals.sqlite'), TiktokenCounter())
    router = Router(e)
    config = (PUBLIC / 'server.json').read_text(encoding='utf-8')
    code = (PUBLIC / 'headroom/_subprocess.py').read_text(encoding='utf-8')
    names = [node.name for node in ast.parse(code).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    inventory = [{'sku': f'SKU-{i:03}', 'available': i % 7, 'reserved': i % 3,
                  'warehouse': 'east' if i % 2 else 'west'} for i in range(180)]
    inventory[103]['available'] = 0
    ledger = [i * 7 - 23 for i in range(180)]
    log = ''.join(f'event {i:03}: ' + ('WARN timeout' if i in (11, 71, 139) else 'OK heartbeat') + '\n' for i in range(150))
    cases = [
        ('public_config', config, 'json', 'lookup', {'path': '/packages/0/runtimeArguments'},
         'Return the complete value at JSON Pointer /packages/0/runtimeArguments.', json.loads(config)['packages'][0]['runtimeArguments'], 'public:server.json'),
        ('public_code', code, 'code', 'complete', {},
         'Return ALL module-level function names in source order as a JSON array. Exclude class methods.', names, 'public:headroom/_subprocess.py'),
        ('inventory', json.dumps(inventory, indent=2), 'json', 'select', {'field': '/sku', 'equals_json': '"SKU-103"'},
         'Return an array containing every complete inventory record whose sku equals SKU-103, in source order.', [inventory[103]], 'synthetic:inventory'),
        ('ledger', json.dumps(ledger, indent=2), 'json', 'aggregate', {'op': 'sum'},
         'Return the exact sum of all ledger amounts as a JSON number.', sum(ledger), 'synthetic:ledger'),
        ('all_warnings', log, 'text', 'complete', {},
         'Return ALL WARN event numbers in source order as a JSON array.', [11, 71, 139], 'synthetic:logs'),
        ('short_sum', '[0.1,0.2]', 'json', 'aggregate', {'op': 'sum'},
         'Return the exact decimal sum as a JSON string.', '0.3', 'synthetic:short'),
    ]
    prefix = ('Answer independent data questions using ONLY the supplied data. Do not call tools or read files. '
              'JSON operation results are computed over their declared complete scope; encodings include explanations. '
              'Return exactly one JSON object keyed by question id; no prose or markdown.\n')
    prompts = {arm: prefix for arm in ('baseline', 'packed', 'optimized')}
    manifest = {'model': 'gpt-6-astra', 'effort': 'medium', 'expected': {}, 'cases': [],
                'quality_claim': 'six_question_mixed_smoke_no_noninferiority',
                'public_commit': 'e67b3c8a29443a60d6b0018fb22f525c5cd7e709',
                'operation_selection': 'caller_supplied_explicit_contract_not_model_inferred',
                'max_calls': 3, 'quota_start': 11}
    for name, raw, kind, op, args, question, expected, provenance in cases:
        packed = e.pack(raw, kind=kind)
        d = router.route(raw, kind=kind, operation=op, arguments=args)
        assert d.view.tokens <= packed.tokens <= e.count(raw)
        manifest['expected'][name] = expected
        manifest['cases'].append({'id': name, 'provenance': provenance,
            'source_sha256': hashlib.sha256(raw.encode()).hexdigest(), 'route': d.route,
            'reason': d.reason, 'operation': op, 'arguments': args, 'coverage': d.view.coverage,
            'raw_tokens': e.count(raw), 'packed_tokens': packed.tokens, 'routed_tokens': d.view.tokens})
        for arm, context in [('baseline', raw), ('packed', packed.text), ('optimized', d.view.text)]:
            prompts[arm] += f'\nQUESTION {name}: {question}\nDATA:\n{context}\nEND DATA\n'
    manifest['prompt_tokens'] = {a: e.count(p) for a, p in prompts.items()}
    for arm, prompt in prompts.items():
        (OUT / f'{arm}.prompt.txt').write_text(prompt, encoding='utf-8')
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'cases': manifest['cases'], 'prompt_tokens': manifest['prompt_tokens']}, ensure_ascii=False))


def summarize():
    m = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    report = {'quality_claim': m['quality_claim'], 'arms': {}}
    for arm in ('baseline', 'packed', 'optimized'):
        path = OUT / f'{arm}.jsonl'
        if not path.exists():
            continue
        events = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
        usages = [ev['usage'] for ev in events if ev.get('type') == 'turn.completed']
        items = [ev['item'] for ev in events if ev.get('type') == 'item.completed']
        native = [ev for ev in events if ev.get('type', '').startswith('item.') and ev.get('item', {}).get('type') not in ('agent_message', 'reasoning')]
        messages = [it['text'] for it in items if it['type'] == 'agent_message']
        try:
            answer = json.loads(messages[-1])
        except (ValueError, IndexError):
            answer = None
        grades = {name: isinstance(answer, dict) and name in answer and strict_json_grade(json.dumps(answer[name]), json.dumps(value)) for name, value in m['expected'].items()}
        report['arms'][arm] = {'valid': len(usages) == 1 and not native and
            json.loads((OUT / f'{arm}.exit.json').read_text())['returncode'] == 0,
            'usage': usages, 'grades': grades,
            'correct': strict_json_grade(json.dumps(answer), json.dumps(m['expected'])), 'answer': answer}
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'baseline', 'packed', 'optimized', 'summarize'])
    p.add_argument('--codex', default='codex')
    args = p.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'summarize':
        summarize()
    else:
        cli.OUT = OUT
        cli.run(args.action, args.codex)
        summarize()
