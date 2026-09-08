"""Run paired strict answer checks with a caller-supplied provider adapter.

python examples/evaluate_model.py --runner my_adapter:run --output results/model.json
Adapter: run(task: Task) -> ModelRun. It may invoke task.tools and must include
EVERY model call's input/output usage, including retrieval, retries and reasoning.
No provider, API key, reasoning level or paid service is selected by this example.
"""
import argparse
import importlib
import json
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from context_economy import Economy, Store, TiktokenCounter
from context_economy.evaluation import Pair, Task, evaluate_pairs


def _read(engine, ref, **kwargs):
    return engine.read(ref, **kwargs).text


def _aggregate(engine, ref, **kwargs):
    return engine.aggregate(ref, **kwargs).text


def _select(engine, ref, **kwargs):
    return engine.select(ref, **kwargs).text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', required=True, help='Python module:function provider adapter')
    parser.add_argument('--output', default='results/model.json')
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    engine = Economy(Store(output.parent/'model-originals.sqlite'), TiktokenCounter())
    rows = [{'id': i, 'status':'active', 'value':i*17} for i in range(100)]
    cases = [('all_paths', json.dumps([f'/src/m_{i}.py' for i in range(100)]),
              'Return ALL paths as a JSON array in their original order, with no omissions.',
              json.dumps([f'/src/m_{i}.py' for i in range(100)])),
             ('middle_record', json.dumps(rows), 'Return the entire object with id 67 as JSON.', json.dumps(rows[67])),
             ('exact_sum', '[0.1,0.2,0.3,9007199254740993]', 'Return the exact sum as a JSON number.', '9007199254740993.6')]
    pairs=[]
    for name, raw, question, expected in cases:
        packed = engine.pack(raw, kind='json')
        # Bind this task's source so the model does not need an undisclosed hash.
        tools={'retrieve':partial(engine.retrieve, packed.original_id),
               'read':partial(_read, engine, packed.original_id),
               'aggregate':partial(_aggregate, engine, packed.original_id),
               'select':partial(_select, engine, packed.original_id)}
        pairs.append(Pair(Task(name,question,raw,tools),Task(name,question,packed.text,tools),expected))
        if name == 'exact_sum':
            computed = engine.aggregate(packed.original_id, op='sum').text
            pairs.append(Pair(Task('computed_sum',question,raw,tools), Task('computed_sum',question,computed,tools),expected))
    module, function = args.runner.split(':',1)
    runner=getattr(importlib.import_module(module),function)
    report=evaluate_pairs(pairs,runner,repetitions=args.repetitions)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"Saved paired result to {output}; quality claim: {report['quality_claim']}")


if __name__=='__main__':
    main()
