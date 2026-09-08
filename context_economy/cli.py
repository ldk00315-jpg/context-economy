from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import Economy, Store, TiktokenCounter, Router, prepare_input


def main():
    parser = argparse.ArgumentParser(description="Lossless-first context preparation and exact local queries")
    parser.add_argument("--db", default=".context-economy/originals.sqlite")
    parser.add_argument("--encoding", default="o200k_base")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser('prepare', help='Adapter-free preparation; stdout is ready-to-send text')
    prepare.add_argument('file')
    prepare.add_argument('--question', default='')
    prepare.add_argument('--kind', choices=['json', 'text', 'code'], default='text')
    prepare.add_argument('--operation', choices=['complete', 'aggregate', 'select', 'lookup', 'explore'], default='complete')
    prepare.add_argument('--arguments', default='{}', help='Explicit operation arguments as a JSON object')
    prepare.add_argument('--budget', type=int)
    prepare.add_argument('--metadata', action='store_true', help='Emit application envelope; only its text field is measured')
    pack = sub.add_parser("pack")
    pack.add_argument("file")
    pack.add_argument("--kind", choices=["json", "text", "code"], default="text")
    pack.add_argument("--budget", type=int)
    pack.add_argument("--byte-exact", action="store_true")
    retrieve = sub.add_parser("retrieve")
    retrieve.add_argument("source")
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("source")
    aggregate.add_argument("op", choices=["sum", "count", "min", "max"])
    aggregate.add_argument("--path", default="")
    aggregate.add_argument("--field")
    read = sub.add_parser("read")
    read.add_argument("source")
    read.add_argument("--start", type=int, default=0)
    read.add_argument("--limit", type=int, default=100)
    read.add_argument("--unit", choices=["lines", "items"], default="lines")
    read.add_argument("--path", default="")
    focus = sub.add_parser("focus")
    focus.add_argument("source")
    focus.add_argument("--term", action="append", required=True)
    focus.add_argument("--budget", type=int, default=800)
    focus.add_argument("--radius", type=int, default=2)
    args = parser.parse_args()
    try:
        store = Store(args.db)
        if args.command == "retrieve":
            # Do not translate newlines; this command is the exact original channel.
            sys.stdout.buffer.write(store.get(args.source).encode("utf-8"))
            return 0
        engine = Economy(store, TiktokenCounter(args.encoding))
        if args.command == 'prepare':
            arguments = json.loads(args.arguments)
            if not isinstance(arguments, dict):
                raise ValueError('--arguments must be a JSON object')
            with Path(args.file).open(encoding='utf-8', newline='') as f:
                prepared = prepare_input(Router(engine), f.read(), args.question,
                    kind=args.kind, operation=args.operation, arguments=arguments, budget=args.budget)
            if args.metadata:
                output = json.dumps(asdict(prepared), ensure_ascii=False)
            else:
                if not prepared.within_budget:
                    print('Input exceeds --budget; no output emitted and no evidence truncated.', file=sys.stderr)
                    return 2
                output = prepared.text
            # No uncounted newline added to the ready-to-send text.
            sys.stdout.buffer.write(output.encode('utf-8'))
            return 0
        if args.command == "pack":
            with Path(args.file).open(encoding="utf-8", newline="") as f:
                view = engine.pack(f.read(), kind=args.kind, budget=args.budget, byte_exact=args.byte_exact)
        elif args.command == "aggregate":
            view = engine.aggregate(args.source, op=args.op, path=args.path, field=args.field)
        elif args.command == "read":
            view = engine.read(args.source, start=args.start, limit=args.limit, unit=args.unit, path=args.path)
        else:
            view = engine.focus(args.source, terms=args.term, budget=args.budget, radius=args.radius)
        # The envelope is for the application. Send view.text to the model; counts refer to that text.
        sys.stdout.buffer.write((json.dumps(asdict(view), ensure_ascii=False) + "\n").encode("utf-8"))
        return 0
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "detail": str(exc)}, ensure_ascii=True), file=sys.stderr)
        return 2
