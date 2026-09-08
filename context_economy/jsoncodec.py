"""Strict JSON with original number lexemes, order, and missing/null distinction."""
from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class Number:
    text: str

    def decimal(self) -> Decimal:
        return Decimal(self.text)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON keys are not normalized")
        result[key] = value
    return result


def loads(text: str) -> Any:
    def invalid(value):
        raise ValueError(f"Non-standard JSON number: {value}")
    return json.loads(text, parse_int=Number, parse_float=Number,
                      parse_constant=invalid, object_pairs_hook=_object)


def dumps(value: Any) -> str:
    if isinstance(value, Number):
        return value.text
    if value is None or type(value) in (str, bool, int):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, list):
        return "[" + ",".join(dumps(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(dumps(k) + ":" + dumps(v) for k, v in value.items()) + "}"
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def equal(a: Any, b: Any) -> bool:
    """JSON value equality: numeric exactness, unordered object keys, ordered arrays."""
    if isinstance(a, Number) and isinstance(b, Number):
        return a.decimal() == b.decimal()
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    return a == b


def table_encode(value: Any) -> str | None:
    # Only identical ordered schemas. Never invent missing fields or coerce nulls.
    if not isinstance(value, list) or len(value) < 2 or not all(isinstance(r, dict) for r in value):
        return None
    columns = list(value[0])
    if not columns or any(list(row) != columns for row in value):
        return None
    return dumps({"columns": columns, "rows": [[row[k] for k in columns] for row in value]})


def table_decode(text: str) -> str:
    table = loads(text)
    cols, rows = table["columns"], table["rows"]
    if not all(isinstance(c, str) for c in cols) or len(cols) != len(set(cols)):
        raise ValueError("Invalid columns")
    if any(not isinstance(row, list) or len(row) != len(cols) for row in rows):
        raise ValueError("Invalid row width")
    return dumps([dict(zip(cols, row)) for row in rows])


def pointer(value: Any, path: str) -> Any:
    if path == "":
        return value
    if not path.startswith("/"):
        raise ValueError("JSON pointer must be empty or start with /")
    for escaped in path[1:].split("/"):
        import re
        if re.search(r"~(?![01])", escaped):
            raise ValueError("Invalid JSON pointer escape")
        key = escaped.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not key.isascii() or not key.isdigit() or (len(key) > 1 and key[0] == "0"):
                raise ValueError("Invalid array index")
            value = value[int(key)]
        elif isinstance(value, dict):
            value = value[key]
        else:
            raise ValueError("Pointer traverses a scalar")
    return value
