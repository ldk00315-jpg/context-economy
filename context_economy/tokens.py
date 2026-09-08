"""No chars/4 fallback: a caller must choose a genuine tokenizer."""
from __future__ import annotations


class TiktokenCounter:
    def __init__(self, encoding: str = "o200k_base"):
        import tiktoken
        self.encoding = tiktoken.get_encoding(encoding)
        self.name = f"tiktoken:{encoding}"

    def __call__(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))
