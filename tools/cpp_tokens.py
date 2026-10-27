"""Lexical comparison for a formatting-only C++/CUDA source migration.

Strings, character literals and raw strings remain intact. Comments and layout
are ignored. This supports the import audit; it is not a C++ parser.
"""
import hashlib
import json
import re

LEXEMES = re.compile(
    r'(?P<raw>(?:u8|u|U|L)?R"(?P<delimiter>[^()\\\s]{0,16})\([\s\S]*?\)(?P=delimiter)")'
    r'|(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)'
    r'|(?P<literal>(?:u8|u|U|L)?"(?:\\[\s\S]|[^"\\])*"|(?:u8|u|U|L)?\'(?:\\[\s\S]|[^\'\\])*\')'
    r'|(?P<word>[A-Za-z_]\w*|(?:\d|\.\d)[\w.]*)'
    r'|(?P<operator><<=|>>=|<=>|->\*|\.\.\.|::|->|\+\+|--|&&|\|\||<=|>=|==|!=|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<|>>)'
    r'|(?P<other>[^\s])'
)


def tokens(text):
    text = re.sub(r"\\\r?\n", "", text)
    return [match.group() for match in LEXEMES.finditer(text)
            if match.lastgroup != "comment"]


def token_digest(text):
    return hashlib.sha256(json.dumps(tokens(text), ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()
