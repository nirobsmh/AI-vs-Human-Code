#!/usr/bin/env python3
"""Python code transformations for robustness testing.

Used for: train on original code → test on transformed code.
Do not retrain models on these transformed snippets.
"""

from __future__ import annotations

import ast
import io
import tokenize

import black


def black_format_code(code: str):
    try:
        return black.format_str(code, mode=black.Mode())
    except Exception:
        return None


def ast_normalize_code(code: str):
    try:
        tree = ast.parse(code)
        normalized = ast.unparse(tree)
        return normalized + "\n"
    except Exception:
        return None


def remove_comments_only(source: str):
    """Remove comments only; keep docstrings and other string literals."""
    try:
        tokens = list(
            tokenize.generate_tokens(io.StringIO(str(source)).readline)
        )
        filtered = [token for token in tokens if token.type != tokenize.COMMENT]
        result = tokenize.untokenize(filtered)
        ast.parse(result)
        return result
    except Exception:
        return None


def remove_comments_and_docstrings(source: str):
    """Remove comments and most standalone docstrings; keep remaining tokens."""
    try:
        io_obj = io.StringIO(source)
        output = []

        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0

        tokens = tokenize.generate_tokens(io_obj.readline)

        for token in tokens:
            token_type = token.type
            token_string = token.string
            start_line, start_col = token.start
            end_line, end_col = token.end

            if start_line > last_lineno:
                last_col = 0

            if start_col > last_col:
                output.append(" " * (start_col - last_col))

            if token_type == tokenize.COMMENT:
                pass
            elif token_type == tokenize.STRING and prev_toktype in {
                tokenize.INDENT,
                tokenize.NEWLINE,
                tokenize.DEDENT,
            }:
                pass
            else:
                output.append(token_string)

            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line

        result = "".join(output)
        ast.parse(result)
        return result
    except Exception:
        return None


TRANSFORMS = {
    "black": black_format_code,
    "ast": ast_normalize_code,
    "comments_only": remove_comments_only,
    "no_comments": remove_comments_and_docstrings,
}


if __name__ == "__main__":
    sample_code = '''
def add(a, b):
    """Return the sum."""

    # add two values
    result = a + b
    return result
'''
    print("ORIGINAL")
    print(sample_code)
    print("\nBLACK")
    print(black_format_code(sample_code))
    print("\nAST NORMALIZED")
    print(ast_normalize_code(sample_code))
    print("\nNO COMMENTS/DOCSTRINGS")
    print(remove_comments_and_docstrings(sample_code))
