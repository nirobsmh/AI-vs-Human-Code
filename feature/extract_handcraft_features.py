#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute code-level features for AI vs. Human classification.

Input CSV columns (minimum):
- code
- generator      (optional)
- label          (0/1 or any string label)
- language       (optional)
- problem_id     (optional; if absent we will output empty string)
- id             (optional; if absent we will synthesize a 1-based row id)

Output CSV columns (exact order):
id,generator,label,language,
n_lines,n_chars,avg_line_len,blank_ratio,comment_ratio,has_docstring,avg_leading_spaces,avg_leading_tabs,
n_identifiers,avg_identifier_len,generic_identifier_ratio,snake_case_ratio,camel_case_ratio,token_entropy,unique_identifier_ratio,
ast_depth,n_ifs,n_loops,n_funcs,n_imports,n_try,n_with,n_comprehensions,n_boolops,cyclomatic_complexity_est,
problem_id

Usage:
    python features.py --in path/to/input.csv --out path/to/output.csv
"""

from __future__ import annotations
import ast
import csv
import re
import math
import argparse
from pathlib import Path
from collections import Counter
import codecs

# ----------------------------
# Helpers: safe parsing & text
# ----------------------------

def _safe_open_text(path: str):
    """Open text file with best-guess encoding and normalized newlines."""
    with open(path, "rb") as rb:
        head = rb.read(4)
    if head.startswith(codecs.BOM_UTF16_LE) or head.startswith(codecs.BOM_UTF16_BE):
        enc = "utf-16"
    elif head.startswith(codecs.BOM_UTF8):
        enc = "utf-8-sig"
    else:
        enc = "utf-8"
    return open(path, "r", encoding=enc, newline="")  # newline='' per csv docs

def _strip_nul_iter(f):
    """Generator that yields lines with NULs removed (streaming)."""
    for line in f:
        yield line.replace("\x00", "")

def safe_parse(code: str):
    try:
        return ast.parse(code)
    except Exception:
        return None

def get_docstring_flag(tree: ast.AST | None) -> int:
    if tree is None:
        return 0
    try:
        ds = ast.get_docstring(tree)
        return 1 if isinstance(ds, str) and len(ds) > 0 else 0
    except Exception:
        return 0

def split_lines_keepends(s: str):
    return s.splitlines()

# -----------------------------------
# Basic text metrics (lines/comments)
# -----------------------------------

def basic_text_stats(code: str):
    lines = split_lines_keepends(code)
    n_lines = len(lines) if lines else 1
    n_chars = len(code)
    avg_line_len = (n_chars / n_lines) if n_lines > 0 else 0.0

    blank = sum(1 for ln in lines if ln.strip() == "")
    blank_ratio = blank / n_lines if n_lines else 0.0

    comment_lines = sum(1 for ln in lines if ln.lstrip().startswith("#"))
    comment_ratio = comment_lines / n_lines if n_lines else 0.0

    non_blank_lines = [ln for ln in lines if ln.strip() != ""]
    def leading_counts(ln):
        spaces = 0
        tabs = 0
        for ch in ln:
            if ch == ' ':
                spaces += 1
            elif ch == '\t':
                tabs += 1
            else:
                break
        return spaces, tabs

    if non_blank_lines:
        space_tab = [leading_counts(ln) for ln in non_blank_lines]
        avg_spaces = sum(s for s, _ in space_tab) / len(space_tab)
        avg_tabs   = sum(t for _, t in space_tab) / len(space_tab)
    else:
        avg_spaces = 0.0
        avg_tabs = 0.0

    return {
        "n_lines": n_lines,
        "n_chars": n_chars,
        "avg_line_len": avg_line_len,
        "blank_ratio": blank_ratio,
        "comment_ratio": comment_ratio,
        "avg_leading_spaces": avg_spaces,
        "avg_leading_tabs": avg_tabs,
    }

# --------------------
# Identifier features
# --------------------

GENERIC_NAMES = {
    "i","j","k","x","y","z","a","b","c","d","p","q","r","s","t","u","v","w","n","m",
    "tmp","temp","val","value","res","result","ret","out","ans","data","item","elem",
    "foo","bar","baz",
}

snake_re = re.compile(r'^[a-z]+(_[a-z0-9]+)+$')
camel_re = re.compile(r'^[a-z]+(?:[A-Z][a-z0-9]*)+$')
pascal_re = re.compile(r'^[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)*$')
dunder_re = re.compile(r'^__.*__$')

def collect_identifiers(tree: ast.AST | None) -> list[str]:
    if tree is None:
        return []
    names: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name):
            names.append(node.id); self.generic_visit(node)
        def visit_arg(self, node: ast.arg):
            if node.arg: names.append(node.arg); self.generic_visit(node)
        def visit_FunctionDef(self, node: ast.FunctionDef):
            names.append(node.name); self.generic_visit(node)
        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            names.append(node.name); self.generic_visit(node)
        def visit_ClassDef(self, node: ast.ClassDef):
            names.append(node.name); self.generic_visit(node)
        def visit_Attribute(self, node: ast.Attribute):
            if node.attr: names.append(node.attr); self.generic_visit(node)
        def visit_alias(self, node: ast.alias):
            if node.asname: names.append(node.asname)
            if node.name: names.append(node.name.split('.')[-1])
            self.generic_visit(node)

    Visitor().visit(tree)
    cleaned = [n for n in names if n and n != "_" and not dunder_re.match(n)]
    return cleaned

def identifier_stats(idents: list[str]):
    n_identifiers = len(idents)
    if n_identifiers == 0:
        return {
            "n_identifiers": 0,
            "avg_identifier_len": 0.0,
            "generic_identifier_ratio": 0.0,
            "snake_case_ratio": 0.0,
            "camel_case_ratio": 0.0,
            "unique_identifier_ratio": 0.0,
        }

    avg_len = sum(len(x) for x in idents) / n_identifiers
    generic_ratio = sum(1 for x in idents if x in GENERIC_NAMES) / n_identifiers
    snake_ratio = sum(1 for x in idents if snake_re.match(x)) / n_identifiers
    camel_like = sum(1 for x in idents if camel_re.match(x) or pascal_re.match(x)) / n_identifiers
    unique_ratio = len(set(idents)) / n_identifiers

    return {
        "n_identifiers": n_identifiers,
        "avg_identifier_len": avg_len,
        "generic_identifier_ratio": generic_ratio,
        "snake_case_ratio": snake_ratio,
        "camel_case_ratio": camel_like,
        "unique_identifier_ratio": unique_ratio,
    }

# ----------------
# Token entropy
# ----------------

word_re = re.compile(r'[A-Za-z_][A-Za-z_0-9]*')

def token_entropy(code: str) -> float:
    tokens = word_re.findall(code)
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = sum(counts.values())
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent

# ---------------
# AST depth
# ---------------

def ast_max_depth(node: ast.AST | None) -> int:
    if node is None:
        return 0
    maxd = 1
    for child in ast.iter_child_nodes(node):
        maxd = max(maxd, 1 + ast_max_depth(child))
    return maxd

# ---------------------------
# Structural element counts
# ---------------------------

def structural_counts(tree: ast.AST | None):
    if tree is None:
        return dict(n_ifs=0, n_loops=0, n_funcs=0, n_imports=0,
                    n_try=0, n_with=0, n_comprehensions=0, n_boolops=0)

    counts = {
        "n_ifs": 0, "n_loops": 0, "n_funcs": 0, "n_imports": 0,
        "n_try": 0, "n_with": 0, "n_comprehensions": 0, "n_boolops": 0,
    }

    class Visitor(ast.NodeVisitor):
        def visit_If(self, node: ast.If):
            counts["n_ifs"] += 1; self.generic_visit(node)
        def visit_IfExp(self, node: ast.IfExp):
            counts["n_ifs"] += 1; self.generic_visit(node)
        def visit_For(self, node: ast.For):
            counts["n_loops"] += 1; self.generic_visit(node)
        def visit_AsyncFor(self, node: ast.AsyncFor):
            counts["n_loops"] += 1; self.generic_visit(node)
        def visit_While(self, node: ast.While):
            counts["n_loops"] += 1; self.generic_visit(node)
        def visit_FunctionDef(self, node: ast.FunctionDef):
            counts["n_funcs"] += 1; self.generic_visit(node)
        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            counts["n_funcs"] += 1; self.generic_visit(node)
        def visit_Import(self, node: ast.Import):
            counts["n_imports"] += 1; self.generic_visit(node)
        def visit_ImportFrom(self, node: ast.ImportFrom):
            counts["n_imports"] += 1; self.generic_visit(node)
        def visit_Try(self, node: ast.Try):
            counts["n_try"] += 1; self.generic_visit(node)
        def visit_With(self, node: ast.With):
            counts["n_with"] += 1; self.generic_visit(node)
        def visit_AsyncWith(self, node: ast.AsyncWith):
            counts["n_with"] += 1; self.generic_visit(node)
        def visit_ListComp(self, node: ast.ListComp):
            counts["n_comprehensions"] += 1; self.generic_visit(node)
        def visit_SetComp(self, node: ast.SetComp):
            counts["n_comprehensions"] += 1; self.generic_visit(node)
        def visit_DictComp(self, node: ast.DictComp):
            counts["n_comprehensions"] += 1; self.generic_visit(node)
        def visit_GeneratorExp(self, node: ast.GeneratorExp):
            counts["n_comprehensions"] += 1; self.generic_visit(node)
        def visit_BoolOp(self, node: ast.BoolOp):
            counts["n_boolops"] += max(1, len(getattr(node, "values", [])) - 1); self.generic_visit(node)

    Visitor().visit(tree)
    return counts

# ---------------------------------------
# Cyclomatic Complexity (simple estimate)
# ---------------------------------------

def estimate_cyclomatic(tree: ast.AST | None) -> int:
    if tree is None:
        return 1
    cc = 1
    class Visitor(ast.NodeVisitor):
        def visit_If(self, node: ast.If):
            nonlocal cc; cc += 1; self.generic_visit(node)
        def visit_IfExp(self, node: ast.IfExp):
            nonlocal cc; cc += 1; self.generic_visit(node)
        def visit_For(self, node: ast.For):
            nonlocal cc; cc += 1; self.generic_visit(node)
        def visit_AsyncFor(self, node: ast.AsyncFor):
            nonlocal cc; cc += 1; self.generic_visit(node)
        def visit_While(self, node: ast.While):
            nonlocal cc; cc += 1; self.generic_visit(node)
        def visit_ListComp(self, node: ast.ListComp):
            nonlocal cc; cc += 1; self.generic_visit(node)
        def visit_SetComp(self, node: ast.SetComp):
            nonlocal cc; cc += 1; self.generic_visit(node)
        def visit_DictComp(self, node: ast.DictComp):
            nonlocal cc; cc += 1; self.generic_visit(node)
        def visit_GeneratorExp(self, node: ast.GeneratorExp):
            nonlocal cc; cc += 1; self.generic_visit(node)
        def visit_BoolOp(self, node: ast.BoolOp):
            nonlocal cc; cc += max(1, len(getattr(node, "values", [])) - 1); self.generic_visit(node)
        def visit_Try(self, node: ast.Try):
            nonlocal cc; cc += len(node.handlers); 
            if node.orelse: cc += 1
            if node.finalbody: cc += 1
            self.generic_visit(node)
        def visit_With(self, node: ast.With):
            nonlocal cc; cc += 1; self.generic_visit(node)
        def visit_AsyncWith(self, node: ast.AsyncWith):
            nonlocal cc; cc += 1; self.generic_visit(node)
    Visitor().visit(tree)
    return cc

# ------------------------------
# Output fields (NEW ORDER)
# ------------------------------

PREPEND_FIELDS = ["id", "generator", "label", "language"]

FEATURE_FIELDS = [
    "n_lines","n_chars","avg_line_len","blank_ratio","comment_ratio","has_docstring",
    "avg_leading_spaces","avg_leading_tabs",
    "n_identifiers","avg_identifier_len","generic_identifier_ratio","snake_case_ratio","camel_case_ratio",
    "token_entropy","unique_identifier_ratio",
    "ast_depth","n_ifs","n_loops","n_funcs","n_imports","n_try","n_with","n_comprehensions","n_boolops",
    "cyclomatic_complexity_est",
]

TRAILING_FIELDS = ["problem_id"]  # keep problem_id at the end

OUTPUT_FIELDS = PREPEND_FIELDS + FEATURE_FIELDS + TRAILING_FIELDS

# ------------------------------
# Row -> features transformation
# ------------------------------

def features_for_code(code: str):
    basics = basic_text_stats(code)
    tree = safe_parse(code)
    has_doc = get_docstring_flag(tree)
    idents = collect_identifiers(tree)
    id_stats = identifier_stats(idents)
    tok_entropy = token_entropy(code)
    depth = ast_max_depth(tree)
    struct = structural_counts(tree)
    cc = estimate_cyclomatic(tree)

    feats = {}
    feats.update(basics)
    feats["comment_ratio"] = basics["comment_ratio"]
    feats["has_docstring"] = has_doc
    feats.update(id_stats)
    feats["token_entropy"] = tok_entropy
    feats["ast_depth"] = depth
    feats.update(struct)
    feats["cyclomatic_complexity_est"] = cc
    return feats

# -----------------
# CSV I/O pipeline
# -----------------

def process_csv(in_path: Path, out_path: Path):
    with _safe_open_text(str(in_path)) as fin, \
         out_path.open("w", newline="", encoding="utf-8") as fout:

        # Sniff dialect on a cleaned sample
        sample = ""
        it = _strip_nul_iter(fin)
        for _ in range(200):
            try:
                sample += next(it)
            except StopIteration:
                break
        fin.seek(0)
        cleaned_iter = _strip_nul_iter(fin)

        try:
            dialect = csv.Sniffer().sniff(sample) if sample else csv.excel
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(cleaned_iter, dialect=dialect)
        if not reader.fieldnames:
            raise RuntimeError("Could not read header row. Check quoting/encoding.")
        reader.fieldnames = [fn.lstrip("\ufeff") if isinstance(fn, str) else fn for fn in reader.fieldnames]

        writer = csv.DictWriter(fout, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for idx, row in enumerate(reader, start=1):
            code = row.get("code", "") or ""
            label = row.get("label", "")
            generator = row.get("generator", "")
            language = row.get("language", "")
            problem_id = row.get("problem_id", "")
            rid = row.get("id")  # may be None / empty
            if rid is None or str(rid).strip() == "":
                rid = str(idx)  # synthesize a 1-based id

            feats = features_for_code(code)

            out_row = {
                # prepended meta
                "id": rid,
                "generator": generator,
                "label": label,
                "language": language,
                # features
                "n_lines": feats["n_lines"],
                "n_chars": feats["n_chars"],
                "avg_line_len": round(feats["avg_line_len"], 6),
                "blank_ratio": round(feats["blank_ratio"], 6),
                "comment_ratio": round(feats["comment_ratio"], 6),
                "has_docstring": feats["has_docstring"],
                "avg_leading_spaces": round(feats["avg_leading_spaces"], 6),
                "avg_leading_tabs": round(feats["avg_leading_tabs"], 6),
                "n_identifiers": feats["n_identifiers"],
                "avg_identifier_len": round(feats["avg_identifier_len"], 6),
                "generic_identifier_ratio": round(feats["generic_identifier_ratio"], 6),
                "snake_case_ratio": round(feats["snake_case_ratio"], 6),
                "camel_case_ratio": round(feats["camel_case_ratio"], 6),
                "token_entropy": round(feats["token_entropy"], 6),
                "unique_identifier_ratio": round(feats["unique_identifier_ratio"], 6),
                "ast_depth": feats["ast_depth"],
                "n_ifs": feats["n_ifs"],
                "n_loops": feats["n_loops"],
                "n_funcs": feats["n_funcs"],
                "n_imports": feats["n_imports"],
                "n_try": feats["n_try"],
                "n_with": feats["n_with"],
                "n_comprehensions": feats["n_comprehensions"],
                "n_boolops": feats["n_boolops"],
                "cyclomatic_complexity_est": feats["cyclomatic_complexity_est"],
                # trailing
                "problem_id": problem_id,
            }
            writer.writerow(out_row)

def main():
    ap = argparse.ArgumentParser(description="Extract code features for AI vs Human classification.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input CSV path with a 'code' column")
    ap.add_argument("--out", dest="out_path", required=True, help="Output CSV path for features")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    process_csv(in_path, out_path)
    print(f"[done] Wrote features -> {out_path}")

if __name__ == "__main__":
    main()
