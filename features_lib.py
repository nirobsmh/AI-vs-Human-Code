# features_lib.py
import ast, math, re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

PYTHON_KEYWORDS = {
    "False","None","True","and","as","assert","async","await","break","class","continue",
    "def","del","elif","else","except","finally","for","from","global","if","import",
    "in","is","lambda","nonlocal","not","or","pass","raise","return","try","while",
    "with","yield"
}
GENERIC_NAME_PATTERNS = [
    re.compile(r"^[a-zA-Z]$"),
    re.compile(r"^[a-zA-Z]\d+$"),
    re.compile(r"^(tmp|temp|var|foo|bar)$")
]
SNAKE_CASE_RE = re.compile(r"^[a-z]+(_[a-z0-9]+)*$")
CAMEL_CASE_RE = re.compile(r"^[a-z]+(?:[A-Z][a-z0-9]+)+$")

def safe_div(n, d): return float(n)/d if d else 0.0
def is_generic_identifier(name: str) -> bool:
    n = name.lower()
    return any(pat.match(n) for pat in GENERIC_NAME_PATTERNS)

def shannon_entropy_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0: return 0.0
    return -sum((c/total)*math.log2(c/total) for c in counts.values())

def extract_lexical_features(code: str) -> Dict[str, Any]:
    lines = code.splitlines()
    n_lines, n_chars = len(lines), len(code)
    blank_lines = sum(1 for l in lines if not l.strip())
    comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
    has_docstring = 1 if ('"""' in code or "'''" in code) else 0
    leading_spaces = [len(l) - len(l.lstrip(" ")) for l in lines if l.strip()]
    leading_tabs   = [len(l) - len(l.lstrip("\t")) for l in lines if l.strip()]
    return {
        "n_lines": n_lines,
        "n_chars": n_chars,
        "avg_line_len": safe_div(n_chars, n_lines),
        "blank_ratio": safe_div(blank_lines, n_lines),
        "comment_ratio": safe_div(comment_lines, n_lines),
        "has_docstring": has_docstring,
        "avg_leading_spaces": float(np.mean(leading_spaces) if leading_spaces else 0.0),
        "avg_leading_tabs": float(np.mean(leading_tabs) if leading_tabs else 0.0),
    }

def extract_identifier_and_style_features(code: str) -> Dict[str, Any]:
    names = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", code)
    identifiers = [n for n in names if n not in PYTHON_KEYWORDS]
    n_identifiers = len(identifiers)
    snake = sum(1 for i in identifiers if SNAKE_CASE_RE.match(i))
    camel = sum(1 for i in identifiers if CAMEL_CASE_RE.match(i))
    token_entropy = shannon_entropy_from_counts(Counter(names))
    generic = sum(1 for i in identifiers if is_generic_identifier(i))
    return {
        "n_identifiers": n_identifiers,
        "avg_identifier_len": safe_div(sum(len(i) for i in identifiers), n_identifiers),
        "generic_identifier_ratio": safe_div(generic, n_identifiers),
        "snake_case_ratio": safe_div(snake, n_identifiers),
        "camel_case_ratio": safe_div(camel, n_identifiers),
        "token_entropy": token_entropy,
        "unique_identifier_ratio": safe_div(len(set(identifiers)), n_identifiers),
    }

@dataclass
class ASTCounts:
    max_depth: int = 0
    n_ifs: int = 0
    n_loops: int = 0
    n_funcs: int = 0
    n_imports: int = 0
    n_try: int = 0
    n_with: int = 0
    n_comprehensions: int = 0
    n_boolops: int = 0

class _DepthVisitor(ast.NodeVisitor):
    def __init__(self): self.counts = ASTCounts()
    def generic_visit(self, node, depth=0):
        if depth > self.counts.max_depth: self.counts.max_depth = depth
        if isinstance(node, (ast.For, ast.While)): self.counts.n_loops += 1
        if isinstance(node, ast.If): self.counts.n_ifs += 1
        if isinstance(node, ast.FunctionDef): self.counts.n_funcs += 1
        if isinstance(node, (ast.Import, ast.ImportFrom)): self.counts.n_imports += 1
        if isinstance(node, ast.Try): self.counts.n_try += 1
        if isinstance(node, ast.With): self.counts.n_with += 1
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            self.counts.n_comprehensions += 1
        if isinstance(node, ast.BoolOp): self.counts.n_boolops += 1
        for child in ast.iter_child_nodes(node):
            self.generic_visit(child, depth + 1)

def extract_ast_structural_features(code: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {k:0 for k in [
            "ast_depth","n_ifs","n_loops","n_funcs","n_imports",
            "n_try","n_with","n_comprehensions","n_boolops"
        ]}
    v = _DepthVisitor(); v.generic_visit(tree, 1)
    c = v.counts
    return {
        "ast_depth": c.max_depth,
        "n_ifs": c.n_ifs,
        "n_loops": c.n_loops,
        "n_funcs": c.n_funcs,
        "n_imports": c.n_imports,
        "n_try": c.n_try,
        "n_with": c.n_with,
        "n_comprehensions": c.n_comprehensions,
        "n_boolops": c.n_boolops,
    }

def estimate_cyclomatic_complexity(ast_feats: Dict[str, Any]) -> int:
    return int(1
        + ast_feats.get("n_ifs",0)
        + ast_feats.get("n_loops",0)
        + ast_feats.get("n_try",0)
        + ast_feats.get("n_with",0)
        + ast_feats.get("n_comprehensions",0)
        + ast_feats.get("n_boolops",0)
    )

def extract_features_for_row(code: str) -> Dict[str, Any]:
    feats = {}
    feats.update(extract_lexical_features(code))
    feats.update(extract_identifier_and_style_features(code))
    ast_feats = extract_ast_structural_features(code)
    feats.update(ast_feats)
    feats["cyclomatic_complexity_est"] = estimate_cyclomatic_complexity(ast_feats)
    return feats
