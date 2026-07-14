#!/usr/bin/env python3
"""
semantic_hash.py -- fingerprint what a script COMPUTES, not how it is commented.
Larry (Peilin) Zhong

run_all.sh gates on the SHA-256 of the source bytes, which is what you want: it
refuses to run if the files are not the archived files, and it caught two machines
executing different code.

But a byte hash also changes when a comment changes, and then a cross-machine log
that verified the numbers no longer matches the archive, even though every number
in it is still correct. The alternatives are both bad: re-run an hour of
simulation to certify a comment, or assert "only the comments moved" and ask the
reader to take it on trust.

So: hash the ABSTRACT SYNTAX TREE with docstrings and comments removed. Two files
with the same semantic hash compute the same thing. That is not an opinion about
the diff, it is a property of the parsed program.

Usage
-----
  python3 semantic_hash.py                 # print the semantic hash of each gated file
  python3 semantic_hash.py OLD_DIR         # compare this archive against another copy

Exit 0 if every file is semantically identical, 1 otherwise.
"""

import ast
import hashlib
import pathlib
import sys

from spectral_null import GATED


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return tree


def semantic_hash(path):
    """SHA-256 of the AST dump, docstrings removed. Comments never reach the AST."""
    tree = ast.parse(pathlib.Path(path).read_bytes())
    dump = ast.dump(strip_docstrings(tree), annotate_fields=True,
                    include_attributes=False)
    return hashlib.sha256(dump.encode()).hexdigest()[:12]


def byte_hash(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:12]


def main():
    here = pathlib.Path(__file__).parent
    other = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None

    print("=" * 78)
    print("SEMANTIC FINGERPRINT")
    print("=" * 78)
    if other:
        print(f"  comparing  {here}")
        print(f"  against    {other}\n")
        print(f"  {'file':<22}{'this: bytes':>13}{'sem':>10}"
              f"{'  |':>3}{'other: bytes':>14}{'sem':>10}   verdict")
    else:
        print(f"  {'file':<22}{'source (bytes)':>16}{'semantic (AST)':>16}")
    print("  " + "-" * 74)

    differ = 0
    for fn in GATED:
        p = here / fn
        if not p.exists():
            print(f"  {fn:<22}  MISSING")
            differ += 1
            continue
        bh, sh = byte_hash(p), semantic_hash(p)
        if not other:
            print(f"  {fn:<22}{bh:>16}{sh:>16}")
            continue
        q = other / fn
        if not q.exists():
            print(f"  {fn:<22}{bh:>13}{sh:>10}  |  {'MISSING':>14}")
            differ += 1
            continue
        bh2, sh2 = byte_hash(q), semantic_hash(q)
        same = sh == sh2
        differ += (not same)
        verdict = "identical" if same else "*** DIFFERENT CODE ***"
        print(f"  {fn:<22}{bh:>13}{sh:>10}  |{bh2:>14}{sh2:>10}   {verdict}")

    print()
    if not other:
        print("  The source hash is what run_all.sh gates on. The semantic hash is what")
        print("  the file COMPUTES: same AST, docstrings and comments removed.")
        print("=" * 78)
        return 0

    if differ:
        print(f"  *** {differ} file(s) compute something different. A log produced by one")
        print(f"  *** archive does not certify the other. Re-run.")
    else:
        print("  Every file computes the same thing. The source bytes differ (comments,")
        print("  docstrings, printed prose), the programs do not. A log produced by")
        print("  either archive certifies both, and no re-run is needed to establish it.")
    print("=" * 78)
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
