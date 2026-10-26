#!/usr/bin/env python3
"""Static policy gate for autonomous candidate code."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

FORBIDDEN_IMPORTS = {
    "ctypes",
    "ftplib",
    "http",
    "importlib",
    "multiprocessing",
    "pickle",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "urllib",
}
FORBIDDEN_CALLS = {"__import__", "compile", "eval", "exec", "input", "open"}
FORBIDDEN_ATTRIBUTES = {
    "os.popen",
    "os.system",
    "pathlib.Path.chmod",
    "pathlib.Path.unlink",
    "torch.load",
}


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def inspect_candidate(path: str | Path) -> list[str]:
    candidate = Path(path)
    tree = ast.parse(candidate.read_text(encoding="utf-8"), filename=str(candidate))
    findings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                root = name.split(".", 1)[0]
                if root in FORBIDDEN_IMPORTS:
                    findings.append(f"line {node.lineno}: forbidden import {name}")
        if isinstance(node, ast.Call):
            call_name = dotted_name(node.func)
            if call_name in FORBIDDEN_CALLS or call_name in FORBIDDEN_ATTRIBUTES:
                findings.append(f"line {node.lineno}: forbidden call {call_name}")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", nargs="?", default="train.py")
    arguments = parser.parse_args()
    findings = inspect_candidate(arguments.candidate)
    if findings:
        print("candidate rejected by security policy")
        for finding in findings:
            print(f"- {finding}")
        raise SystemExit(1)
    print(f"security policy passed: {arguments.candidate}")


if __name__ == "__main__":
    main()
