#!/usr/bin/env python3
"""Validate a competition submission zip (agent.py + weights.pt)."""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

MAX_BYTES = 5 * 1024 * 1024
MAX_PARAMS = 50_000
MAX_ACT_SECONDS = 0.010
ALLOWED_IMPORTS = frozenset({"os", "numpy", "torch"})
FORBIDDEN_CALL_ROOTS = frozenset(
    {"socket", "requests", "urllib", "subprocess", "pickle"}
)


def _fail(msg: str) -> int:
    print(f"INVALID: {msg}")
    return 1


def _ok(msg: str) -> int:
    print(f"VALID: {msg}")
    return 0


def _module_root(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        cur: ast.AST = node
        while isinstance(cur, ast.Attribute):
            cur = cur.value
        if isinstance(cur, ast.Name):
            return cur.id
    return None


def _is_write_open(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Name) and func.id == "open":
        pass
    elif isinstance(func, ast.Attribute) and func.attr == "open":
        pass
    else:
        return False
    if len(call.args) >= 2:
        mode = call.args[1]
        if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
            return any(c in mode.value for c in "wax+")
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return any(c in kw.value.value for c in "wax+")
    return False


def _check_ast(source: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"agent.py has a syntax error: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    return f"disallowed import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                return "disallowed relative import"
            root = node.module.split(".")[0]
            if root not in ALLOWED_IMPORTS:
                return f"disallowed import: {node.module}"

        if isinstance(node, ast.Call):
            if _is_write_open(node):
                return "filesystem writes via open() are not allowed"
            root = _module_root(node.func)
            if root in FORBIDDEN_CALL_ROOTS:
                return f"disallowed call involving {root}"
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "os" and node.func.attr in {
                    "system",
                    "remove",
                    "unlink",
                    "popen",
                }:
                    return f"disallowed call: os.{node.func.attr}"

    # Scan act() body for exploration noise.
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Agent":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "act":
                    act_src = ast.get_source_segment(source, item) or ""
                    for needle in ("np.random", "torch.rand", "torch.randn", "random."):
                        if needle in act_src:
                            return f"act() must be deterministic; found {needle}"
    return None


def _count_params(net: torch.nn.Module) -> int:
    try:
        params = list(net.parameters())
    except Exception:
        params = []
    if params:
        return int(sum(p.numel() for p in params))
    try:
        state = net.state_dict()
    except Exception:
        return 0
    total = 0
    for tensor in state.values():
        if torch.is_tensor(tensor):
            total += int(tensor.numel())
    return total


def validate(zip_path: Path) -> int:
    if not zip_path.is_file():
        return _fail(f"zip not found: {zip_path}")

    zip_size = zip_path.stat().st_size
    if zip_size > MAX_BYTES:
        return _fail(f"zip file size {zip_size} exceeds {MAX_BYTES} bytes")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            # Reject path traversal and nested paths.
            members = []
            for name in names:
                if name.endswith("/"):
                    continue
                pure = Path(name)
                if pure.is_absolute() or ".." in pure.parts:
                    return _fail(f"path traversal rejected: {name}")
                if len(pure.parts) != 1:
                    return _fail(f"nested paths not allowed: {name}")
                members.append(pure.name)

            if sorted(members) != ["agent.py", "weights.pt"]:
                return _fail(
                    f"zip members must be exactly agent.py and weights.pt; got {sorted(members)}"
                )

            uncompressed = sum(info.file_size for info in zf.infolist() if not info.is_dir())
            if uncompressed > MAX_BYTES:
                return _fail(f"uncompressed size {uncompressed} exceeds {MAX_BYTES} bytes")

            with tempfile.TemporaryDirectory(prefix="hockey_validate_") as tmp:
                tmp_path = Path(tmp)
                zf.extractall(tmp_path)
                agent_py = tmp_path / "agent.py"
                weights_pt = tmp_path / "weights.pt"

                source = agent_py.read_text(encoding="utf-8")
                ast_error = _check_ast(source)
                if ast_error:
                    return _fail(ast_error)

                try:
                    net = torch.jit.load(str(weights_pt), map_location="cpu")
                except Exception as exc:
                    return _fail(f"weights.pt must load with torch.jit.load: {exc}")

                n_params = _count_params(net)
                if n_params > MAX_PARAMS:
                    return _fail(f"policy has {n_params} parameters; max is {MAX_PARAMS}")

                module_name = "_hockey_validate_agent"
                spec = importlib.util.spec_from_file_location(module_name, agent_py)
                if spec is None or spec.loader is None:
                    return _fail("could not import agent.py")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                old_cwd = os.getcwd()
                try:
                    os.chdir(tmp_path)
                    sys.path.insert(0, str(tmp_path))
                    spec.loader.exec_module(module)
                    agent_cls = getattr(module, "Agent", None)
                    if agent_cls is None:
                        return _fail("agent.py must define class Agent")
                    agent = agent_cls()
                except Exception as exc:
                    return _fail(f"failed to instantiate Agent: {exc}")
                finally:
                    os.chdir(old_cwd)
                    if str(tmp_path) in sys.path:
                        sys.path.remove(str(tmp_path))
                    sys.modules.pop(module_name, None)

                try:
                    action = agent.act(np.zeros(18, dtype=np.float32))
                except Exception as exc:
                    return _fail(f"act() raised: {exc}")

                arr = np.asarray(action)
                if arr.shape != (4,):
                    return _fail(f"act() must return shape (4,), got {arr.shape}")
                if arr.dtype != np.float32:
                    return _fail(f"act() must return float32, got {arr.dtype}")
                if not np.all(np.isfinite(arr)):
                    return _fail("act() returned non-finite values")
                if np.any(arr < -1.0) or np.any(arr > 1.0):
                    return _fail("act() values must be in [-1, 1]")

                # Warmup then time.
                zeros = np.zeros(18, dtype=np.float32)
                for _ in range(2):
                    agent.act(zeros)
                for i in range(10):
                    t0 = time.perf_counter()
                    agent.act(zeros)
                    dt = time.perf_counter() - t0
                    if dt >= MAX_ACT_SECONDS:
                        return _fail(
                            f"act() call {i} took {dt * 1000:.2f} ms; limit is {MAX_ACT_SECONDS * 1000:.0f} ms"
                        )

                return _ok(
                    f"submission ok ({n_params} params, zip {zip_size} bytes)"
                )
    except zipfile.BadZipFile:
        return _fail("not a valid zip file")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("Usage: python validate.py submission.zip", file=sys.stderr)
        return 2
    return validate(Path(argv[0]))


if __name__ == "__main__":
    raise SystemExit(main())
