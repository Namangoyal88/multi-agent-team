"""Restricted experiment execution with AST validation and a minimal runtime.

This is defense-in-depth for LLM-generated experiments, not a complete security
boundary. For hostile/untrusted code, run experiments in a container or VM.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Only computational standard-library modules are permitted. No filesystem, network,
# process, dynamic-import, or native-extension modules are allowed in this runtime.
ALLOWED_IMPORTS = {
    "math", "statistics", "random", "json", "re", "collections",
    "itertools", "functools", "decimal", "fractions",
}
FORBIDDEN_NAMES = {
    "exec", "eval", "compile", "__import__", "open", "input", "breakpoint",
    "globals", "locals", "getattr", "setattr", "delattr", "vars", "help", "dir",
}
DANGEROUS_ATTRS = {
    "system", "popen", "Popen", "run", "call", "check_call", "check_output",
    "getoutput", "spawn", "spawnl", "spawnlp", "fork", "kill",
}

SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "enumerate": enumerate,
    "filter": filter, "float": float, "frozenset": frozenset, "int": int, "len": len,
    "list": list, "map": map, "max": max, "min": min, "pow": pow, "print": print,
    "range": range, "repr": repr, "reversed": reversed, "round": round, "set": set,
    "slice": slice, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "ZeroDivisionError": ZeroDivisionError, "ArithmeticError": ArithmeticError,
}


class ExecutionDisabled(Exception):
    pass


def _root_module(name: str) -> str:
    return name.split(".", 1)[0]


def validate_code(code: str) -> list[str]:
    problems: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]

    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if _root_module(a.name) not in ALLOWED_IMPORTS:
                    problems.append(f"import of '{a.name}' is not allowed")
        elif isinstance(n, ast.ImportFrom):
            if _root_module(n.module or "") not in ALLOWED_IMPORTS:
                problems.append(f"import from '{n.module}' is not allowed")
        elif isinstance(n, ast.Name):
            if n.id in FORBIDDEN_NAMES or (n.id.startswith("__") and n.id.endswith("__")):
                problems.append(f"name '{n.id}' is not allowed")
        elif isinstance(n, ast.Attribute):
            if n.attr in DANGEROUS_ATTRS:
                problems.append(f"dangerous attribute '{n.attr}' is not allowed")
            if n.attr.startswith("__") and n.attr.endswith("__"):
                problems.append(f"dunder attribute '{n.attr}' is not allowed")
    return list(dict.fromkeys(problems))


def _runner_source(script: Path) -> str:
    # The wrapper itself retains import/exec capabilities, but these names are never visible to
    # validated user code because user code receives a separate restricted globals dictionary.
    safe_names = repr(tuple(SAFE_BUILTINS))
    imports = repr(sorted(ALLOWED_IMPORTS))
    script_repr = repr(str(script))
    return f'''import builtins as _real_builtins\nfrom pathlib import Path as _Path\n\n_ALLOWED_IMPORTS = {imports}\n_SAFE = {{name: _real_builtins.__dict__[name] for name in {safe_names}}}\n\ndef _safe_import(name, globals=None, locals=None, fromlist=(), level=0):\n    root = name.split(".", 1)[0]\n    if root not in _ALLOWED_IMPORTS:\n        raise ImportError(f"import of {{name!r}} is blocked by experiment sandbox")\n    return _real_builtins.__import__(name, globals, locals, fromlist, level)\n\n_SAFE["__import__"] = _safe_import\n_source = _Path({script_repr}).read_text(encoding="utf-8")\n_globals = {{"__name__": "__main__", "__builtins__": _SAFE}}\nexec(compile(_source, {script_repr}, "exec"), _globals, _globals)\n'''


def run_experiment(code: str, *, enabled: bool, timeout: int = 30,
                   artifact_dir: str | None = None) -> dict:
    if not enabled:
        raise ExecutionDisabled("Code execution is disabled (set ENABLE_CODE_EXECUTION=true).")
    problems = validate_code(code)
    if problems:
        return {"executed": False, "status": "rejected", "problems": problems, "metrics": None,
                "stdout": "", "stderr": "", "artifacts": []}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        script = tmp_path / "experiment.py"
        runner = tmp_path / "runner.py"
        script.write_text(code, encoding="utf-8")
        runner.write_text(_runner_source(script), encoding="utf-8")
        start = time.time()
        try:
            p = subprocess.run([sys.executable, "-I", str(runner)], cwd=tmp, capture_output=True,
                               text=True, timeout=timeout, env={})
            status = "ok" if p.returncode == 0 else "error"
            stdout, stderr, rc = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as exc:
            status, rc = "timeout", None
            raw_out = exc.stdout or ""
            stdout = raw_out.decode(errors="replace") if isinstance(raw_out, bytes) else raw_out
            stderr = f"timed out after {timeout}s"

        metrics = None
        for line in reversed(stdout.splitlines()):
            if line.startswith("METRICS_JSON:"):
                try:
                    metrics = json.loads(line.split(":", 1)[1])
                except ValueError:
                    stderr = (stderr + "\n" if stderr else "") + "invalid METRICS_JSON line"
                break

        artifacts = []
        for f in tmp_path.iterdir():
            if f.name in {"experiment.py", "runner.py"} or not f.is_file():
                continue
            artifacts.append({"name": f.name, "size": f.stat().st_size,
                              "content": f.read_text(errors="replace")[:20000]})
        if artifact_dir:
            d = Path(artifact_dir)
            d.mkdir(parents=True, exist_ok=True)
            (d / "experiment.py").write_text(code, encoding="utf-8")
            for a in artifacts:
                (d / a["name"]).write_text(a["content"], encoding="utf-8")
    return {"executed": True, "status": status, "returncode": rc, "stdout": stdout[-8000:],
            "stderr": stderr[-4000:], "metrics": metrics, "artifacts": artifacts,
            "duration_s": round(time.time() - start, 2)}
