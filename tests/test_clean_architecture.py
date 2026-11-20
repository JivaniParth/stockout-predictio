"""
Clean Architecture Boundary Enforcement Tests.
Verifies that domain entities and decision policies adhere strictly to Clean Architecture:
Domain and decision code must NOT import infrastructure, database, web frameworks, or deployment runtimes.
"""

import ast
from pathlib import Path
import pytest


FORBIDDEN_FRAMEWORKS = {
    "fastapi",
    "starlette",
    "streamlit",
    "sqlite3",
    "torch.onnx",
    "uvicorn",
    "psutil",
}


def get_all_python_files(directory: Path):
    if not directory.exists():
        return []
    return [p for p in directory.rglob("*.py") if p.is_file() and not p.name.startswith("__")]


def test_decision_layer_dependency_inversion():
    """Asserts that src/scms/decision contains zero infrastructure or framework dependencies."""
    decision_dir = Path("src/scms/decision")
    assert decision_dir.exists(), "src/scms/decision directory must exist"

    py_files = get_all_python_files(decision_dir)
    assert len(py_files) >= 1, "At least one decision policy module must be tested"

    violations = []
    for file_path in py_files:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split(".")[0]
                    if root_pkg in FORBIDDEN_FRAMEWORKS:
                        violations.append(f"{file_path}: imports forbidden framework '{alias.name}'")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_pkg = node.module.split(".")[0]
                    if root_pkg in FORBIDDEN_FRAMEWORKS:
                        violations.append(f"{file_path}: imports from forbidden module '{node.module}'")

    assert not violations, "Clean architecture violations detected in decision layer:\n" + "\n".join(violations)


def test_use_cases_depend_on_ports_not_adapters():
    """Asserts that application use cases do not directly import concrete infrastructure adapters."""
    app_use_cases_dir = Path("src/application/use_cases")
    if not app_use_cases_dir.exists():
        pytest.skip("Legacy src/application/use_cases not present in workspace")

    forbidden_adapter_imports = {"src.infrastructure.adapters", "src.infrastructure.persistence"}
    violations = []

    for file_path in get_all_python_files(app_use_cases_dir):
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for forbidden in forbidden_adapter_imports:
                    if node.module.startswith(forbidden):
                        violations.append(f"{file_path}: use-case imports concrete adapter '{node.module}'")

    assert not violations, "Use cases must depend on ports, not concrete adapters:\n" + "\n".join(violations)
