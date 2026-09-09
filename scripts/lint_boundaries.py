import ast
import sys
from pathlib import Path

FORBIDDEN_IMPORTS = {
    "core": [
        "torch",
        "transformers",
        "onnx",
        "onnxruntime",
        "fastapi",
        "riskforge.modeling",
        "riskforge.serving",
    ],
    "normalization": [
        "torch",
        "transformers",
        "onnx",
        "onnxruntime",
        "riskforge.modeling",
        "riskforge.serving",
    ],
    "supervision": [
        "torch",
        "transformers",
        "onnx",
        "onnxruntime",
        "riskforge.modeling",
        "riskforge.serving",
    ],
    "metrics": [
        "torch",
        "transformers",
        "onnx",
        "onnxruntime",
        "riskforge.modeling",
    ],
    "serving": ["torch", "transformers"],
    "application": [
        "torch",
        "transformers",
        "onnx",
        "onnxruntime",
        "fastapi",
        "riskforge.modeling",
        "riskforge.normalization",
        "riskforge.supervision",
        "riskforge.serving",
        "riskforge.persistence",
        "riskforge.review",
    ],
    "persistence": [
        "torch",
        "transformers",
        "onnx",
        "onnxruntime",
        "fastapi",
        "riskforge.modeling",
        "riskforge.normalization",
        "riskforge.supervision",
        "riskforge.serving",
        "riskforge.application",
        "riskforge.review",
    ],
    "review": [
        "torch",
        "transformers",
        "onnx",
        "onnxruntime",
        "fastapi",
        "riskforge.modeling",
        "riskforge.normalization",
        "riskforge.supervision",
        "riskforge.serving",
        "riskforge.metrics",
    ],
    "api": [
        "torch",
        "transformers",
        "onnx",
        "onnxruntime",
        "riskforge.modeling",
        "riskforge.normalization",
        "riskforge.supervision",
        "riskforge.serving",
        "riskforge.metrics",
        "riskforge.persistence",
        "riskforge.review",
    ],
    "runtime": [
        "torch",
        "transformers",
        "onnx",
        "onnxruntime",
        "riskforge.modeling",
    ],
    "authentication": [
        "torch",
        "transformers",
        "onnx",
        "onnxruntime",
        "fastapi",
        "riskforge.core",
        "riskforge.modeling",
        "riskforge.normalization",
        "riskforge.supervision",
        "riskforge.serving",
        "riskforge.metrics",
        "riskforge.persistence",
        "riskforge.application",
        "riskforge.review",
        "riskforge.api",
        "riskforge.runtime",
    ],
}


def check_file(file_path: Path, subsystem: str) -> list[str]:
    violations = []
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    forbidden = FORBIDDEN_IMPORTS.get(subsystem, [])

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for forbidden_module in forbidden:
                    if alias.name == forbidden_module or alias.name.startswith(forbidden_module + "."):
                        violations.append(f"Line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for forbidden_module in forbidden:
                if module == forbidden_module or module.startswith(forbidden_module + "."):
                    violations.append(f"Line {node.lineno}: from {module} import ...")
    return violations


def main() -> int:
    base = Path(__file__).resolve().parents[1] / "src" / "riskforge"
    failed = False

    for subsystem in FORBIDDEN_IMPORTS:
        sub_dir = base / subsystem
        if not sub_dir.exists():
            continue
        for py_file in sub_dir.glob("**/*.py"):
            violations = check_file(py_file, subsystem)
            if violations:
                failed = True
                print(f"[BOUNDARY VIOLATION] {py_file.relative_to(base)}:")
                for violation in violations:
                    print(f"  {violation} (Forbidden for subsystem '{subsystem}')")

    if failed:
        print("\nBoundary check failed: Modules must adhere to CONTEXT.md dependency rules.")
        return 1
    print("Boundary check passed: Zero illegal cross-subsystem dependencies.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
