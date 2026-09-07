import ast
import sys
from pathlib import Path

FORBIDDEN_IMPORTS = {
    "core": ["torch", "transformers", "onnx", "onnxruntime", "fastapi", "riskforge.modeling", "riskforge.serving"],
    "normalization": ["torch", "transformers", "onnx", "onnxruntime", "riskforge.modeling", "riskforge.serving"],
    "supervision": ["torch", "transformers", "onnx", "onnxruntime", "riskforge.modeling", "riskforge.serving"],
    "metrics": ["torch", "transformers", "onnx", "onnxruntime", "riskforge.modeling"],
    "serving": ["torch", "transformers"]
}

def check_file(file_path: Path, subsystem: str) -> list[str]:
    violations = []
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    forbidden = FORBIDDEN_IMPORTS.get(subsystem, [])

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for f in forbidden:
                    if alias.name == f or alias.name.startswith(f + "."):
                        violations.append(f"Line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for f in forbidden:
                if mod == f or mod.startswith(f + "."):
                    violations.append(f"Line {node.lineno}: from {mod} import ...")
    return violations

def main() -> int:
    base = Path(__file__).resolve().parents[1] / "src" / "riskforge"
    failed = False

    for subsystem, forbidden in FORBIDDEN_IMPORTS.items():
        sub_dir = base / subsystem
        if not sub_dir.exists():
            continue
        for py_file in sub_dir.glob("**/*.py"):
            violations = check_file(py_file, subsystem)
            if violations:
                failed = True
                print(f"[BOUNDARY VIOLATION] {py_file.relative_to(base)}:")
                for v in violations:
                    print(f"  {v} (Forbidden for subsystem '{subsystem}')")

    if failed:
        print("\nBoundary check failed: Modules must adhere to CONTEXT.md dependency rules.")
        return 1
    print("Boundary check passed: Zero illegal cross-subsystem dependencies.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
