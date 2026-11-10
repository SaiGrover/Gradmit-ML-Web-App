"""Check that every notebook parses, is fully executed and holds no error output.

    python scripts/check_notebooks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
EXPECTED = [
    "01_data_audit_and_eda.ipynb",
    "02_model_comparison_and_baselines.ipynb",
    "03_threshold_sensitivity_and_statistics.ipynb",
    "04_error_analysis_and_final_model.ipynb",
]


def check(path: Path) -> list[str]:
    problems = []
    nb = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code" and "".join(c["source"]).strip()]
    if not code_cells:
        problems.append("no code cells")
    counts = [c.get("execution_count") for c in code_cells]
    if any(c is None for c in counts):
        problems.append("contains unexecuted code cells")
    elif counts != list(range(1, len(counts) + 1)):
        problems.append("cells were not executed top to bottom in one run")
    for cell in code_cells:
        if any(o.get("output_type") == "error" for o in cell.get("outputs", [])):
            problems.append("contains an error output")
            break
    if not any(c.get("outputs") for c in code_cells):
        problems.append("has no saved outputs")
    return problems


def main() -> int:
    failed = False
    for name in EXPECTED:
        path = NOTEBOOKS / name
        problems = check(path) if path.exists() else ["missing"]
        print(f"{'FAIL' if problems else 'ok  '} {name} {'; '.join(problems)}")
        failed = failed or bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
