"""One command to reproduce every number in the report.

Usage:
    python run_all.py            # all tasks
    python run_all.py --task 2   # one task
"""
import argparse

def run_task3() -> None:
    """Execute the Task 3 notebook; it writes cluster.csv, results.json and figures itself."""
    import subprocess
    import sys
    from pathlib import Path

    here = Path(__file__).resolve().parent
    out = here / "results" / "task3"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
         str(here / "task3_clustering" / "task3_clustering.ipynb"), "--output-dir", str(out),
         "--ExecutePreprocessor.timeout=-1"],
        check=True,
    )


TASKS = {1: [], 2: [], 3: [run_task3]}  # each task registers its pipeline steps here


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=sorted(TASKS), default=None)
    args = parser.parse_args()
    for t in ([args.task] if args.task else sorted(TASKS)):
        if not TASKS[t]:
            print(f"Task {t}: no steps implemented yet")
        for step in TASKS[t]:
            print(f"Task {t}: running {step.__module__}")
            step()


if __name__ == "__main__":
    main()
