"""One command to reproduce every number in the report.

Usage:
    python run_all.py            # all tasks
    python run_all.py --task 2   # one task
"""
import argparse

TASKS = {1: [], 2: [], 3: []}  # each task registers its pipeline steps here


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
