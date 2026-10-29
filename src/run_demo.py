"""Run the agent on a few held-out test cases and compare with the real outcome.
Usage:  python src\run_demo.py 3      (runs the first 3 test cases)
"""
import sys

import pandas as pd

from harness import run_agent
from search import CaseSearch


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    searcher = CaseSearch()
    test = pd.read_csv("data/cases_test.csv").head(n)
    right = 0
    for _, row in test.iterrows():
        out = run_agent(row.to_dict(), searcher)
        truth = "overturned" if row["label"] == 1 else "upheld"
        right += out["prediction"] == truth
        print(f"\ncase {row['case_id']}: predicted {out['prediction']} ({out['confidence']}), real outcome {truth}, status {out['status']}")
        print("  reasons:", out["reasons"])
    print(f"\n{right} of {n} matched the real outcome.")


if __name__ == "__main__":
    main()