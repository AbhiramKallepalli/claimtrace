"""Run the agent on a few held-out test cases and compare with the real outcome.
Usage:  python src/run_demo.py 3      (runs the first 3 test cases)
"""
import sys

import pandas as pd

import classifier
from harness import run_agent
from search import CaseSearch


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    searcher = CaseSearch()
    estimator = classifier.load()
    test = pd.read_csv("data/cases_test.csv").head(n)
    answered = right = 0
    for _, row in test.iterrows():
        out = run_agent(row.to_dict(), searcher, estimator=estimator)
        truth = "overturned" if row["label"] == 1 else "upheld"
        print(f"\ncase {row['case_id']}: {out['prediction']} ({out['confidence']}), real outcome {truth}, status {out['status']}")
        print("  reasons:", out["reasons"])
        if out["prediction"] != "unsure":
            answered += 1
            right += out["prediction"] == truth
    print(f"\nAnswered {answered} of {n}; {right} of those matched the real outcome. The rest went to a human.")


if __name__ == "__main__":
    main()