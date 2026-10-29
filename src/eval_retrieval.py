"""Does the search alone predict the outcome? Majority vote of the most similar past cases.
Usage:  python src\eval_retrieval.py 300
"""
import sys

import pandas as pd

from search import CaseSearch


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    searcher = CaseSearch()
    test = pd.read_csv("data/cases_test.csv").head(n)
    hits = {1: 0, 5: 0, 11: 0}
    for i, (_, row) in enumerate(test.iterrows()):
        found = searcher.search(row["diagnosis"], row["treatment"], row["request"], k=11)
        votes = [1 if r["outcome"] == "overturned" else 0 for r in found]
        for k in hits:
            top = votes[:k]
            hits[k] += int((sum(top) / len(top) >= 0.5) == bool(row["label"]))
        if (i + 1) % 100 == 0:
            print(f"  done {i + 1} / {n}", flush=True)
    base = max(test["label"].mean(), 1 - test["label"].mean())
    print(f"\nTested {n} held-out cases")
    for k, h in hits.items():
        print(f"  majority of top {k} similar cases: {h / n:.3f}")
    print(f"  always guess the common answer:    {base:.3f}")


if __name__ == "__main__":
    main()