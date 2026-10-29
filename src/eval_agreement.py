"""When the similar past cases agree strongly, is the prediction more accurate?
Usage:  python src\eval_agreement.py 300
"""
import sys

import pandas as pd

from search import CaseSearch


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    searcher = CaseSearch()
    test = pd.read_csv("data/cases_test.csv").head(n)
    rows = []
    for _, row in test.iterrows():
        found = searcher.search(row["diagnosis"], row["treatment"], row["request"], k=5)
        overturned = sum(r["outcome"] == "overturned" for r in found)
        rows.append((overturned, int(row["label"])))
    print(f"\nTested {n} held-out cases. 'agreement' = how many of the 5 similar cases voted the same way.")
    for need in (3, 4, 5):
        picked = [(o, y) for o, y in rows if max(o, 5 - o) >= need]
        if not picked:
            continue
        right = sum((o >= 3) == bool(y) for o, y in picked)
        print(f"  at least {need} of 5 agree: answers {len(picked) / n:.0%} of cases, "
              f"correct {right / len(picked):.1%} of those")


if __name__ == "__main__":
    main()