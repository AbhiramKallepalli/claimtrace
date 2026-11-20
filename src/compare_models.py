"""Compare local models on the same held-out cases.
Usage:  python src/compare_models.py llama3.2:3b 40
        python src/compare_models.py qwen2.5:7b 40
Same random sample (fixed seed) every time, so models see identical cases.
"""
import sys
import time

import pandas as pd

import classifier
import harness
from harness import run_agent
from search import CaseSearch


def main():
    model = sys.argv[1]
    n = int(sys.argv[2])
    harness.MODEL = model
    searcher = CaseSearch()
    estimator = classifier.load()
    test = pd.read_csv("data/cases_test.csv").sample(n=n, random_state=42)
    rows = []
    start = time.time()
    for i, (_, row) in enumerate(test.iterrows(), 1):
        t0 = time.time()
        out = run_agent(row.to_dict(), searcher, estimator=estimator)
        truth = "overturned" if row["label"] == 1 else "upheld"
        rows.append({
            "case_id": row["case_id"],
            "truth": truth,
            "prediction": out["prediction"],
            "confidence": out["confidence"],
            "status": out["status"],
            "overridden": bool(out.get("overridden", False)),
            "seconds": round(time.time() - t0, 1),
        })
        print(f"{i}/{n} {row['case_id']} {out['status']} {rows[-1]['seconds']}s", flush=True)

    df = pd.DataFrame(rows)
    path = "outputs/compare_" + model.replace(":", "_") + ".csv"
    df.to_csv(path, index=False)
    answered = df[df["prediction"] != "unsure"]
    right = int((answered["prediction"] == answered["truth"]).sum())
    with_note = int((df["status"] == "ok").sum())
    print("\n==== " + model + " ====")
    print(f"cases: {n}, took {(time.time() - start) / 60:.1f} min")
    print(f"answered by system: {len(answered)}, correct: {right}")
    print(f"sent to human: {n - len(answered)}")
    print(f"model gave a usable explanation: {with_note} of {len(answered)} answered")
    print(f"model disagreed and was overridden: {int(df['overridden'].sum())}")
    print("saved to " + path)


if __name__ == "__main__":
    main()