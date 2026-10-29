"""Simple, fast outcome estimator: word patterns + case fields -> probability the denial is overturned.
Used as a tool by the agent and as a baseline in the report.
The "answer / not sure" threshold is chosen on a validation slice of the TRAIN data, never on the test cases.

Usage:  python src/classifier.py
"""
import os
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

TRAIN = "data/cases_train.csv"
TEST = "data/cases_test.csv"
MODEL_FILE = "outputs/classifier.pkl"
VAL_SIZE = 2000
SEED = 7
TARGET_ACCURACY = 0.80
BANDS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6)


def meta(df):
    return (df["diagnosis"] + " | " + df["treatment"] + " | " + df["case_type"] + " | "
            + df["age"].astype(str) + " | " + df["gender"].astype(str)).values


class Estimator:
    def __init__(self):
        self.words = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)
        self.fields = TfidfVectorizer(token_pattern=r"[^|]+", min_df=2)
        self.model = LogisticRegression(C=2, max_iter=2000)
        self.threshold = 0.4

    def fit(self, df):
        X = hstack([self.words.fit_transform(df["request"]), self.fields.fit_transform(meta(df))]).tocsr()
        self.model.fit(X, df["label"])
        return self

    def prob_overturned(self, df):
        X = hstack([self.words.transform(df["request"]), self.fields.transform(meta(df))]).tocsr()
        return self.model.predict_proba(X)[:, 1]

    def estimate_one(self, case):
        """Return (prediction, confidence 0..1, probability overturned) for one case dict."""
        p = float(self.prob_overturned(pd.DataFrame([case]))[0])
        conf = abs(p - 0.5) * 2
        pred = "overturned" if p >= 0.5 else "upheld"
        if conf < self.threshold:
            pred = "unsure"
        return pred, conf, p


def table(p, y, title):
    conf = np.abs(p - 0.5) * 2
    print(title)
    for t in BANDS:
        s = conf >= t
        if s.sum() == 0:
            continue
        acc = ((p[s] >= 0.5) == y[s]).mean()
        print(f"  confidence >= {t:.1f}: answers {s.mean():.0%} of cases, correct {acc:.1%} of those")


def pick_threshold(p, y):
    conf = np.abs(p - 0.5) * 2
    best = 0.9
    for t in np.arange(0.0, 0.95, 0.05):
        s = conf >= t
        if s.sum() < 100:
            break
        if ((p[s] >= 0.5) == y[s]).mean() >= TARGET_ACCURACY:
            return float(round(t, 2))
    return best


def load():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


def main():
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    val = train.sample(n=VAL_SIZE, random_state=SEED)
    fit_part = train.drop(val.index)
    est = Estimator().fit(fit_part)
    pv = est.prob_overturned(val)
    table(pv, val["label"].values, f"\nValidation slice ({VAL_SIZE} cases from the training data):")
    est.threshold = pick_threshold(pv, val["label"].values)
    print(f"\nChosen threshold (lowest confidence that reaches {TARGET_ACCURACY:.0%} on validation): {est.threshold}")
    pt = est.prob_overturned(test)
    table(pt, test["label"].values, f"\nTest cases ({len(test)} held out, used once):")
    s = np.abs(pt - 0.5) * 2 >= est.threshold
    acc = ((pt[s] >= 0.5) == test["label"].values[s]).mean()
    print(f"\nWith the chosen threshold the system answers {s.mean():.0%} of test cases, "
          f"correct {acc:.1%} of those, and says 'not sure' on the rest.")
    os.makedirs("outputs", exist_ok=True)
    import classifier as this_module  # so the saved file points at classifier.Estimator, not __main__
    est.__class__ = this_module.Estimator
    with open(MODEL_FILE, "wb") as f:
    
        pickle.dump(est, f)
    print(f"Saved to {MODEL_FILE}")


if __name__ == "__main__":
    main()