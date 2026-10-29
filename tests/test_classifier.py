import pandas as pd
import classifier


def tiny():
    rows = []
    for i in range(60):
        over = i % 2
        rows.append({"request": "needs genomic testing" if over else "wants cosmetic surgery",
                     "diagnosis": "A", "treatment": "B", "case_type": "Medical Necessity",
                     "age": "41 to 50", "gender": "Female", "label": over})
    return pd.DataFrame(rows)


def test_learns_and_abstains_when_threshold_high():
    est = classifier.Estimator().fit(tiny())
    case = {"request": "needs genomic testing", "diagnosis": "A", "treatment": "B",
            "case_type": "Medical Necessity", "age": "41 to 50", "gender": "Female"}
    pred, conf, p = est.estimate_one(case)
    assert pred == "overturned" and p > 0.5
    est.threshold = 0.99
    assert est.estimate_one(case)[0] == "unsure"


def test_pick_threshold_returns_a_number():
    import numpy as np
    p = np.array([0.9] * 150 + [0.1] * 150)
    y = np.array([1] * 150 + [0] * 150)
    assert 0.0 <= classifier.pick_threshold(p, y) <= 0.9