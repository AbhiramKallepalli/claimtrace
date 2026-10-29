"""Step 1 of ClaimTrace: clean the California IMR file.

Splits each case into:
  request   = the short case summary (what the patient asked for)  -> the model may see this
  reasoning = the reviewer's reasoning (gives the answer away)     -> only used as a library of PAST cases
Then holds out a test set that is never used for searching.
"""
import re
import pandas as pd

RAW = "data/imr_raw.csv"
TRAIN_OUT = "data/cases_train.csv"
TEST_OUT = "data/cases_test.csv"
TEST_SIZE = 1500
SEED = 42

MARKER = re.compile(r"\s*Fin\w{0,3}ings?\s*:", re.I)
PREFIX = re.compile(r"^\s*Nature of Statutory Criteria/Case Summary:\s*", re.I)
LEAK = re.compile(r"overturn|uphold|upheld|not medically necessary|is medically necessary|reviewer found", re.I)


def split_text(text):
    m = MARKER.search(text)
    if not m:
        return None, None
    request = PREFIX.sub("", text[: m.start()]).strip()
    reasoning = text[m.end():].strip()
    return request, reasoning


def main():
    df = pd.read_csv(RAW, low_memory=False)
    print("rows in raw file:", len(df))

    df = df.drop_duplicates(subset="ReferenceID")
    parts = df["Findings"].apply(split_text)
    df["request"] = parts.str[0]
    df["reasoning"] = parts.str[1]

    df = df[df["request"].notna()]
    print("rows with a Findings marker:", len(df))
    df = df[df["request"].str.len() >= 25]
    print("rows with a usable request:", len(df))
    df = df[~df["request"].str.contains(LEAK)]
    print("rows after removing answer-leaking requests:", len(df))
    df = df[df["reasoning"].str.len() >= 100]
    print("rows with usable reasoning:", len(df))

    out = pd.DataFrame({
        "case_id": df["ReferenceID"],
        "year": df["ReportYear"],
        "diagnosis": df["DiagnosisCategory"],
        "treatment": df["TreatmentCategory"],
        "case_type": df["Type"],
        "age": df["AgeRange"].fillna("unknown"),
        "gender": df["PatientGender"].fillna("unknown"),
        "request": df["request"],
        "reasoning": df["reasoning"],
        "label": (df["Determination"].str.startswith("Overturned")).astype(int),
    })

    test = out.sample(n=TEST_SIZE, random_state=SEED)
    train = out.drop(test.index)
    train.to_csv(TRAIN_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)

    print("\ntrain cases (search library):", len(train))
    print("test cases (held out):", len(test))
    print("share overturned in test:", round(test["label"].mean(), 3))
    print("baseline accuracy (always guess the common answer):",
          round(max(test["label"].mean(), 1 - test["label"].mean()), 3))
    print("\nsample request:", test.iloc[0]["request"][:200])


if __name__ == "__main__":
    main()