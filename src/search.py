"""Step 2 of ClaimTrace: find similar PAST cases for a new request.

Hybrid search = keyword search (BM25) + meaning search (embeddings from Ollama),
merged with reciprocal rank fusion. Only training cases are searchable, so a test
case can never find itself.
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import requests
from rank_bm25 import BM25Okapi

TRAIN = "data/cases_train.csv"
EMB_FILE = "data/train_embeddings.npy"
OLLAMA_URL = "http://localhost:11434/api/embed"
EMB_MODEL = "nomic-embed-text"


def case_text(diagnosis, treatment, request):
    return f"{diagnosis} | {treatment} | {request}"[:1200]


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def embed(texts, prefix):
    """Turn texts into number vectors using the local Ollama model."""
    out = []
    for i in range(0, len(texts), 32):
        batch = [prefix + t for t in texts[i:i + 32]]
        r = requests.post(OLLAMA_URL, json={"model": EMB_MODEL, "input": batch}, timeout=300)
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
        if (i // 32) % 20 == 0:
            print(f"  embedded {min(i + 32, len(texts))} / {len(texts)}", flush=True)
    arr = np.array(out, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr


class CaseSearch:
    def __init__(self, embed_fn=embed):
        self.df = pd.read_csv(TRAIN)
        self.texts = [case_text(d, t, r) for d, t, r in
                      zip(self.df["diagnosis"], self.df["treatment"], self.df["request"])]
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        self.embed_fn = embed_fn
        if os.path.exists(EMB_FILE):
            self.emb = np.load(EMB_FILE)
        else:
            print("Building the meaning index (one time only, takes several minutes)...")
            self.emb = self.embed_fn(self.texts, "search_document: ")
            np.save(EMB_FILE, self.emb)
        assert len(self.emb) == len(self.df), "index does not match data - delete data/train_embeddings.npy"

    def search(self, diagnosis, treatment, request, k=5, exclude_id=None):
        query = case_text(diagnosis, treatment, request)
        kw_scores = self.bm25.get_scores(tokenize(query))
        q = self.embed_fn([query], "search_query: ")[0]
        sem_scores = self.emb @ q
        kw_rank = np.argsort(-kw_scores)[:100]
        sem_rank = np.argsort(-sem_scores)[:100]
        fused = {}
        for rank, idx in enumerate(kw_rank):
            fused[idx] = fused.get(idx, 0) + 1 / (60 + rank)
        for rank, idx in enumerate(sem_rank):
            fused[idx] = fused.get(idx, 0) + 1 / (60 + rank)
        results = []
        for idx, score in sorted(fused.items(), key=lambda x: -x[1]):
            row = self.df.iloc[idx]
            if exclude_id is not None and row["case_id"] == exclude_id:
                continue
            results.append({
                "case_id": row["case_id"],
                "outcome": "overturned" if row["label"] == 1 else "upheld",
                "treatment": row["treatment"],
                "request": row["request"][:300],
                "reasoning": row["reasoning"][:600],
                "score": round(score, 4),
            })
            if len(results) == k:
                break
        return results


if __name__ == "__main__":
    s = CaseSearch()
    test = pd.read_csv("data/cases_test.csv")
    row = test.iloc[0]
    print("\nNEW REQUEST:", row["request"][:250])
    print("TRUE OUTCOME (hidden from the model):", "overturned" if row["label"] == 1 else "upheld", "\n")
    for r in s.search(row["diagnosis"], row["treatment"], row["request"], k=3):
        print(f"- [{r['outcome'].upper()}] {r['treatment']}: {r['request'][:150]}")