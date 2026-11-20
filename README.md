# ClaimTrace

ClaimTrace estimates whether a California health insurance denial will be overturned or upheld on independent medical review. It reads a short request, finds similar past appeals, and gives an estimate. If the evidence is mixed, it says "not sure" and sends the case to a human reviewer instead of guessing.

Everything runs locally with free tools (Ollama, Python). No paid services.

Status: finished, apart from one item listed under "Not done". Final report is available from the author.

## How it works

1. Data: public California DMHC Independent Medical Review cases (42,749 cases, about 22,957 usable after cleaning).
2. Leakage fix: each case is split into the short request (the only thing the system sees) and the reviewer's reasoning (kept apart and used only as a library of other past cases). 1,500 cases are held out as a final test and are never used for searching or training.
3. Search: keyword search (BM25) plus meaning search (nomic-embed-text embeddings), merged with reciprocal rank fusion.
4. Estimator: TF-IDF plus logistic regression gives a probability. Confidence is abs(p - 0.5) * 2. The cutoff (0.4) is the lowest confidence that reaches 80 percent on a validation slice, not the test set.
5. Harness: my own agent loop around a local model (Ollama). It handles tool calls, trims results, stops after a turn limit, checks the answer format, and writes a decision log.
6. Decision rule: the system decides and the model only explains. Below the cutoff the case goes to a human. If the model disagrees with the estimator, the estimator wins. If the model fails, the estimator answers alone.

## Results (1,500 held-out test cases)

Always guessing the common answer: about 51 percent.

| Confidence cutoff | Share answered | Correct on those |
| --- | --- | --- |
| 0.0 | 100% | 68.1% |
| 0.2 | 74% | 74.1% |
| 0.3 | 61% | 77.5% |
| 0.4 (chosen) | 49% | 81.4% |
| 0.5 | 36% | 84.0% |
| 0.6 | 26% | 88.1% |

Model comparison on the same 40 random test cases (accuracy is the same because the estimator decides):

| | Llama 3.2 3B | Qwen 2.5 7B |
| --- | --- | --- |
| Usable explanation | 19 of 23 | 23 of 23 |
| Overridden by the estimator | 7 | 4 |
| Time for 40 cases | 25 min | 76 min |

Security: 20 hidden-instruction attack texts, each tried in the request, in a retrieved case, and against an "unsure" estimator. The decision never changed. Note that the model is faked in these tests, so they show the decision rule is safe, not that the real model ignores hidden instructions in its notes.

96 automated tests pass.

## Run it

Needs Python and Ollama with llama3.2:3b, qwen2.5:7b and nomic-embed-text pulled. Download the DMHC data into the data folder first. Start Ollama before running anything that uses a model.

```
python src\prepare_data.py
python src\classifier.py
python src\run_demo.py 5
python src\compare_models.py llama3.2:3b 40
python src\compare_models.py qwen2.5:7b 40
python -m pytest
```

## Files

- src/prepare_data.py: cleaning, request/reasoning split, held-out test set
- src/search.py: hybrid case search
- src/classifier.py: estimator and confidence cutoff
- src/harness.py: agent loop and decision rule
- src/injection_filter.py: flags text that looks like a hidden instruction
- src/run_demo.py: runs the agent on a few held-out cases
- src/compare_models.py: compares local models on the same cases
- tests/: automated tests, including tests/test_security.py

## Not done

A rules lookup tool from the original plan was not built, because I could not verify a rules source. The proposal also listed LangGraph, MCP and reranking; I wrote my own harness instead and did not add those.