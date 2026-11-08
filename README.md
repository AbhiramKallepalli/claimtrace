\# ClaimTrace



ClaimTrace estimates whether a California health insurance denial will be overturned or upheld on independent medical review. It reads a short request, finds similar past appeals, and gives an estimate. If the evidence is mixed, it says "not sure" and sends the case to a human reviewer instead of guessing.



Everything runs locally with free tools (Ollama, Python). No paid services.



Status: in progress. Target finish Nov 20, 2026. Results below are early.



\## How it works



1\. Data: public California DMHC Independent Medical Review cases (about 42,700 cases, about 23,000 usable after cleaning).

2\. Leakage fix: each case is split into the short request (the only thing the system sees) and the reviewer's reasoning (kept apart and used only as a library of other past cases). 1,500 cases are held out as a final test and are never used for searching or training.

3\. Search: keyword search (BM25) plus meaning search (nomic-embed-text embeddings), merged with reciprocal rank fusion.

4\. Estimator: TF-IDF plus logistic regression gives a probability. The confidence cutoff is chosen on a validation set, not the test set.

5\. Harness: my own agent loop around a local model (Llama 3.2 3B through Ollama). It handles the tool calls, trims results, stops after a turn limit, checks the answer format, and writes a decision log.

6\. Decision rule: the system decides and the model only explains. Below the confidence cutoff the case goes to a human.



\## Early results



\- Always guessing the common answer: about 51 percent.

\- Estimator on the 1,500 held-out cases: about 68 percent.

\- When confident, it answers about half of the cases and is right about 81 percent of those. The rest go to a human.

\- 15 automated tests pass.



\## Run it



Needs Python and Ollama with llama3.2:3b and nomic-embed-text pulled. Download the DMHC data into the data folder first.

```

python src\\prepare\_data.py

python src\\classifier.py

python src\\run\_demo.py 5

python -m pytest

```

\## Files



\- src/prepare\_data.py: cleaning, request/reasoning split, held-out test set

\- src/search.py: hybrid case search

\- src/classifier.py: estimator and confidence cutoff

\- src/harness.py: agent loop and decision rule

\- src/run\_demo.py: runs the agent on held-out cases

\- tests/: automated tests



\## Next



Compare the 3B model with a 7B model, add hidden-instruction security tests, add an insurance rules lookup, score the held-out cases, and write the final report.

