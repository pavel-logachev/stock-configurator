# Offline evaluation

The public repository contains the evaluator, its contracts, synthetic tests, and a single bootstrap case. It does **not** contain distributor payloads, customer requests, production model outputs, prices, or accepted golden cases.

`dataset.json` is intentionally marked `bootstrap`. The evaluator must remain blocked until enough independently reviewed cases are added and all hash-bound evidence is available. A passing synthetic test proves the mechanics of the gate, not the commercial quality of a model-generated configuration.

## What is versioned

- evaluation schema and thresholds;
- synthetic request and matrix references;
- expected result states and atomic criteria;
- code that compares a baseline and a candidate run;
- tests for hashes, path confinement, blind review, regressions, and privacy.

## What stays local

Real requests, inventory matrices, model outputs, annotations, and reports belong under ignored paths:

```text
evaluation/simple_stock/local/
evaluation/simple_stock/reports/
```

Never commit exported business data, credentials, distributor responses, or customer requirements.

## Evaluation command

```bash
python -m app.cli.evaluate_simple_stock_pipeline \
  --dataset evaluation/simple_stock/v1/dataset.json \
  --baseline-run evaluation/simple_stock/local/baseline/run.json \
  --candidate-run evaluation/simple_stock/local/candidate/run.json \
  --blind-review evaluation/simple_stock/local/review.json \
  --output evaluation/simple_stock/reports/latest.json
```

Exit codes:

- `0` — release gates passed;
- `1` — regression or critical error found;
- `2` — blocked because evidence is incomplete or invalid.

The public synthetic dataset should normally produce `2` until a maintainer deliberately builds a valid local evaluation corpus.
