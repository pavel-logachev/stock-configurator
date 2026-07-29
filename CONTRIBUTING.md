# Contributing

Contributions should make the system easier to inspect, reproduce, or operate safely.

## Before opening a pull request

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
```

Use Python 3.12 or newer. Tests must pass without live distributor, Telegram, or LLM credentials.

## Change discipline

- Keep one semantic change axis per pull request.
- Add or update tests for changed behavior.
- Preserve deterministic validation and the engineering-review boundary.
- Prefer explicit `no_recommendation` over unsupported fallback output.
- Do not add customer data, real prices, inventory snapshots, private prompts, or credentials.
- Do not weaken path confinement or hash binding in evaluation tooling.
- Document new environment variables in `.env.example` without real values.

## Connector changes

Use mocked responses in tests. Confirm rate limits, retry semantics, and error redaction. A live manual smoke check may supplement tests but cannot replace them.

## AI behavior changes

Record the changed prompt or model setting as a versioned stage. Compare a baseline and candidate with the offline evaluator. Do not claim quality improvement from a single anecdotal example.

## Documentation

Separate facts established by code or tests from operational assumptions. Do not publish internal run IDs, private endpoints, vendor payloads, or performance numbers without reproducible public evidence.

By submitting a contribution, you agree that it is licensed under AGPL-3.0-only and that you have the right to provide it under that license.
