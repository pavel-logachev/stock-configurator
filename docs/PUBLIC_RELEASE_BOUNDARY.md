# Public release boundary

This repository is a clean-room public edition of Stock Configurator. It was assembled from a reviewed source snapshot rather than by publishing the private repository or rewriting its history.

## Included

- application source under `app/`;
- database migrations;
- automated tests except private agent-tooling checks;
- public distributor connector implementations;
- synthetic request and specification examples;
- a synthetic pipeline contract and bootstrap evaluation dataset;
- local Docker configuration;
- public documentation and CI.

## Excluded

- private Git commits, branches, tags, and remotes;
- agent memory, prompts, handoffs, and local operating instructions;
- deployment scripts and server topology;
- environment files, credentials, API tokens, and account identifiers;
- customer names, requests, contact details, and commercial documents;
- distributor responses, stock snapshots, prices, and product exports;
- production model configuration, run identifiers, outputs, and metrics;
- accepted golden cases and human annotations.

## Provenance

The source snapshot was authored under Pavel Logachev's private project history. No third-party source files or previous license notices were present in the exported application allowlist. Runtime dependencies retain their own licenses and are not vendored into this repository.

The public repository starts with a new history so that deleted or unrelated private material cannot be recovered from earlier commits.

## Claims this repository does not make

- It is not a hosted SaaS service or public demo.
- It does not include access to distributor systems.
- It does not certify hardware compatibility.
- It does not prove production quality for any model or prompt.
- It does not replace engineering or commercial review.

## Reproducing the public checks

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
```

Secret scanning and dependency review also run before and after publication. Real-data acceptance remains a private operational process because the required evidence cannot be published safely.
