# Architecture

## Design rule

The language model may perform semantic work. It does not own system facts.

Code remains responsible for authentication, network limits, data normalization, storage, evidence binding, validation, report rendering, privacy controls, and reproducibility. A model output is an input to the reconciler, not a final commercial decision.

## Runtime path

1. **Ingestion** - an authorized distributor connector retrieves categories, products, prices, and stock.
2. **Normalization** - source-specific payloads become a common catalog model with provenance.
3. **Request extraction** - free-form text becomes a structured `StockSpec`.
4. **Candidate planning** - code identifies relevant categories and bounds the evidence package.
5. **Matrix construction** - candidates are grouped by requested role with stable identifiers and source facts.
6. **Composition** - an OpenAI-compatible model proposes a draft configuration using only the supplied matrix.
7. **Reconciliation** - deterministic checks validate IDs, quantities, prices, role coverage, material claims, and result state.
8. **Persistence and reports** - the service stores the run and renders Markdown or Excel artifacts.
9. **Engineering review** - a person confirms compatibility, scope, and final commercial content.

## Trust boundaries

### External systems

Distributor, LLM, web-evidence, and Telegram endpoints are untrusted network dependencies. Each connector has explicit timeouts and typed error handling. Credentials enter through environment variables only.

### Model output

Model output is untrusted structured data. Pydantic models parse the result; the reconciler rejects unknown products, unsupported prices, missing roles, and material claims that are not grounded in the evidence package.

### Business data

Catalog snapshots, customer requests, generated outputs, and evaluation annotations are business data. The public repository contains synthetic examples only. Local evaluation directories are ignored.

## Failure policy

The system prefers an explicit `no_recommendation` or blocked result over a plausible-looking unsupported configuration. Missing evidence must remain visible in validation errors, warnings, and review notes.

## Evaluation boundary

A release candidate is compared with a baseline through hash-bound bundles. Inputs, prompts, model settings, outputs, matrices, and annotations are referenced by SHA-256. Blind-review receipts bind decisions to specific baseline and candidate artifacts.

The public dataset is intentionally insufficient for production acceptance. Maintainers must build their own authorized local corpus before treating evaluator output as a release decision.

## Deployment shape

The provided Compose stack contains:

- a FastAPI container;
- PostgreSQL on a private Docker network;
- an optional Telegram profile;
- a named database volume.

Only the API is bound to the host, and it is restricted to `127.0.0.1` by default. Production authentication, TLS termination, backups, observability, and network policy are deployment responsibilities and are intentionally outside this source release.
