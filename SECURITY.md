# Security policy

## Supported version

Security fixes are applied to the latest commit on `main`. The project has not declared a stable production release line.

## Reporting a vulnerability

Use GitHub private vulnerability reporting for this repository. Do not open a public issue for a suspected secret exposure, authentication bypass, remote-code execution path, or leak of business data.

Include:

- affected commit;
- reproduction steps using synthetic data;
- impact and preconditions;
- suggested mitigation, if known.

Do not include real credentials, customer data, distributor payloads, or private endpoints in the report.

## Secrets

The repository must not contain usable API tokens, passwords, private keys, connection strings, Telegram bot tokens, or distributor credentials. Local configuration belongs in `.env`, which is ignored.

If a secret reaches Git history, deleting the current file is not sufficient. Rotate the credential first, then remove it from history and verify the rewritten repository with a secret scanner.

## Deployment responsibility

The supplied Compose file is for localhost development. Public deployment requires an independent review of authentication, TLS, network exposure, backups, observability, rate limiting, data retention, and incident recovery.

## AI-specific boundary

Treat model output as untrusted input. Security reports are welcome for paths that let a model output bypass identifier grounding, evidence checks, deterministic validation, or the mandatory engineering-review state.
