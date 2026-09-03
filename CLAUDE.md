# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Commands, architecture, and gotchas for this repo live in [AGENTS.md](AGENTS.md) — read that
first; it's the tool-agnostic source of truth kept in sync for any agentic coding tool. This
file only covers behavior specific to Claude Code.

## Project skills

- `.claude/skills/run-eval-suite/SKILL.md` — running and interpreting `eval/run_eval.py` phases.
- `.claude/skills/add-data-source/SKILL.md` — wiring up a new ingestion source (bronze/silver
  paths, ingestion module, entity-resolution hookup).

## Repo-specific notes

- Full architecture/technical reference: [SPEC.md](SPEC.md).
- `eval/results/` is an experimentation log of historical run outputs, not curated
  documentation — don't treat files there as ground truth for current behavior.
- `.env.example` previously contained real-looking AWS key values committed since the initial
  commit; the file has been scrubbed, but the key itself is still in git history and should be
  rotated in IAM if that hasn't happened yet. Treat any future secret-shaped value found in a
  tracked file as a live incident to flag immediately, not a placeholder to reuse.
- Secrets are moving to AWS Secrets Manager (`config.py`'s `SecretsManagerSource`,
  `SECRETS_MANAGER_SECRET_ID`) rather than plaintext `.env`/container env vars — prefer that
  path when adding new sensitive settings.
