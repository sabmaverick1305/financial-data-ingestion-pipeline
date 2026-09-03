# AGENTS.md

Guidance for AI coding agents (Claude Code, Cursor, Aider, Codex CLI, etc.) working in this repository.

## What this is

FIES (Financial Intelligence Evaluation Suite) ingests Indian mutual fund data from three
independent sources — AMFI (PDF/XLS research docs + daily NAV files), mfapi.in (per-scheme
NAV/master records), and SEBI (per-AMC Scheme Information Documents) — reconciles them through
an entity-resolution layer, and answers questions over the result via a LangGraph-routed
question-answering pipeline that picks between RAG retrieval, Text-to-SQL, fund-performance SQL,
and a causal reasoning engine depending on the query. See [SPEC.md](SPEC.md) for the full
architecture writeup.

## Commands

```bash
# setup
python -m venv .venv && source .venv/bin/activate
make install-dev            # pip install -e ".[dev]"
cp .env.example .env

# day to day
make lint                   # ruff check + ruff format --check (src, tests)
make format                 # ruff format + ruff check --fix
make type-check              # mypy src (strict mode)
make test                   # pytest (testpaths=tests, asyncio_mode=auto)
make test-cov                # pytest --cov-report=html

# single test
pytest tests/unit/test_transformer.py
pytest tests/unit/test_transformer.py::test_specific_case -v

# run the CLI / pipeline
make run ARGS="..."          # python -m financial_pipeline.cli $(ARGS)
python scripts/serve.py [--reload] [--workers 2]     # FastAPI API on :8080
python scripts/search.py "query text" [--mode semantic|keyword|hybrid]

# ingestion workers (run individually, each --loop for continuous polling)
python scripts/fetch_amfi_research_files.py
python scripts/process_text_worker.py --loop
python scripts/process_table_worker.py --loop
python scripts/process_chunk_worker.py --loop
python scripts/process_embed_worker.py --loop
python scripts/run_parallel_orchestrator.py [--dry-run]
python scripts/reset_stale_claims.py [--report|--dry-run]

# eval suite (see .claude/skills/run-eval-suite/SKILL.md)
python eval/run_eval.py --phase all       # or intent|sql|retrieval|answer|guardrail|data_quality

# infra (wraps infra/deploy.sh)
make deploy-infra / destroy-infra / invoke-lambda / infra-status
```

Package name is `financial_pipeline` (src-layout under `src/`), installed via `pip install -e .`.
Requires Python >=3.11. Ruff config: line-length 130, rules `E,F,I,UP`, `E501` ignored. Mypy:
`strict = true` on `src`.

## Architecture (big picture)

**Two independent data paths converge in Postgres, then a single LangGraph answers questions
over both.**

1. **Unstructured path (PDF/XLS → RAG):** `scripts/fetch_amfi_research_files.py` scrapes AMFI,
   writes to S3 (`bronze/` medallion layout, see `docs/s3-data-layout.md`), upserts
   `document_metadata` as a work queue. Five ECS-worker scripts advance
   `document_metadata.processing_status` through
   `uploaded → text_extracted → tables_extracted → processed → embedded`, claimed via
   `SELECT ... FOR UPDATE SKIP LOCKED` + a `claim_expires_at` TTL (recovered by
   `scripts/reset_stale_claims.py`). Final embeddings land in `document_chunks`
   (`pgvector`, 384-dim, HNSW index).

2. **Structured path (mfapi.in → SQL):** `mf_ingestion/sync.py::run_sync()` pulls scheme
   master + NAV history into `mf_scheme_master`/`mf_nav_history`/`mf_scheme_sync_status`;
   `mf_performance/calculator.py` derives `mf_scheme_performance` (returns, CAGR, volatility).
   Every synced scheme also feeds entity resolution automatically via
   `services/entity_ingestion.py::ingest_scheme_plan()`.

3. **Entity resolution:** the three sources name the same AMC/scheme/category differently.
   Rules live declaratively in `domain/resolution/*.yaml`; `services/entity_resolver.py`,
   `services/canonical_name_normalizer.py`, `services/entity_store.py` (idempotent upserts) and
   `services/entity_ingestion.py` implement them into
   `financial_entity_master`/`_identifier`/`_relationship`. `services/lineage.py` wraps every
   write as a mandatory gateway that always records lineage, never raising on lineage failure.

4. **Semantic/reasoning layer (`domain/semantic/`):** a strict dependency chain —
   `vocabulary.yaml` → `taxonomy.yaml` → `thesaurus.yaml` → `financial_ontology.yaml` →
   `financial_relationships.yaml` → `reasoning_rules.yaml` — loaded and cross-validated by
   `src/financial_pipeline/semantic/semantic_engine.py::get_engine()`. Downstream layers may
   only reference ids declared upstream; violations fail at load time, not query time. Consumed
   by `services/ontology_resolver.py`, `retrieval/ontology_expansion.py`,
   `retrieval/ontology_reranker.py`, `reasoning/reasoning_engine.py`, and
   `fies/generator/query_generator.py` (the eval-corpus generator, distinct from `eval/` which
   *runs* evals).

5. **Query-time routing (`graph/graph.py::build_graph()`):** a LangGraph `StateGraph` whose
   entry node classifies the query and branches to one of: `query_sql` (Vanna.ai text-to-SQL
   against `amfi_fund_stats`/`amfi_amc_stats`), `fund_performance_sql` (per-scheme NAV/return
   queries against `mf_scheme_master`/`mf_nav_history`/`mf_scheme_performance` — a different
   dataset from `query_sql`), `plan_years` (analytical year-range aggregation), `reasoning`
   (causal "why" queries answered deterministically from `reasoning_rules.yaml`, skipping the
   LLM), or the default RAG `route`. RAG fans out via `Send` to
   `retrieve_dense`/`retrieve_sparse`/`retrieve_table`/`retrieve_metadata`, converges through
   `rrf_fusion` → `rerank` → `context_optimizer` → `grade_context` (CRAG retry loop back to
   `route` if weak), then all paths converge at `augment` → `pre_guardrail` → `generate` →
   `post_guardrail` → (`repair` loop or) `format_response`. State is persisted per `thread_id`
   via a Postgres-backed checkpointer (`storage/checkpointer.py`).

6. **Text-to-SQL policy layer:** `text_to_sql/vanna_agent.py::_validate_and_prepare()` gates all
   generated SQL — SELECT-only, table allowlist, auto-`LIMIT`, date-filter warnings — before
   execution with a 30s Postgres timeout. Vanna is trained via `scripts/train_vanna.py --reset`
   against a ChromaDB store persisted at `.vanna_chromadb/`.

7. **Eval harness (`eval/`):** six phases — `intent`, `sql`, `retrieval`, `answer`, `guardrail`,
   `data_quality` — each with a runner (`eval/runners/`), metrics module (`eval/metrics/`), and
   fixed ground truth (`eval/corpus/`). Results optionally upload to LangSmith. Note
   `src/financial_pipeline/evaluation/` is a second, older/separate evaluation module (cost
   tracking, observability) — don't confuse it with top-level `eval/`.

8. **Two deployable images:** `Dockerfile` (ingestion/processing workers — one entrypoint,
   ECS task `command` selects which `scripts/process_*_worker.py` runs) and `Dockerfile.api`
   (FastAPI server, `requirements-api.txt` only — excludes ingestion-only deps like docling/
   pymupdf, pins CPU-only torch, bakes in both the embedding and reranker models at build time).

## Gotchas worth knowing before touching related code

- `chunker.py::chunk_text()` previously had an infinite-loop OOM bug (see
  `docs/postmortem-oom-infinite-loop.md`): when `end == text_len` and `end - overlap < start`,
  `start` failed to advance. Any change to chunking's start/end/overlap arithmetic needs the
  parameterized edge-case suite re-run.
- `config.py`'s `OPENAI_API_KEY` env var actually holds the **Anthropic** key (auto-detected via
  `sk-ant-` prefix); `OPENAI_MINI_API_KEY` holds a real OpenAI key used only for intent/date
  extraction and fails closed to rule-based paths if absent. Don't assume the var name matches
  the provider.
- `query_sql` (Vanna, `amfi_fund_stats`/`amfi_amc_stats`) and `fund_performance_sql`
  (`mf_scheme_performance` family) are separate SQL paths over separate tables — don't conflate
  them when routing or debugging a SQL answer.
- `config/` at the repo root is an intentionally empty placeholder — not wired to anything yet.
- Secrets (`POSTGRES_URL`, `OPENAI_API_KEY`, `OPENAI_MINI_API_KEY`, `LANGSMITH_API_KEY`, etc.)
  can come from AWS Secrets Manager instead of `.env`/plaintext env vars — set
  `SECRETS_MANAGER_SECRET_ID` to a secret whose JSON keys match `Settings` field names.
  See `config.py`'s `SecretsManagerSource`. Real env vars still take precedence, so this is
  additive, not a breaking change to local `.env` dev. Separately, all 6 ECS task definitions
  (`amfi-{text,table,ocr,chunk,embed}-worker`, `amfi-mf-nav-sync-worker`) now inject
  `POSTGRES_URL` via ECS-native `Secrets: ValueFrom` from the shared `fies/postgres-url` secret
  rather than a plaintext task-def `Environment` value — no app code involved for that path.
- `infra/lambda/amfi_connector/` is misleadingly named: per `docs/s3-data-layout.md` it runs as
  an ECS Fargate task, not an actual AWS Lambda.
