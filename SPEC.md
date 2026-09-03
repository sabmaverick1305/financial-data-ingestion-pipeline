# SPEC.md — FIES Architecture & Technical Specification

Technical reference for the Financial Intelligence Evaluation Suite (FIES): what the system is
built from and how data moves through it, as implemented today. For commands and quick
orientation see [AGENTS.md](AGENTS.md).

## 1. Scope

FIES ingests Indian mutual fund data from three independent, non-overlapping sources, reconciles
their conflicting naming/identity schemes into one canonical entity graph, and answers natural-
language questions over the result by routing each query to whichever backend can answer it best:
unstructured RAG retrieval, structured Text-to-SQL, fund-performance SQL, or a causal reasoning
engine.

**Sources:**
- **AMFI India** — 447 PDF/XLS research documents (2009–2025) plus a daily `NAVAll.txt` bulk file
- **mfapi.in** — 37,600+ per-scheme NAV/master records
- **SEBI** — per-AMC Scheme Information Documents (SIDs)

## 2. Storage layout

### 2.1 S3 (medallion architecture — `docs/s3-data-layout.md`)

```
bronze/amfi/{monthly_aum,quarterly_aum,other}/{date}/{filename}   # raw PDFs/XLS
bronze/amfi/nav/{date}/NAVAll.txt                                  # daily NAV bulk file
bronze/sebi/sid/{amc_entity_id}/{filename}
silver/amfi/{monthly,quarterly,other}/.../{text,tables,chunks}.json
gold/                                                              # reserved, not implemented
```

`gold` is a placeholder — derived/aggregated data lives directly in Postgres today
(`amfi_fund_stats`, `mf_scheme_performance`, `document_chunks`). Old pre-medallion S3 paths are
left in place, untouched, alongside the new layout.

### 2.2 Postgres (RDS PostgreSQL 16 + pgvector 0.8)

Work-queue / document tables (`infra/sql/schema.sql`):
- `document_metadata` — one row per source document; `processing_status` drives the worker
  pipeline, `claim_expires_at` implements a claim TTL for crash recovery.
- `document_processing_log` — append-only processing history.
- `document_chunks` — `vector(384)` column (HNSW index) holding chunk embeddings.

Structured mutual-fund tables:
- `mf_scheme_master`, `mf_nav_history`, `mf_scheme_sync_status` — from the mfapi.in sync.
- `mf_scheme_performance` — derived returns/CAGR/volatility/52-week hi-lo.
- `amfi_fund_stats`, `amfi_amc_stats` — AMFI-derived aggregates (separate dataset from the
  `mf_scheme_*` family; the two are queried by different SQL routes, see §5).

Entity-resolution tables:
- `financial_entity_master`, `financial_entity_identifier`, `financial_entity_relationship`.

## 3. Ingestion pipeline

### 3.1 Unstructured (PDF/XLS)

`scripts/fetch_amfi_research_files.py` scrapes amfiindia.com, classifies each document
(`ingestion/page_scraper.py::classify_filename()`), uploads to S3, and upserts
`document_metadata`.

Five ECS Fargate workers advance each document through
`uploaded → text_extracted → tables_extracted → processed → embedded`, each claiming rows via
`SELECT ... FOR UPDATE SKIP LOCKED` with a 30-minute claim TTL (stale claims recovered by
`scripts/reset_stale_claims.py`):

| Worker | Resources | Tool | Script |
|---|---|---|---|
| amfi-text-worker | 1 vCPU / 2 GB | PyMuPDF | `scripts/process_text_worker.py` |
| amfi-table-worker | 4 vCPU / 16 GB | Docling | `scripts/process_table_worker.py` |
| amfi-ocr-worker | 4 vCPU / 16 GB | Docling + OCR | `scripts/process_ocr_worker.py` |
| amfi-chunk-worker | 1 vCPU / 2 GB | sliding-window chunker | `scripts/process_chunk_worker.py` |
| amfi-embed-worker | 1 vCPU / 2 GB | `all-MiniLM-L6-v2` → pgvector | `scripts/process_embed_worker.py` |

`scripts/run_parallel_orchestrator.py` drives local/dry-run execution of the chain.

**Known failure mode:** `processing/chunker.py::chunk_text()` had a 3-week production OOM
incident (`docs/postmortem-oom-infinite-loop.md`) — when `end == text_len` and
`end - overlap < start`, `start` failed to advance, causing an infinite re-append loop regardless
of container memory. Fixed by breaking before recomputing `start` once `end >= text_len`;
verified against a 12-case parameterized edge-case suite. Any change to the chunker's
start/end/overlap arithmetic must re-run that suite.

### 3.2 Structured (mfapi.in)

`mf_ingestion/sync.py::run_sync()` pulls scheme master + NAV history into
`mf_scheme_master`/`mf_nav_history`/`mf_scheme_sync_status`. `mf_performance/calculator.py` +
`run.py` derive `mf_scheme_performance`. Every synced scheme also triggers
`services/entity_ingestion.py::ingest_scheme_plan()`, feeding entity resolution automatically —
this path has no separate "backfill" step; it runs on every sync.

### 3.3 SEBI

`sebi_ingestion/sync.py` + `sebi_ingestion/amc_sources.py` — scaffold-level as of this writing.

## 4. Entity resolution

The three sources name the same AMC/scheme/category differently. Rules are declared, not coded:

- `domain/resolution/entity_resolution_rules.yaml` — matching rules
- `domain/resolution/matching_thresholds.yaml` — similarity thresholds

Implementation in `src/financial_pipeline/services/`:
- `canonical_name_normalizer.py` — name normalization
- `entity_resolver.py` — matching logic
- `amfi_category_source.py` — AMFI category identity source
- `entity_store.py` — idempotent upserts (`ON CONFLICT DO NOTHING`) into
  `financial_entity_master`/`_identifier`/`_relationship`
- `entity_ingestion.py` — ingestion-time entry point
- `entity_reconciliation.py` / `relationship_service.py` — graph reconciliation and relationship
  reads
- `ontology_resolver.py` — bridges entity resolution to the semantic/reasoning layer (§5)
- `lineage.py` — a **mandatory gateway**: every entity write is wrapped so a lineage row is
  always recorded (stage `entity_resolution`, `bronze_to_silver`/`silver_to_gold`, etc.); lineage
  write failures never raise, so a lineage outage cannot block ingestion.

## 5. Semantic / reasoning engine

`domain/` is a pure declarative YAML knowledge base (no code), split into three areas:

- **`domain/semantic/`** — a strict dependency chain, each layer only allowed to reference ids
  declared upstream:
  `vocabulary.yaml` (concepts) → `taxonomy.yaml` (scheme_type/AMC/category hierarchy) →
  `thesaurus.yaml` (synonyms) → `financial_ontology.yaml` (formal concept definitions) →
  `financial_relationships.yaml` (causal relations) → `reasoning_rules.yaml` (IF/THEN causal
  inference rules)
- **`domain/entity_model/`** — `entity_types.yaml`, `relationship_types.yaml`,
  `identifier_types.yaml`, `lifecycle_statuses.yaml`, `canonical_naming_rules.yaml`,
  `source_trust.yaml`
- **`domain/resolution/`** — see §4

`src/financial_pipeline/semantic/semantic_engine.py::get_engine()` loads and cross-validates the
whole stack in one pass, failing loudly at load time (not query time) if a downstream layer
references an id the upstream layers never declared. Consumers: `services/ontology_resolver.py`,
`retrieval/ontology_expansion.py`, `retrieval/ontology_reranker.py`,
`reasoning/reasoning_engine.py`, `fies/generator/query_generator.py`.

`fies/` (distinct from top-level `eval/`) is the **eval-corpus generator**:
`fies/generator/query_generator.py` compiles evaluation questions from the ontology;
`fies/ontology/{capabilities,templates,execution_labels}.yaml` define query capability templates
and expected-execution labels (some marked `status: unimplemented`, e.g. `T_REASON_001`). The
older `fies/ontology/entities.yaml`/`metrics.yaml` are retired in favor of `domain/semantic/`.

## 6. Query-time routing (LangGraph)

`src/financial_pipeline/graph/graph.py::build_graph(factory, analytical, sql, fund_performance,
reasoning, checkpointer)` builds a `StateGraph` (state: `graph/state.py::RAGState`).

Entry node `analyze_query` branches (`is_range_query` and query classification) to:

| Branch | Backend | Notes |
|---|---|---|
| `query_sql` | Vanna.ai text-to-SQL | `amfi_fund_stats`/`amfi_amc_stats` |
| `fund_performance_sql` | direct SQL | `mf_scheme_master`/`mf_nav_history`/`mf_scheme_performance` — a different dataset from `query_sql`, do not conflate |
| `plan_years` | analytical agent | year-range aggregation |
| `reasoning` | `reasoning/reasoning_engine.py` | causal "why" queries; sets `structured_answer` deterministically from `reasoning_rules.yaml`, **skips the LLM entirely** |
| `route` (default) | RAG | 5-stage hybrid retrieval, see below |

**RAG path:** `route` fans out via LangGraph `Send` to `retrieve_dense` / `retrieve_sparse` /
`retrieve_table` / `retrieve_metadata` in parallel, converging at:

1. `rrf_fusion` — reciprocal rank fusion of dense + sparse candidates (~200 → 75)
2. `rerank` — cross-encoder (`cross-encoder/ms-marco-MiniLM-L-12-v2`) rerank (75 → top 20)
3. `context_optimizer` — dedup/diversity optimization (top 20 → top 10)
4. `grade_context` — CRAG-style grading; weak context loops back to `route` via `rewrite_query`

**All paths converge** at `augment` → `pre_guardrail` → `generate` → `post_guardrail` →
(`repair` bounded retry loop, or) → `format_response` → `END`. This means `query_sql`,
`fund_performance_sql`, and `reasoning` all reuse the same guardrail/citation machinery as RAG
answers, even though they may skip the LLM (`reasoning`) or the retrieval stages entirely.

State is persisted per `thread_id` via `storage/checkpointer.py::build_checkpointer()`, a
Postgres-backed LangGraph `BaseCheckpointSaver`.

## 7. Guardrails

`augmentation/guardrails.py`:
- **Pre-generation:** investment-advice block, unsupported-claims check, source-requirement check
- **Post-generation:** citation validation, hallucination detection, answer safety, numeric
  consistency

## 8. Text-to-SQL

`text_to_sql/vanna_agent.py`:
- `build_vanna_agent()` wires `vanna.anthropic.Anthropic_Chat` + `vanna.chromadb.ChromaDB_VectorStore`
  (persisted under `.vanna_chromadb/` by default)
- Trained via `scripts/train_vanna.py --reset` (DDL + docs + question→SQL examples)
- `ask()` generates SQL, then `_validate_and_prepare()` gates it: SELECT-only, table allowlist,
  auto-`LIMIT`, date-filter warnings — before execution with a 30-second Postgres statement
  timeout
- A deterministic corrective retry re-prompts on validation failure before giving up

## 9. Retrieval internals (`retrieval/`)

- `query_understanding.py::QueryAnalyzer` — no-LLM intent detection (rule-based)
- `hybrid.py::HybridRetriever` / `ContextOptimizer` — dense+sparse retrieval and post-rerank
  optimization
- `ontology_expansion.py` / `ontology_reranker.py` — semantic-layer-aware query expansion and
  reranking
- `pipeline.py::RetrievalPipeline`, `rag.py::RAGPipeline` — orchestration

Embeddings: `all-MiniLM-L6-v2` (384-dim, CPU). Reranker:
`cross-encoder/ms-marco-MiniLM-L-12-v2` (bumped from `L-6-v2` in the most recent architecture
change).

## 10. API server

`src/financial_pipeline/api/main.py` — FastAPI app, lifespan-managed singleton state
(`DocumentRepository`, `Retriever`, `RetrievalPipeline`, `RAGPipeline`, compiled LangGraph,
checkpointer, `QueryLogRepository`).

Endpoints: `GET /healthz`, `GET /readyz`, `POST /api/search`, `POST /api/ask`,
`POST /api/feedback`, `GET /api/documents`, `GET /api/stats`. Swagger at `/docs`.

Prometheus metrics: `amfi_query_latency_ms`, `amfi_queries_total`, `amfi_intent_stage_total`,
`amfi_llm_tokens_total`, `amfi_guardrail_blocks_total`, `amfi_analytical_queries_total`.

Runtime env vars (documented in `Dockerfile.api`): `POSTGRES_URL`, `OPENAI_API_KEY` (actually
holds the **Anthropic** key — auto-detected via `sk-ant-` prefix in `config.py`),
`OPENAI_MINI_API_KEY` (real OpenAI key, used for intent/date extraction only; fails closed to
rule-based paths if absent), `LANGSMITH_API_KEY`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT`,
optional AWS creds for CloudWatch.

**Secrets source:** all of the above can be provided as plain env vars (`.env` locally, container
env vars in deploy) or via AWS Secrets Manager, in two complementary ways:

- **App-level (`config.py::SecretsManagerSource`):** set `SECRETS_MANAGER_SECRET_ID` to a JSON
  secret whose keys match `Settings` field names (`postgres_url`, `openai_api_key`, etc.). Merged
  in below real env vars but above `.env`, so an explicit env var always wins and local dev is
  unaffected unless `SECRETS_MANAGER_SECRET_ID` is deliberately set; a fetch failure logs a
  warning and falls through to `.env`/defaults rather than raising.
- **ECS-native (task-definition `Secrets`):** all 6 ECS Fargate task definitions
  (`amfi-{text,table,ocr,chunk,embed}-worker`, `amfi-mf-nav-sync-worker`) inject `POSTGRES_URL`
  via `Secrets: ValueFrom` pointing at the shared `fies/postgres-url` Secrets Manager secret,
  rather than a plaintext container `Environment` value — ECS resolves it into a regular env var
  before the container starts, so no application code is involved. The shared
  `amfi-ecs-execution-role` execution role needs `secretsmanager:GetSecretValue` on that secret's
  ARN to do so; for the `mf-nav-sync` worker this grant is version-controlled as an
  `AWS::IAM::Policy` resource in `infra/cloudformation/mf-nav-sync.yaml` (the same role is reused
  by the other 5 worker families, so one grant covers all of them). The 5 non-`mf-nav-sync`
  worker task definitions have no CloudFormation template in this repo — `deploy.yml`'s
  `update-workers` job re-registers them directly via `aws ecs register-task-definition`,
  patching only the image field and leaving the `Secrets`/`Environment` block untouched, so
  future image deploys keep the Secrets Manager wiring automatically.

## 11. Evaluation harness (`eval/`)

Six phases, each with a runner (`eval/runners/`), metrics module (`eval/metrics/`), and fixed
ground truth (`eval/corpus/`):

| Phase | Runner | Checks |
|---|---|---|
| intent | `run_intent_eval.py` | intent classification accuracy |
| sql | `run_sql_eval.py` | generated SQL correctness |
| retrieval | `run_retrieval_eval.py` | retrieved-chunk relevance |
| answer | `run_answer_eval.py` | answer quality (incl. RAGAS metrics) |
| guardrail | `run_guardrail_eval.py` | guardrail trigger correctness |
| data_quality | `run_dataquality_eval.py` | e.g. `DQ005`/`DQ006` — entity-resolution/relationship completeness (`_check_schemes_missing_belongs_to`, `_check_relationship_completeness`) |

Driven by `eval/run_eval.py --phase all|<phase> [--ids Q001 Q002] [--out file.json]`, with
optional LangSmith upload. `eval/results/` accumulates historical run outputs — treat it as an
experiment log, not curated documentation.

`src/financial_pipeline/evaluation/` is a **separate, older** module (cost tracking,
observability via LangSmith) — not to be confused with top-level `eval/`.

## 12. Deployment

**Compute:** AWS ECS Fargate, ARM64/Graviton3. **Images:**
- `Dockerfile` — ingestion/processing workers, single entrypoint (`ENTRYPOINT ["python"]`); the
  ECS task definition's `command` selects which `scripts/process_*_worker.py` runs.
- `Dockerfile.api` — API server, 3-stage build, `requirements-api.txt` only (excludes docling/
  pymupdf/pdfplumber/pyarrow/openpyxl/xlrd/schedule/click/alembic), CPU-only torch pinned first,
  both embedding and reranker models baked in at build time. Single uvicorn worker (each worker
  loads ~350 MB of ML models; target instance is t3.micro/1 GB RAM).

**CI/CD** (`.github/workflows/`):
- `ci.yml` — lint (`ruff`), format check, inline smoke tests (core imports, intent-detection
  accuracy, guardrail behavior, context-optimizer dedup), plus `hadolint` on `Dockerfile`.
- `deploy.yml` — on push to `main`: build ARM64 image → push to ECR (`amfi-doc-processor`) →
  run a one-shot migration task (`DocumentRepository.create_tables()`) → update 6 ECS task
  defs (`amfi-{text,table,ocr,chunk,embed}-worker`, `amfi-mf-nav-sync-worker`) → smoke-test.
- `rollback.yml` — manual-only, requires the `production` GitHub Environment approval gate;
  repoints 5 worker task defs (excludes `mf-nav-sync-worker`) to a prior image tag.

**Infra as code:** `infra/cloudformation/` (`amfi-pipeline.yaml`, `mf-nav-sync.yaml` — deployed,
scheduled daily; `amfi-nav-fetch.yaml` — template ready, not yet deployed). `infra/deploy.sh` /
`infra/deploy_mf_nav_sync.sh` wrap the Makefile's `deploy-infra`/`destroy-infra`/`invoke-lambda`/
`infra-status` targets. `infra/lambda/amfi_connector/` runs as an ECS Fargate task despite the
directory name.
