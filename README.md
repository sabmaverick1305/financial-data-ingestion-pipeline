# Financial Intelligence Evaluation Suite (FIES)

End-to-end pipeline for ingesting, resolving, and querying Indian mutual fund
data across three independent sources — **AMFI India** (447 PDF/XLS research
documents 2009–2025, plus the daily NAVAll.txt bulk file), **mfapi.in**
(37,600+ per-scheme NAV/master records), and **SEBI** (per-AMC Scheme
Information Documents) — through a single LangGraph-routed question-answering
layer that picks between unstructured RAG retrieval and structured
Text-to-SQL depending on the question.

![Architecture](docs/architecture-v2.png)

---

## What it does

```
                              ┌─ AMFI PDFs/XLS ─→ Extract → Chunk → Embed → pgvector
                              │
  Three source systems ──────┼─ AMFI NAVAll.txt ─→ parse → Entity Resolution ─┐
                              │                                                │
                              └─ mfapi.in ─────────→ mf_ingestion sync ────────┤
                                                                                ▼
                                                            financial_entity_master/
                                                            _identifier/_relationship
                                                                                │
  User question ──→ LangGraph Query Router (graph/) ◄──────────────────────────┘
        │
        ├─ factual/regulatory/trend  → Hybrid Retrieval (dense + sparse + rerank) → Augmentation (guardrails → LLM → guardrails)
        ├─ tabular/aggregate         → Text-to-SQL (Vanna + Claude) → policy-checked SQL → Postgres
        └─ causal ("why did X...")   → Reasoning Engine (domain/semantic/reasoning_rules.yaml)
                                                                                │
                                                                                ▼
                                                              Grounded Answer + Citations/SQL
```

**Result:** Ask natural-language questions spanning both worlds — "What SEBI
guidelines affected derivatives trading?" (RAG), "Total AUM of Large Cap
funds in Dec 2024" (Text-to-SQL), or "Why did AUM increase while net inflow
decreased for Large Cap funds between 2020 and 2024?" (reasoning engine) —
and get a grounded answer, correctly routed to the right subsystem, with
citations or the executed SQL shown.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Compute** | AWS ECS Fargate (ARM64 / Graviton3) |
| **Database** | RDS PostgreSQL 16 + pgvector 0.8 |
| **Storage** | Amazon S3 |
| **Container registry** | Amazon ECR |
| **Monitoring** | Amazon CloudWatch |
| **Scheduling** | Amazon EventBridge |
| **Embedding model** | `all-MiniLM-L6-v2` (sentence-transformers, 384 dims, CPU) |
| **Re-ranking** | `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| **LLM** | Anthropic Claude (auto-detected from `sk-ant-` key) or OpenAI |
| **Query routing** | LangGraph `StateGraph` (`graph/`) |
| **Text-to-SQL** | Vanna.ai + Claude Haiku, ChromaDB-persisted training store |
| **PDF extraction** | Docling (CPU-only, table structure), PyMuPDF |
| **Spreadsheets** | pandas + openpyxl + xlrd (magic-byte format detection) |
| **API** | FastAPI + uvicorn |
| **CI/CD** | GitHub Actions |

---

## Pipeline Stages

### Ingestion
```bash
python scripts/fetch_amfi_research_files.py
```
Scrapes amfiindia.com, classifies documents (monthly/quarterly), uploads to S3, upserts metadata into RDS.

S3 objects follow a bronze/silver/gold medallion layout across all sources
(AMFI PDFs, mfapi.in per-scheme data, SEBI SIDs) — see
[docs/s3-data-layout.md](docs/s3-data-layout.md) for the full structure and
each subtree's producer.

### Document Processing (ECS Fargate workers)

| Worker | Memory | What it does |
|---|---|---|
| `amfi-text-worker` | 1 vCPU / 2 GB | PyMuPDF text extraction, font-based text-layer detection |
| `amfi-table-worker` | 4 vCPU / 16 GB | Docling CPU layout + table structure (native-text PDFs) |
| `amfi-ocr-worker` | 4 vCPU / 16 GB | Docling CPU + OCR (scanned PDFs) |
| `amfi-chunk-worker` | 1 vCPU / 2 GB | Sliding-window text chunking → `chunks.json` |
| `amfi-embed-worker` | 1 vCPU / 2 GB | `all-MiniLM-L6-v2` embeddings → pgvector |

All workers use **`SELECT ... FOR UPDATE SKIP LOCKED`** for safe concurrent processing. Claim TTL (30 min) + `record_failure()` prevent stuck documents after worker crashes.

### Status Lifecycle
```
uploaded → text_extracted → tables_extracted → processed → embedded
                                                              ↓
                                                        (queryable)
```

---

## Retrieval Layer

5-stage hybrid pipeline designed to scale from 24K to 15M+ chunks:

```
Stage 0  Query Router     detect intent (factual/trend/tabular/comparison/regulatory)
Stage 1  Dual Retrieval   parallel dense (pgvector HNSW) + sparse (BM25/GIN) → 200 candidates
Stage 2  RRF Fusion       Reciprocal Rank Fusion → 75 unique candidates
Stage 3  Cross-Encoder    ms-marco-MiniLM re-ranks → top 20
Stage 4  Optimizer        dedup · source diversity · recency · table preservation → top 10
```

HNSW tuning by corpus size:
```sql
-- Current (24K chunks):   m=16, ef_construction=64
-- At 15M chunks (500K docs): m=48, ef_construction=200
SET hnsw.ef_search = 100;
```

---

## Augmentation Layer

```
Retrieved chunks
      │
      ▼  PRE-GENERATION GUARDRAILS
      │  1. Investment advice block   ← blocks before LLM call (no tokens spent)
      │  2. Unsupported claims check
      │  3. Source requirement (min N sources per intent)
      │
      ▼  PROMPT BUILDER  (intent-aware: regulatory / factual / trend / comparison)
      │
      ▼  LLM GENERATION  (Claude Haiku 4.5 / GPT-4o-mini, auto-detected)
      │
      ▼  POST-GENERATION GUARDRAILS
         4. Citation validation       [N] markers map to real sources
         5. Hallucination detection   embedding similarity of answer vs chunks
         6. Answer safety             blocks financial advice language
         7. Numeric consistency       numbers in answer verified against chunks
```

---

## Structured Data: mfapi.in + AMFI Stats

A second, independent data path alongside the PDF/RAG pipeline: per-scheme
mutual fund data (NAV history, scheme master, computed performance) from
mfapi.in, plus AMFI's own aggregate monthly stats tables.

```
mfapi.in scheme-master (S3) → mf_ingestion/sync.py → mf_scheme_master
                                       │                mf_nav_history
                                       │                mf_scheme_sync_status
                                       ▼
                          mf_performance/calculator.py → mf_scheme_performance
                             (returns, CAGR, volatility, 52w high/low —
                              pure computation, recomputed wholesale per run)
```

- **`mf_ingestion/sync.py`** — `run_sync()`: reads the latest scheme-master snapshot from S3, fetches each scheme's NAV history from `api.mfapi.in` (thread-pooled, retried on rate-limit/network errors), upserts into Postgres. Scheduled daily via EventBridge → ECS Fargate (`infra/cloudformation/mf-nav-sync.yaml`) — incremental lookback window, idempotent upserts.
- **`mf_performance/`** — derived, not sourced: `calculator.py` computes returns/CAGR/volatility from NAV history; `run.py` recomputes `mf_scheme_performance` wholesale after each sync.
- **`amfi_fund_stats`** (2020–2026) / **`amfi_amc_stats`** (2009–2019) — AMFI's own aggregate monthly report tables (category-level AUM/mobilization/redemption), strictly non-overlapping in real data. Populated via `scripts/populate_amfi_stats.py` / `scripts/backfill_amfi_*_stats_from_s3.py`.

---

## Text-to-SQL (`text_to_sql/`)

Structured questions ("Total AUM of Large Cap Fund in Dec 2024") route to a
[Vanna.ai](https://vanna.ai)-based SQL agent instead of RAG retrieval.

```
Question → Vanna (Claude Haiku + ChromaDB training store: DDL/docs/examples)
              → generated SQL
              → policy layer (SELECT-only · table allowlist · auto LIMIT · date-filter warning)
              → deterministic period_year/table-mismatch check + one corrective retry
              → Postgres execution (30s timeout) → markdown answer
```

- **`vanna_agent.py`** — `build_vanna_agent()` / `ask()`. Every generated query is policy-checked (`_validate_and_prepare`) before execution, including Vanna's own internal "intermediate SQL" disambiguation queries. A response that isn't even an attempted SQL statement (the LLM explaining why a query is out of range) is treated as "no results," not a blocked policy violation.
- **`scripts/train_vanna.py --reset`** — rebuilds the persisted ChromaDB training store (DDL + docs + question→SQL examples). Training examples are a stronger signal than doc prose alone for steering SQL generation — run after any schema or vocabulary change.
- Config note: Vanna's `Anthropic_Chat` and `VannaBase` share one `max_tokens` attribute for two different purposes (LLM response length *and* DDL/doc context budget) — `build_vanna_agent()` sets it explicitly to `14000` so the full schema/doc/example context is never silently truncated.

---

## Entity Resolution (`domain/` + `services/`)

The three source systems (AMFI, mfapi.in, SEBI) each name the same
real-world things — an AMC, a scheme, a category — differently. A
declarative rule layer plus a resolution pipeline unifies them into one
identity graph.

```
domain/semantic/          taxonomy.yaml (canonical AMC/category ids), thesaurus.yaml (synonyms),
domain/entity_model/      entity_types.yaml, relationship_types.yaml, canonical_naming_rules.yaml
domain/resolution/        entity_resolution_rules.yaml, matching_thresholds.yaml
        │  (declarative rules — grounded in real code, not aspirational)
        ▼
services/  canonical_name_normalizer.py   scheme_name → base_fund_name/plan/option; normalize_name()
           entity_resolver.py             resolve_amc_name() / resolve_category() — pattern + synonym matching
           amfi_category_source.py        scheme_code → AMFI's own NAVAll.txt category (highest-priority signal)
           entity_store.py                lookup_entity / get_or_create_entity / add_identifier / add_relationship
           entity_ingestion.py            ingest_scheme_plan() — wires all of the above into one call
        │
        ▼
financial_entity_master (organization/category/scheme/scheme_plan)
financial_entity_identifier (scheme_code, amc_entity_id, category_taxonomy_id — per source_system)
financial_entity_relationship (has_plan, manages, belongs_to)
```

`entity_ingestion.ingest_scheme_plan()` is called for every scheme on every
`mf_ingestion/sync.py` run — new schemes get a canonical entity, identifier
mapping, and relationships automatically, using AMFI's own category
(`amfi_category_source.load_amfi_category_map()`, cached per-process) as the
highest-confidence signal ahead of mfapi's own `category` field, which is
unreliable for a majority of rows.

---

## Evaluation Suite (`eval/`)

```bash
python eval/run_eval.py --phase all   # or: intent | sql | retrieval | answer | guardrail
```

Five phases against a fixed ground-truth corpus (`eval/corpus/`):

| Phase | Measures |
|---|---|
| **intent** | routing/intent-type/metric/scheme-type/year-extraction accuracy |
| **sql** | parse rate, policy-pass rate, table/column accuracy, row-count accuracy |
| **retrieval** | hit@8, precision@8, recall@8, MRR, keyword coverage |
| **answer** | fact coverage, citation presence, hallucination rate, safety compliance |
| **guardrail** | false positive/negative rate, block-rate by error type, layer accuracy |

`fies/` holds the eval corpus generation machinery (`generator/query_generator.py` compiles questions from `domain/semantic/`'s ontology; `ontology/` defines query capability templates and expected-execution labels).

---

## Quickstart

### Prerequisites
- Python 3.11+
- PostgreSQL 16 with pgvector extension
- AWS credentials (S3 + ECS access)

### Setup

```bash
git clone https://github.com/sabmaverick1305/financial-data-ingestion-pipeline.git
cd financial-data-ingestion-pipeline

python -m venv .venv
source .venv/bin/activate
pip install -e .

cp .env.example .env
# Edit .env — add POSTGRES_URL, AWS credentials, OPENAI_API_KEY
```

### Database Setup

```bash
# Enable pgvector (requires superuser on RDS)
psql $POSTGRES_URL -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Run schema migrations
python -c "
from financial_pipeline.config import settings
from financial_pipeline.storage.document_repo import DocumentRepository
DocumentRepository(settings.postgres_url).create_tables()
"
```

### Run the full pipeline locally

```bash
# Stage 1: Ingest documents from AMFI website
python scripts/fetch_amfi_research_files.py

# Stage 2–5: Process all documents (runs on ECS in production)
python scripts/process_text_worker.py --loop
python scripts/process_table_worker.py --loop
python scripts/process_chunk_worker.py --loop
python scripts/process_embed_worker.py --loop
```

### Run the orchestrator (parallel ECS workers)

```bash
# Auto-scales concurrency: text(3) → table(5) → ocr(3) → chunk(3) → embed(3)
python scripts/run_parallel_orchestrator.py

# Dry-run to see what would launch without hitting AWS
python scripts/run_parallel_orchestrator.py --dry-run
```

### Start the Retrieval API

```bash
python scripts/serve.py              # production
python scripts/serve.py --reload     # dev (auto-reload)
python scripts/serve.py --workers 2  # multi-process
```

Swagger UI: **http://localhost:8080/docs**

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/healthz` | Liveness probe |
| `GET` | `/readyz` | Readiness probe (DB ping) |
| `POST` | `/api/search` | Hybrid semantic + keyword search |
| `POST` | `/api/ask` | Q&A — LangGraph-routed to RAG or Text-to-SQL, with citations + guardrails |
| `POST` | `/api/feedback` | Record user feedback on a prior `/api/ask` response |
| `GET` | `/api/documents` | List/filter ingested documents |
| `GET` | `/api/stats` | Pipeline health and queue depths |

### Example: Search
```bash
curl -X POST http://localhost:8080/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What SEBI guidelines affected derivatives trading?",
    "mode": "hybrid",
    "category": "quarterly",
    "limit": 5
  }'
```

### Example: Ask (RAG)
```bash
curl -X POST http://localhost:8080/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the major regulatory changes SEBI made for mutual funds?",
    "category": "quarterly",
    "top_k": 6
  }'
```

Response includes:
- `answer` — grounded answer with `[1]`, `[2]` citation markers
- `sources` — list of cited documents with file name, period, preview
- `guardrail` — dual-layer safety report (`pre_passed`, `post_passed`, `answer_safe`, `hallucination_risk`)

---

## CLI Search

```bash
# Hybrid search (default)
python scripts/search.py "SIP growth trend over the years"

# Semantic only, filtered by period
python scripts/search.py "equity scheme count" --mode semantic --year 2024 --month 3

# Keyword exact match
python scripts/search.py "SBI Bluechip Fund" --mode keyword --category monthly
```

---

## Operational Tools

```bash
# Reset stale claims (zombie workers left docs stuck in processing states)
python scripts/reset_stale_claims.py --report    # show health + failed docs
python scripts/reset_stale_claims.py --dry-run   # preview what would be reset
python scripts/reset_stale_claims.py             # reset expired claims

# Run evaluation dataset (7 ground-truth questions, measures faithfulness + abstention)
python -m financial_pipeline.augmentation.evaluation --limit 3
```

---

## Configuration

All config in `.env`:

```env
# Database
POSTGRES_URL=postgresql+psycopg2://user:pass@host:5432/dbname

# AWS
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
S3_BUCKET=mf-finance-kb

# LLM — auto-detected from key prefix
# sk-ant-... → Anthropic Claude (claude-haiku-4-5-20251001)
# sk-...     → OpenAI (gpt-4o-mini)
OPENAI_API_KEY=sk-ant-...

# Optional: override model
ANTHROPIC_MODEL=claude-sonnet-4-6
OPENAI_MODEL=gpt-4o

# Retrieval API
API_HOST=0.0.0.0
API_PORT=8080
API_TOP_K=8
```

---

## CI/CD (GitHub Actions)

| Workflow | Trigger | Jobs |
|---|---|---|
| **CI** | every push / PR | lint (ruff) · import validation · guardrail unit tests · Dockerfile lint |
| **Build & Deploy** | push to `main` | build ARM64 image → ECR · DB migration · update 5 task definitions · smoke test |
| **Rollback** | manual only | repoint all workers to a previous image tag |

Required GitHub Secrets:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

---

## Project Structure

```
├── scripts/
│   ├── fetch_amfi_research_files.py   # AMFI PDF/XLS ingestion
│   ├── fetch_amfi_nav.py              # AMFI NAVAll.txt fetch → S3 (bronze raw + silver CSV)
│   ├── process_*_worker.py            # pipeline workers (text/table/ocr/chunk/embed)
│   ├── run_parallel_orchestrator.py   # ECS task orchestration
│   ├── reset_stale_claims.py          # ops: claim TTL recovery
│   ├── train_vanna.py                 # rebuild Vanna's ChromaDB training store
│   ├── populate_amfi_stats.py         # amfi_fund_stats / amfi_amc_stats population
│   ├── search.py                      # CLI search interface
│   └── serve.py                       # API server launcher
│
├── src/financial_pipeline/
│   ├── processing/
│   │   ├── extractor.py               # PyMuPDF + Docling extractors
│   │   └── chunker.py                 # sliding-window chunker
│   ├── retrieval/
│   │   ├── hybrid.py                  # 5-stage HybridRetriever
│   │   ├── query_understanding.py     # intent detection (no LLM)
│   │   └── pipeline.py                # RetrievalPipeline orchestrator
│   ├── augmentation/
│   │   ├── guardrails.py              # pre + post generation guardrails
│   │   ├── pipeline.py                # AugmentationPipeline (7 stages)
│   │   ├── prompts.py                 # intent-aware prompt templates
│   │   └── evaluation.py             # QA dataset + metrics
│   ├── graph/                          # LangGraph query router (RAG vs SQL vs reasoning)
│   ├── reasoning/                      # causal "why" queries over domain/semantic/reasoning_rules.yaml
│   ├── text_to_sql/                    # Vanna + Claude SQL agent, policy layer
│   ├── mf_ingestion/                   # mfapi.in scheme/NAV sync
│   ├── mf_performance/                 # derived returns/CAGR/volatility
│   ├── sebi_ingestion/                 # SEBI SID scrape (scaffold-level)
│   ├── semantic/                       # SemanticEngine — façade over domain/ YAML
│   ├── services/                       # entity resolution: normalize/resolve/store/ingest
│   ├── storage/
│   │   └── document_repo.py           # pgvector search + claim TTL
│   └── api/
│       └── main.py                    # FastAPI app
│
├── domain/                             # declarative rules: semantic/, entity_model/, resolution/
├── eval/                               # 5-phase eval suite (intent/sql/retrieval/answer/guardrail)
├── fies/                               # eval corpus generator + query ontology
│
├── infra/
│   ├── sql/schema.sql                  # pgvector + HNSW indexes
│   └── cloudformation/                 # EventBridge Scheduler → ECS Fargate task defs
│       ├── amfi-pipeline.yaml          # PDF processing workers
│       ├── mf-nav-sync.yaml            # daily mfapi.in incremental sync (deployed, scheduled)
│       └── amfi-nav-fetch.yaml         # daily AMFI NAVAll.txt fetch (template ready, not yet deployed)
│
└── .github/workflows/
    ├── ci.yml
    ├── deploy.yml
    └── rollback.yml
```

---

## Key Design Decisions

**PostgreSQL as work queue** — `SELECT ... FOR UPDATE SKIP LOCKED` gives atomic work distribution without Redis or SQS. The `document_metadata` table is the queue.

**Split workers by memory profile** — PyMuPDF (50 MB) and Docling (4–8 GB) run in separate Fargate tasks. This was the fix for 3 weeks of OOM crashes ([read the post-mortem](docs/postmortem-oom-infinite-loop.md)).

**Claim TTL over dead-letter queues** — `claim_expires_at TIMESTAMPTZ` column + EventBridge every 5 min replaces SQS DLQ complexity. A document stuck in a transient state for >30 min is automatically reverted.

**Dual guardrails** — Pre-generation guardrails block investment advice queries *before* the LLM is called (zero tokens spent). Post-generation guardrails catch hallucinated numbers and unsafe financial language in the output.

**Schema versioning** — All S3 artifacts written to `processed/v1/amfi/...`. Consumers can distinguish schemas; `schema_version` column in RDS tracks which version each document used.

**Idempotent entity resolution, not batch-only** — `services/entity_store.py`'s `get_or_create_entity`/`add_identifier`/`add_relationship` are all `ON CONFLICT DO NOTHING` against real unique constraints, so `entity_ingestion.ingest_scheme_plan()` is safe to call once per scheme on every sync run (one psycopg2 connection per worker thread, not per scheme — see `mf_ingestion/sync.py`), not just as a one-off bulk backfill.

**Real evaluation data, not fixtures** — `eval/`'s ground truth is checked against a live Postgres instance (`eval/corpus/expected_results.json`'s `row_count`/`row_count_min`/`row_count_max`), so metric regressions get root-caused against real data boundaries (e.g. `amfi_fund_stats` covers 2020–2026 only, `amfi_amc_stats` 2009–2019 only, zero overlap) rather than assumed to be LLM sampling noise.

---

## License

MIT
