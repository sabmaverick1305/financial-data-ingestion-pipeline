# AMFI Financial Document Pipeline

End-to-end pipeline for ingesting, processing, indexing, and querying **447 AMFI India financial documents** (2009–2025) — monthly mutual fund repository reports, quarterly journals, and spreadsheets — into a production-grade RAG knowledge base.

![Architecture](docs/architecture-v2.png)

---

## What it does

```
AMFI India → Ingest → Extract → Chunk → Embed → pgvector
                                                     ↓
                                     Hybrid Retrieval (dense + sparse + rerank)
                                                     ↓
                                     Augmentation (pre-guardrails → LLM → post-guardrails)
                                                     ↓
                                             Grounded Answer + Citations
```

**Result:** Ask natural-language questions about Indian mutual fund data — AUM trends, SEBI regulatory changes, scheme counts, fund lists — and receive grounded answers citing specific AMFI documents with numbered references.

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
| **Re-ranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **LLM** | Anthropic Claude (auto-detected from `sk-ant-` key) or OpenAI |
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
| `POST` | `/api/ask` | RAG Q&A with citations + dual guardrails |
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
│   ├── fetch_amfi_research_files.py   # ingestion
│   ├── process_*_worker.py            # pipeline workers (text/table/ocr/chunk/embed)
│   ├── run_parallel_orchestrator.py   # ECS task orchestration
│   ├── reset_stale_claims.py          # ops: claim TTL recovery
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
│   ├── storage/
│   │   └── document_repo.py           # pgvector search + claim TTL
│   └── api/
│       └── main.py                    # FastAPI app
│
├── infra/
│   └── sql/schema.sql                 # pgvector + HNSW indexes
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

---

## License

MIT
