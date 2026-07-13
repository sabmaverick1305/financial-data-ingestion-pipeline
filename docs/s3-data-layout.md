# S3 data layout — bronze / silver / gold

This bucket (`mf-finance-kb`) follows a medallion architecture: each object
lives in exactly one of three layers depending on how processed it is.

```
bronze/   raw, as-fetched from the source — no parsing, no cleaning
silver/   cleaned, extracted, validated — still one row/file per source
          document, but structured (chunks.json, text.json, parsed CSVs)
gold/     curated, aggregated, consumption-ready — reserved, not yet used
          (see "gold/ (reserved)" below)
```

Prefer copying data forward through the layers over mutating in place —
bronze/ objects should never be overwritten by a downstream processing step;
if a re-process is needed, write a new silver/ object and let the two layers
diverge if that's what actually happened (e.g. a re-extraction with a fixed
parser).

## bronze/

```
bronze/
  amfi/
    scheme_master/{YYYY-MM-DD}/mf_scheme_master.json
      — scheme_code/scheme_name list. Read by mf_ingestion (see below);
        currently nothing in this codebase WRITES this file — it was
        uploaded manually. If that changes, update
        settings.mf_scheme_master_s3_prefix's producer accordingly.
    monthly_aum/{YYYY-MM-DD}/{filename}
    quarterly_aum/{YYYY-MM-DD}/{filename}
    other/{YYYY-MM-DD}/{filename}
      — raw PDF/XLS downloads from amfiindia.com's research page, classified
        by classify_filename() (page_scraper.py). Written by
        scripts/fetch_amfi_research_files.py.
    nav/{YYYY-MM-DD}/NAVAll.txt
      — AMFI's own daily consolidated per-scheme NAV file (distinct from
        mfapi.in's per-scheme API below). Written by the amfi-nav-connector
        ECS task (infra/lambda/amfi_connector/handler.py — despite the
        "lambda" directory name, it currently runs as an ECS Fargate task,
        not an actual Lambda function).
  mfapi/
    scheme_master/   — reserved, currently empty. mf_ingestion actually
                        reads its scheme list from bronze/amfi/scheme_master/
                        (see note above); this subtree is here in case a
                        genuinely mfapi.in-native scheme export is added later.
    nav/             — reserved, currently empty. mf_ingestion
                        (mf_ingestion/sync.py) calls api.mfapi.in per scheme
                        and writes straight to Postgres (mf_nav_history) —
                        it does not archive the raw JSON response to S3.
                        Add archiving here if that ever becomes necessary
                        (e.g. for replay/audit).
  sebi/
    sid/{amc_entity_id}/{filename}
      — Scheme Information Document PDFs, scraped per-AMC (SEBI has no
        single bulk SID feed). Written by
        src/financial_pipeline/sebi_ingestion/sync.py (`pipeline
        sebi-sid-sync`). amc_entity_id matches domain/semantic/taxonomy.yaml's
        amcs: entity_id. See sebi_ingestion/amc_sources.py for the current
        AMC registry (3 confirmed-working sources as of this writing —
        extend as more AMCs are added).
```

## silver/

```
silver/
  amfi/
    monthly/{year}/{month}/{chunks,text,tables...}.json
    quarterly/{year}/vol{N}/issue{N}/{chunks,text,tables...}.json
    other/{stem}/{chunks,text,tables...}.json
      — extracted text/chunks/tables from the bronze/amfi/{monthly_aum,
        quarterly_aum,other}/ PDFs, produced by the ECS worker pipeline
        (scripts/process_text_worker.py -> process_table_worker.py /
        process_ocr_worker.py -> process_chunk_worker.py). Path built by
        scripts/_worker_common.py's processed_prefix(). Schema version is
        tracked in Postgres (document_metadata.schema_version), not in the
        path.
    nav_csv/{YYYY-MM-DD}/NAVAll.csv
      — parsed/typed CSV from bronze/amfi/nav/'s NAVAll.txt. Written by the
        standalone scripts/fetch_amfi_nav.py (a local/manual alternative to
        the amfi-nav-connector ECS task — not part of the automated
        pipeline).
```

Nothing currently writes a `silver/mfapi/` or `silver/sebi/` tree — both
sources' "cleaned" representation is the Postgres tables themselves
(`mf_scheme_master`/`mf_nav_history`/`mf_scheme_performance` for mfapi.in;
none yet for SEBI SIDs, since sebi_ingestion is a bronze-only scaffold so far).

## gold/ (reserved)

Not yet implemented. Most of what would count as "gold" — curated,
aggregated, consumption-ready — already lives in Postgres rather than S3:
`amfi_fund_stats`, `mf_scheme_performance`, `document_chunks` (with
embeddings, for RAG retrieval). An S3 `gold/` layer would mainly be for
periodic *exports* of those tables for external/BI consumption, e.g.:

```
gold/
  mf_scheme_performance/{date}/*.parquet   — nightly snapshot export
  amfi_fund_stats/{date}/*.parquet
  reports/                                  — ad-hoc curated report data
```

No export job exists yet. If/when one is built, follow the same
one-way-forward-through-the-layers principle: gold/ exports are derived
from Postgres, not from silver/ or bronze/ directly.

## Migration note

When this layout was introduced, existing data was copied (not moved) from
its pre-medallion locations into the equivalent bronze/silver/ paths above.
The old locations (`amfi/mf_scheme_master/`, `amfi/nav/raw/`,
`amfi/research/{date}/{monthly,quarterly,unknown}/`,
`processed/{amfi/,v1/amfi/}monthly/...`) were left in place, untouched —
they're safe to delete once the new paths have been running long enough to
trust, but that cleanup hasn't been done. Postgres rows created before the
migration (`document_metadata.s3_raw_key`/`s3_processed_key`) still point at
the old paths, which still work; only new documents get bronze/silver keys.

## Adding a new source

1. Pick the right top-level bucket under `bronze/` (source system, e.g.
   `amfi/`, `mfapi/`, `sebi/`) and a subfolder name describing the content
   (not the format — `scheme_master/`, not `json/`).
2. Write raw, unmodified fetched bytes there. No parsing, no field
   renaming, no filtering.
3. If a downstream step produces a cleaned/structured/extracted artifact
   from those raw bytes, write it to the equivalent `silver/` path — same
   subfolder name, same source-system top level.
4. Update this file.
