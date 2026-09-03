"""Benchmark ParserRouter's routing decisions and parser-quality deltas
across a directory of PDFs.

Infrastructure for tuning `DefaultRoutingPolicy`, not a tuned result: this
repo has no representative AMFI/SEBI PDFs checked in, so `--input-dir` is
required for a real run. Without it, this runs against two tiny synthetic
PDFs generated in-process purely so the script is runnable/demonstrable —
their numbers say nothing about real-document routing quality and must not
be used to justify policy changes.

Usage:
    python scripts/benchmark_parser_routing.py --input-dir path/to/pdfs [--execute] [--output report.json]
    python scripts/benchmark_parser_routing.py --dry-run  # synthetic PDFs, no real corpus needed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fies_parser.engine.exceptions import ParserEngineError  # noqa: E402
from fies_parser.engine.models import ParseRequest, SourceDocument  # noqa: E402
from fies_parser.engine.parser_engine import ParserEngine  # noqa: E402
from fies_parser.preflight.document_profiler import DocumentProfiler  # noqa: E402
from fies_parser.routing.routing_policy import DefaultRoutingPolicy  # noqa: E402
from financial_pipeline.logging import configure_logging  # noqa: E402
from financial_pipeline.processing.parser_composition_root import build_registry  # noqa: E402


def _make_synthetic_pdfs(tmp_dir: Path) -> list[Path]:
    import pymupdf

    paths = []

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Synthetic text-only document")
    page.insert_text((72, 120), "No tables, no images — should route to a fast parser.")
    path = tmp_dir / "synthetic_text.pdf"
    doc.save(path)
    doc.close()
    paths.append(path)

    doc = pymupdf.open()
    page = doc.new_page()
    x0, y0, cell_w, cell_h, rows, cols = 72, 100, 100, 20, 4, 3
    for r in range(rows + 1):
        page.draw_line((x0, y0 + r * cell_h), (x0 + cols * cell_w, y0 + r * cell_h))
    for c in range(cols + 1):
        page.draw_line((x0 + c * cell_w, y0), (x0 + c * cell_w, y0 + rows * cell_h))
    for r in range(rows):
        for c in range(cols):
            page.insert_text((x0 + c * cell_w + 5, y0 + r * cell_h + 14), f"R{r}C{c}", fontsize=8)
    path = tmp_dir / "synthetic_table.pdf"
    doc.save(path)
    doc.close()
    paths.append(path)

    return paths


def _benchmark_one(pdf_path: Path, registry: Any, engine: ParserEngine, execute: bool) -> dict[str, Any]:
    file_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    document = SourceDocument(
        document_id=file_hash,
        file_path=pdf_path,
        file_name=pdf_path.name,
        mime_type="application/pdf",
        file_hash=file_hash,
        source="benchmark",
    )
    request = ParseRequest(document=document)

    profile = DocumentProfiler().profile(document)
    available = {name: registry.get(name).capabilities for name in registry.list_parsers()}
    decision = DefaultRoutingPolicy().decide(profile, available)

    row: dict[str, Any] = {
        "file": pdf_path.name,
        "page_count": profile.page_count,
        "has_text_layer": profile.has_text_layer,
        "likely_has_tables": profile.likely_has_tables,
        "routed_parser": decision.parser_name,
        "routing_reason": decision.reason,
    }

    if execute:
        for parser_name in registry.list_parsers():
            try:
                start = time.perf_counter()
                candidate = engine.run(parser_name, request)
                duration = time.perf_counter() - start
                row[f"{parser_name}_duration_s"] = round(duration, 4)
                row[f"{parser_name}_element_count"] = len(candidate.elements)
                row[f"{parser_name}_table_count"] = len(candidate.tables)
                row[f"{parser_name}_warning_count"] = len(candidate.warnings)
            except ParserEngineError as exc:
                row[f"{parser_name}_error"] = str(exc)

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory of PDFs to benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic PDFs instead of --input-dir")
    parser.add_argument("--execute", action="store_true", help="Actually run every registered parser (slow: loads Docling)")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of files processed")
    parser.add_argument("--output", type=Path, default=None, help="Write the JSON report here (default: stdout)")
    args = parser.parse_args()

    configure_logging(level="INFO", fmt="console")

    if not args.input_dir and not args.dry_run:
        parser.error("--input-dir is required unless --dry-run is set")

    import tempfile

    with tempfile.TemporaryDirectory(prefix="fies_parser_benchmark_") as tmp_dir:
        if args.dry_run:
            pdf_paths = _make_synthetic_pdfs(Path(tmp_dir))
        else:
            pdf_paths = sorted(args.input_dir.glob("*.pdf"))

        if args.limit:
            pdf_paths = pdf_paths[: args.limit]

        if not pdf_paths:
            parser.error(f"no PDFs found in {args.input_dir}")

        registry = build_registry()
        engine = ParserEngine(registry)

        rows = [_benchmark_one(path, registry, engine, execute=args.execute) for path in pdf_paths]

    agreement_note = (
        "SYNTHETIC DATA — not representative of real routing quality. Run with --input-dir "
        "against real AMFI/SEBI documents before using this to tune DefaultRoutingPolicy."
        if args.dry_run
        else None
    )
    report = {"documents": rows, "count": len(rows), "note": agreement_note}

    output_text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(output_text)
        print(f"Wrote report for {len(rows)} document(s) to {args.output}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
