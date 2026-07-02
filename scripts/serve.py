#!/usr/bin/env python3
"""Launch the AMFI retrieval API server.

Usage:
    # Development (auto-reload on code change)
    python scripts/serve.py --reload

    # Production (multiple workers)
    python scripts/serve.py --workers 2

    # Custom host / port
    python scripts/serve.py --host 127.0.0.1 --port 9000

The server reads all config from .env (POSTGRES_URL, OPENAI_API_KEY, etc.)
Swagger UI: http://localhost:8080/docs
ReDoc:      http://localhost:8080/redoc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uvicorn

from financial_pipeline.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=settings.api_host)
    parser.add_argument("--port", type=int, default=settings.api_port)
    parser.add_argument("--workers", type=int, default=1, help="Number of uvicorn worker processes")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on source changes (dev only)")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════╗
║   AMFI Financial Document Retrieval API  v2.0        ║
╠══════════════════════════════════════════════════════╣
║  http://{args.host}:{args.port}
║  Swagger UI → /docs
║  Embed model → {settings.embed_model}
║  LLM model   → {settings.openai_model}
║  LLM ready   → {"YES" if settings.openai_api_key else "NO (set OPENAI_API_KEY for /api/ask)"}
╚══════════════════════════════════════════════════════╝
""")

    uvicorn.run(
        "financial_pipeline.api.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
