import click

from financial_pipeline.__version__ import __version__
from financial_pipeline.config import settings
from financial_pipeline.logging import configure_logging


@click.group()
@click.version_option(__version__)
def main() -> None:
    """Financial Data Ingestion Pipeline CLI."""
    configure_logging(level=settings.log_level, fmt=settings.log_format)


@main.command()
@click.option("--source", required=True, help="Data source name to run")
@click.option("--date", "run_date", default=None, help="Target date (YYYY-MM-DD), defaults to today")
def run(source: str, run_date: str | None) -> None:
    """Run a single ingestion source."""
    from datetime import date

    import structlog

    log = structlog.get_logger()
    target = run_date or date.today().isoformat()
    log.info("pipeline.run", source=source, date=target)
    click.echo(f"Running source '{source}' for date {target}")


@main.command()
def sources() -> None:
    """List all registered ingestion sources."""
    click.echo("No sources registered yet.")


@main.command("mf-nav-sync")
@click.option("--s3-key", default=None, help="Exact scheme-master JSON key in S3 (defaults to resolving the latest snapshot under settings.mf_scheme_master_s3_prefix)")
@click.option("--start-date", default=None, help="NAV history start date, YYYY-MM-DD (default: settings.mfapi_default_start_date)")
@click.option("--end-date", default=None, help="NAV history end date, YYYY-MM-DD (default: today)")
@click.option("--max-workers", default=None, type=int, help="Concurrent scheme-processing threads (default: settings.mfapi_max_workers)")
def mf_nav_sync(s3_key: str | None, start_date: str | None, end_date: str | None, max_workers: int | None) -> None:
    """Sync mutual fund scheme master + NAV history from mfapi.in into Postgres."""
    from financial_pipeline.mf_ingestion.sync import run_sync

    result = run_sync(
        postgres_url=settings.postgres_url,
        s3_bucket=settings.s3_bucket,
        s3_key=s3_key,
        s3_prefix=settings.mf_scheme_master_s3_prefix,
        aws_region=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
        start_date=start_date or settings.mfapi_default_start_date,
        end_date=end_date,
        request_delay_seconds=settings.mfapi_request_delay_seconds,
        max_workers=max_workers or settings.mfapi_max_workers,
    )
    click.echo(result)
