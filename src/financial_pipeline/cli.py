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
