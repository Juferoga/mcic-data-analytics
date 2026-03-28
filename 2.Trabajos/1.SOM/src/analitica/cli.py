"""CLI entry point for Analitica."""

import sys
from pathlib import Path

import click

from analitica.etl import Pipeline, CSVSource, CSVDestination


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Analitica - ETL Engine with SOM."""
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
def run(input_file: str, output_file: str):
    """Run ETL pipeline on INPUT_FILE saving to OUTPUT_FILE."""
    pipeline = (
        Pipeline()
        .extract_from(CSVSource(input_file))
        .load_to(CSVDestination(output_file))
    )
    result = pipeline.run()
    click.echo(f"✓ Processed {len(result)} rows")
    click.echo(f"✓ Output saved to: {output_file}")


@cli.command()
@click.argument("file", type=click.Path(exists=True))
def inspect(file: str):
    """Inspect a CSV file and display basic info."""
    import pandas as pd

    df = pd.read_csv(file)
    click.echo(f"\n=== {Path(file).name} ===")
    click.echo(f"Rows: {len(df)}")
    click.echo(f"Columns: {len(df.columns)}")
    click.echo(f"\nColumn types:")
    for col, dtype in df.dtypes.items():
        click.echo(f"  {col}: {dtype}")
    click.echo(f"\nFirst 5 rows:")
    click.echo(df.head().to_string())


def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
