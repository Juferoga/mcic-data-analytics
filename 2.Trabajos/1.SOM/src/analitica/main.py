"""Main entry point for Analitica."""

from pathlib import Path

from analitica.etl import Pipeline, CSVSource, CSVDestination


def main():
    """Demonstrate basic ETL pipeline usage."""
    # Example: Simple pipeline with identity transformer
    pipeline = (
        Pipeline()
        .extract_from(CSVSource("data/input.csv"))
        .load_to(CSVDestination("data/output.csv"))
    )

    result = pipeline.run()
    print(f"Processed {len(result)} rows")
    return result


if __name__ == "__main__":
    main()
