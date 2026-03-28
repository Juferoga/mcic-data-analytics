"""Custom exceptions for the ETL module."""


class PipelineError(Exception):
    """Raised when pipeline execution fails."""

    pass


class SourceError(Exception):
    """Raised when data extraction fails."""

    pass


class DestinationError(Exception):
    """Raised when data saving fails."""

    pass


class TransformerError(Exception):
    """Raised when transformation fails."""

    pass
