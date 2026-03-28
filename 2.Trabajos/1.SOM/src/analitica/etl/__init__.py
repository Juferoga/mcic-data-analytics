"""ETL Module - Motor de Extracción, Transformación y Carga."""

from .pipeline import Pipeline
from .source import Source, CSVSource
from .destination import Destination, CSVDestination
from .transformer import BaseTransformer, IdentityTransformer
from .exceptions import PipelineError, SourceError, DestinationError, TransformerError

# Sources
from .sources import CSVSource, ExcelSource, JSONSource

# Destinations
from .destinations import CSVDestination, ExcelDestination, JSONDestination

__all__ = [
    # Core
    "Pipeline",
    "Source",
    "Destination",
    "BaseTransformer",
    "IdentityTransformer",
    "PipelineError",
    "SourceError",
    "DestinationError",
    "TransformerError",
    # Sources
    "CSVSource",
    "ExcelSource",
    "JSONSource",
    # Destinations
    "CSVDestination",
    "ExcelDestination",
    "JSONDestination",
]
