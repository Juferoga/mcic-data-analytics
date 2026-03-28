"""Data loading destinations - kept for backward compatibility."""

from .destinations.base import Destination
from .destinations.csv import CSVDestination
from .destinations.excel import ExcelDestination
from .destinations.json import JSONDestination

__all__ = ["Destination", "CSVDestination", "ExcelDestination", "JSONDestination"]
