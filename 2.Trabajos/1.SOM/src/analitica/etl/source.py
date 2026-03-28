"""Data extraction sources - kept for backward compatibility."""

from .sources.base import Source
from .sources.csv import CSVSource
from .sources.excel import ExcelSource
from .sources.json import JSONSource

__all__ = ["Source", "CSVSource", "ExcelSource", "JSONSource"]
