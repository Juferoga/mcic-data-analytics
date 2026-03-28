# Technical Design: data-sources

## Overview

This design documents the implementation of expanded data source support for the ETL module. The change adds Excel and JSON source/destination handlers while reorganizing the module structure for better scalability.

## Architecture

### Module Reorganization

```
src/analitica/etl/
├── __init__.py           # Updated exports
├── source.py             # Kept for backward compatibility (re-exports)
├── destination.py        # Kept for backward compatibility (re-exports)
├── pipeline.py           # Unchanged
├── transformer.py        # Unchanged
├── exceptions.py         # Unchanged
├── sources/
│   ├── __init__.py       # New: exports all sources
│   ├── base.py           # New: Source abstract class (moved from source.py)
│   ├── csv.py            # New: CSVSource (moved from source.py)
│   ├── excel.py          # New: ExcelSource
│   └── json.py           # New: JSONSource
└── destinations/
    ├── __init__.py       # New: exports all destinations
    ├── base.py           # New: Destination abstract class (moved from destination.py)
    ├── csv.py            # New: CSVDestination (moved from destination.py)
    ├── excel.py          # New: ExcelDestination
    └── json.py           # New: JSONDestination
```

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| Separate `base.py` files | Keeps abstract classes separate from implementations for clarity |
| Maintain backward compatibility | Keep `source.py` and `destination.py` as re-export modules |
| Use pandas for file I/O | Leverages existing pandas dependency; provides consistent API |
| Support common pandas parameters | Allow users to pass `**kwargs` to customize pandas functions |

## Component Specifications

### 1. Source Base Class (sources/base.py)

```python
from abc import ABC, abstractmethod
import pandas as pd

class Source(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    def extract(self) -> pd.DataFrame:
        """Extract data from the source."""
        pass
```

### 2. CSVSource (sources/csv.py)

**Location**: `src/analitica/etl/sources/csv.py`

**Inherits from**: `Source`

**Implementation**: Wraps `pandas.read_csv`

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | str \| Path | Path to the CSV file |
| `**kwargs` | dict | Additional arguments passed to `pd.read_csv` |

**Error Handling**: Raises `SourceError` for file not found, parse errors, or other failures.

### 3. ExcelSource (sources/excel.py)

**Location**: `src/analitica/etl/sources/excel.py`

**Inherits from**: `Source`

**Implementation**: Wraps `pandas.read_excel`

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | str \| Path | Path to the Excel file |
| `sheet_name` | str \| int \| list \| None | Sheet name, index, or list of sheets (default: 0) |
| `engine` | str \| None | Excel engine: 'openpyxl', 'xlrd', or None (auto-detect) |
| `**kwargs` | dict | Additional arguments passed to `pd.read_excel` |

**Error Handling**: Raises `SourceError` for file not found, sheet not found, or parsing errors.

### 4. JSONSource (sources/json.py)

**Location**: `src/analitica/etl/sources/json.py`

**Inherits from**: `Source`

**Implementation**: Wraps `pandas.read_json`

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `path_or_buffer` | str \| Path \| IO | Path to JSON file or file-like object |
| `orient` | str \| None | JSON orientation: 'records', 'index', 'columns', 'values', 'split' (default: 'columns') |
| `**kwargs` | dict | Additional arguments passed to `pd.read_json` |

**Error Handling**: Raises `SourceError` for file not found or JSON parse errors.

### 5. Destination Base Class (destinations/base.py)

```python
from abc import ABC, abstractmethod
import pandas as pd

class Destination(ABC):
    """Abstract base class for data destinations."""
    
    @abstractmethod
    def save(self, data: pd.DataFrame) -> None:
        """Save data to the destination."""
        pass
```

### 6. CSVDestination (destinations/csv.py)

**Location**: `src/analitica/etl/destinations/csv.py`

**Inherits from**: `Destination`

**Implementation**: Wraps `pandas.DataFrame.to_csv`

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | str \| Path | Path to the output CSV file |
| `**kwargs` | dict | Additional arguments passed to `pd.to_csv` |

**Error Handling**: Raises `DestinationError` for permission errors or write failures.

### 7. ExcelDestination (destinations/excel.py)

**Location**: `src/analitica/etl/destinations/excel.py`

**Inherits from**: `Destination`

**Implementation**: Wraps `pandas.DataFrame.to_excel` with `openpyxl` engine

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | str \| Path | Path to the output Excel file |
| `sheet_name` | str | Sheet name (default: 'Sheet1') |
| `engine` | str | Excel engine: 'openpyxl' (default), 'xlsxwriter' |
| `index` | bool | Write row names (default: False) |
| `**kwargs` | dict | Additional arguments passed to `pd.to_excel` |

**Error Handling**: Raises `DestinationError` for permission errors, engine unavailability, or write failures.

### 8. JSONDestination (destinations/json.py)

**Location**: `src/analitica/etl/destinations/json.py`

**Inherits from**: `Destination`

**Implementation**: Wraps `pandas.DataFrame.to_json`

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | str \| Path | Path to the output JSON file |
| `orient` | str | JSON orientation: 'records', 'index', 'columns', 'values', 'split' (default: 'records') |
| `indent` | int \| None | JSON indentation (default: 2) |
| `**kwargs` | dict | Additional arguments passed to `pd.to_json` |

**Error Handling**: Raises `DestinationError` for permission errors or write failures.

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Pipeline                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────┐    ┌───────────┐  │
│  │  Source  │───▶│  DataFrame   │───▶│   Transformer   │───▶│    Save   │  │
│  └──────────┘    └──────────────┘    └─────────────────┘    └───────────┘  │
│       │                                                      │               │
│       │              Supported Sources:                     │               │
│       ├──────── CSVSource (pandas.read_csv)                 │               │
│       ├──────── ExcelSource (pandas.read_excel)              │               │
│       └──────── JSONSource (pandas.read_json)               │               │
│                                                          │               │
│                                          Supported Destinations:           │
│                                          ├──────── CSVDestination          │
│                                          ├──────── ExcelDestination        │
│                                          └──────── JSONDestination         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/analitica/etl/sources/__init__.py` | Create | Export all source classes |
| `src/analitica/etl/sources/base.py` | Create | Source abstract base class |
| `src/analitica/etl/sources/csv.py` | Create | CSVSource implementation |
| `src/analitica/etl/sources/excel.py` | Create | ExcelSource implementation |
| `src/analitica/etl/sources/json.py` | Create | JSONSource implementation |
| `src/analitica/etl/destinations/__init__.py` | Create | Export all destination classes |
| `src/analitica/etl/destinations/base.py` | Create | Destination abstract base class |
| `src/analitica/etl/destinations/csv.py` | Create | CSVDestination implementation |
| `src/analitica/etl/destinations/excel.py` | Create | ExcelDestination implementation |
| `src/analitica/etl/destinations/json.py` | Create | JSONDestination implementation |
| `src/analitica/etl/source.py` | Modify | Re-export from sources package |
| `src/analitica/etl/destination.py` | Modify | Re-export from destinations package |
| `src/analitica/etl/__init__.py` | Modify | Update exports to include new classes |

## Backward Compatibility

The existing public API remains unchanged:

```python
# These continue to work:
from analitica.etl import Source, CSVSource, Destination, CSVDestination

# New imports also available:
from analitica.etl.sources import ExcelSource, JSONSource
from analitica.etl.destinations import ExcelDestination, JSONDestination
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | >=1.5.0 | DataFrame operations, file I/O |
| `openpyxl` | >=3.0.0 | Excel .xlsx read/write support |
| `xlrd` | >=2.0.0 | Legacy Excel .xls read support (optional) |

Add to `requirements.txt`:
```
openpyxl>=3.0.0
xlrd>=2.0.0
```

## Testing Strategy

1. **Unit Tests**: Test each source/destination in isolation
   - Mock file I/O operations
   - Test error handling paths

2. **Integration Tests**: Test with actual files
   - Create temp files, read/write, verify contents
   - Test with various pandas parameters

3. **Pipeline Integration**: Ensure new sources/destinations work with Pipeline
   - Verify fluent interface continues to work

## Edge Cases

| Scenario | Handling |
|----------|----------|
| Excel file with multiple sheets | `sheet_name` parameter accepts int, str, or list |
| JSON orientation mismatch | Document expected orientations; let pandas raise on mismatch |
| Large Excel files | Use `engine='openpyxl'` for better performance; document memory considerations |
| Missing optional dependencies | Catch ImportError and raise descriptive error message |
| Empty DataFrame | Allow saving empty DataFrames (pandas handles this naturally) |

## Implementation Order

1. Create `sources/` and `destinations/` directories
2. Create `base.py` files with abstract classes
3. Implement `csv.py` files (move existing code)
4. Implement `excel.py` and `json.py` files
5. Create `__init__.py` files for packages
6. Update `source.py` and `destination.py` for backward compatibility
7. Update main `__init__.py` exports
8. Add tests

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| openpyxl not installed | Low | High | Document dependency; catch ImportError gracefully |
| Excel file corruption | Low | Medium | Wrap in try/except; provide clear error messages |
| JSON encoding issues | Low | Medium | Default to UTF-8; allow encoding override via kwargs |
| Breaking existing imports | Medium | Medium | Maintain backward-compatible re-exports |
