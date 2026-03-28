# Proposal: data-sources

## Intent

Add support for Excel (.xlsx, .xls) and JSON data sources/destinations to the ETL engine. This expands the system's versatility for real-world data pipelines where CSV is insufficient (e.g., Excel reports, API JSON exports).

## Scope

### In Scope
- `ExcelSource`: Read .xlsx and .xls files using pandas with openpyxl engine
- `ExcelDestination`: Write DataFrames to .xlsx files
- `JSONSource`: Read JSON files (array of objects format)
- `JSONDestination`: Write DataFrames to JSON files
- Update Pipeline to accept new source/destination types

### Out of Scope
- Database sources/destinations (SQL, NoSQL)
- Streaming JSON or nested JSON structures
- Excel with multiple sheets
- Authentication for file sources

## Approach

**Extend Existing Interfaces**: Add new Source/Destination implementations that inherit from base classes in `src/analitica/etl/source.py` and `src/analitica/etl/destination.py`.

- `ExcelSource` → uses `pd.read_excel(path, engine='openpyxl')`
- `ExcelDestination` → uses `df.to_excel(path, engine='openpyxl')`
- `JSONSource` → uses `pd.read_json(path, orient='records')`
- `JSONDestination` → uses `df.to_json(path, orient='records')`

**Add Dependencies**: Include `openpyxl` in `pyproject.toml` dependencies.

**Interface Consistency**: Maintain same `__init__(path, **kwargs)` signature and `extract()`/`save(data)` methods.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/analitica/etl/source.py` | Modified | Add ExcelSource, JSONSource classes |
| `src/analitica/etl/destination.py` | Modified | Add ExcelDestination, JSONDestination classes |
| `src/analitica/etl/__init__.py` | Modified | Export new source/destination classes |
| `pyproject.toml` | Modified | Add openpyxl dependency |
| `tests/` | New | Unit tests for Excel/JSON sources/destinations |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Large Excel files cause memory issues | Low | Document memory considerations in docs |
| JSON nested structures not supported | Low | Scope to flat array-of-objects; document limitation |
| openpyxl version compatibility | Low | Pin openpyxl>=3.0.0 in dependencies |

## Rollback Plan

1. Remove `ExcelSource`, `ExcelDestination`, `JSONSource`, `JSONDestination` classes from source/destination modules
2. Revert `src/analitica/etl/__init__.py` exports
3. Remove `openpyxl` from `pyproject.toml` dependencies
4. Delete test files in `tests/` for new components

All changes are additive; rollback is straightforward deletion.

## Dependencies

- `pandas>=2.0.0` (existing)
- `openpyxl>=3.0.0` (new - for Excel support)
- `numpy>=1.24.0` (existing)

## Success Criteria

- [ ] ExcelSource reads .xlsx files into DataFrame
- [ ] ExcelDestination writes DataFrame to .xlsx file
- [ ] JSONSource reads JSON array-of-objects into DataFrame
- [ ] JSONDestination writes DataFrame to JSON file
- [ ] Pipeline accepts new source/destination types via fluent interface
- [ ] All new components have unit tests
- [ ] openpyxl added to project dependencies
