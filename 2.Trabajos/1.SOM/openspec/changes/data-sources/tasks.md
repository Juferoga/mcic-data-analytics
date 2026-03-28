# Tasks: Data Sources Expansion

## Phase 1: Module Reorganization

- [x] 1.1 Create `src/analitica/etl/sources/` directory
- [x] 1.2 Create `src/analitica/etl/destinations/` directory
- [x] 1.3 Move CSVSource to `src/analitica/etl/sources/csv.py`
- [x] 1.4 Move CSVDestination to `src/analitica/etl/destinations/csv.py`
- [x] 1.5 Update `src/analitica/etl/__init__.py` exports

## Phase 2: Excel Support

- [x] 2.1 Create `src/analitica/etl/sources/excel.py` with ExcelSource
- [x] 2.2 Create `src/analitica/etl/destinations/excel.py` with ExcelDestination
- [x] 2.3 Update __init__.py exports

## Phase 3: JSON Support

- [x] 3.1 Create `src/analitica/etl/sources/json.py` with JSONSource
- [x] 3.2 Create `src/analitica/etl/destinations/json.py` with JSONDestination
- [x] 3.3 Update __init__.py exports

## Phase 4: Dependencies

- [x] 4.1 Add openpyxl to dependencies (for Excel support)
- [x] 4.2 Update pyproject.toml

## Phase 5: Tests

- [x] 5.1 Create `tests/test_sources.py` for ExcelSource and JSONSource
- [x] 5.2 Create `tests/test_destinations.py` for ExcelDestination and JSONDestination
- [x] 5.3 Run pytest

## Phase 6: Sample Data

- [x] 6.1 Create `data/samples/sales.xlsx` (Excel sample)
- [x] 6.2 Create `data/samples/products.json` (JSON sample)

## Implementation Order

1. **Phase 1 first**: Module reorganization — creates the new directory structure and moves existing code
2. **Phase 2 next**: Excel support — new source/destination implementations
3. **Phase 3 next**: JSON support — new source/destination implementations
4. **Phase 4**: Dependencies — add openpyxl for Excel support
5. **Phase 5**: Tests — verify all new components work
6. **Phase 6 final**: Sample data for demos and integration testing

## Verification

- [x] All sources/destinations can read/write their respective formats
- [x] Tests pass: `pytest tests/ -v`
- [x] Sample data files are valid and loadable

(End of file - total 52 lines)
