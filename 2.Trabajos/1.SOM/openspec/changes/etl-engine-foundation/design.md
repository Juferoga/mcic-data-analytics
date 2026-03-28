# Design: etl-engine-foundation

## Technical Approach

Implement a fluent ETL pipeline with composable Source → Transformer* → Destination stages using pandas DataFrames as the data carrier. Abstract base classes enable extensibility while keeping the core minimal.

## Architecture Decisions

### Decision: Fluent Interface for Pipeline

**Choice**: `pipeline.extract_from(src).transform(t).load_to(dest).run()`
**Alternatives**: Config dict, `pipeline.run(source, transformers, destination)` positional args
**Rationale**: Fluent interface enables readable, declarative pipelines and natural chaining. Each method returns `self`, allowing inline or step-by-step construction.

### Decision: Abstract Base Classes for Source/Destination/Transformer

**Choice**: `class Source(ABC)` with `@abstractmethod def extract() -> pd.DataFrame`
**Alternatives**: Protocol classes (structural typing), concrete-only classes with factory registration
**Rationale**: ABCs enforce the interface contract at instantiation time (fail-fast) while remaining familiar to Python developers. Protocols would defer errors to runtime calls. Factory registration adds indirection without benefit for v1.

### Decision: Single DataFrame State in Pipeline

**Choice**: Pipeline holds `self._df: Optional[pd.DataFrame]`
**Alternatives**: Immutable pass-through with `transform(df) -> df`, context manager with scoped state
**Rationale**: Mutable state simplifies debugging (inspect `pipeline.df` mid-flight) and matches pandas mental model. Immutable would require threading state through all calls.

### Decision: Exception Hierarchy

**Choice**: `PipelineError` → `SourceError`, `DestinationError`, `TransformerError`
**Alternatives**: Single `ETLError`, tuples of exception classes
**Rationale**: Hierarchical allows catching broad `PipelineError` or specific stage failures. Clean separation by stage aids debugging.

## Data Flow

```
┌─────────────┐     extract()      ┌─────────────┐
│   Source    │ ──────────────────→│   Pipeline  │
│ (CSVSource) │     pd.DataFrame   │   (_df)     │
└─────────────┘                    └──────┬──────┘
                                          │ fit_transform()
                                    ┌─────▼──────┐
                                    │ Transformer│
                                    │ (Identity) │
                                    └─────┬──────┘
                                          │ pd.DataFrame
                                    (repeat for N transformers)
                                          │
                                    ┌─────▼──────┐     save()
                                    │  Pipeline   │ ───────────→ ┌─────────────┐
                                    │   (_df)     │  pd.DataFrame │ Destination │
                                    └─────────────┘               │(CSVDest)    │
                                                                   └─────────────┘
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/analitica/etl/pipeline.py` | Create | Pipeline class with fluent interface |
| `src/analitica/etl/source.py` | Create | `Source` ABC + `CSVSource` implementation |
| `src/analitica/etl/destination.py` | Create | `Destination` ABC + `CSVDestination` |
| `src/analitica/etl/transformer.py` | Create | `Transformer` ABC + `IdentityTransformer` |
| `src/analitica/etl/exceptions.py` | Create | Exception hierarchy |
| `src/analitica/etl/__init__.py` | Modify | Export public API |
| `tests/etl/test_pipeline.py` | Create | Unit tests for Pipeline |
| `tests/etl/test_sources.py` | Create | Unit tests for CSVSource |
| `tests/etl/test_destinations.py` | Create | Unit tests for CSVDestination |
| `tests/etl/test_transformers.py` | Create | Unit tests for transformers |
| `tests/conftest.py` | Create | Shared pytest fixtures |

## Interfaces / Contracts

```python
# Pipeline class signature
class Pipeline:
    def __init__(self): ...
    def extract_from(self, source: Source) -> Pipeline: ...
    def add_transformer(self, transformer: Transformer) -> Pipeline: ...
    def load_to(self, destination: Destination) -> Pipeline: ...
    def run(self) -> Pipeline: ...

# Source contract
class Source(ABC):
    @abstractmethod
    def extract(self) -> pd.DataFrame: ...

class CSVSource(Source):
    def __init__(self, path: str | Path, **kwargs): ...
    def extract(self) -> pd.DataFrame: ...

# Destination contract
class Destination(ABC):
    @abstractmethod
    def save(self, df: pd.DataFrame) -> None: ...

class CSVDestination(Destination):
    def __init__(self, path: str | Path, **kwargs): ...
    def save(self, df: pd.DataFrame) -> None: ...

# Transformer contract
class Transformer(ABC):
    @abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame: ...
```

## Testing Strategy

| Layer | What to Test | Approach |
|-------|-------------|----------|
| Unit | Pipeline methods, Source/Dest I/O, Transformer passthrough | `unittest.mock` for file I/O, temp CSV files |
| Integration | Full pipeline execution with real CSV | pytest fixtures with sample data |
| E2E | CLI invocation | subprocess + assert output |

**Fixtures needed**:
- `sample_csv`: Path to test data CSV
- `temp_output`: tmp_path fixture for CSVDestination
- `empty_pipeline`: Pre-built Pipeline instance

## Migration / Rollout

No migration required. Additive changes only:
- New files under `src/analitica/etl/`
- New test files under `tests/etl/`
- `__init__.py` gains exports (non-breaking)

## Open Questions

- [ ] Should `CSVSource` accept `Path` objects or string only? (Design uses `str | Path`)
- [ ] Add type hints throughout? (Design includes them)
- [ ] Support streaming/chunked reads for large files in future? (Note in docs only)
