# ETL Pipeline Specification

## Purpose

Define the ETL pipeline infrastructure enabling data extraction, transformation, and loading through a fluent, composable interface using pandas DataFrames.

---

## ADDED Requirements

### Requirement: Pipeline Fluent Interface

The system **MUST** provide a fluent interface supporting chained method calls in the sequence `extract_from().transform().load_to().run()`.

The Pipeline class **SHALL** accept `Source`, `Transformer[]`, and `Destination` via constructor or chain methods.

Each chain method **MUST** return `self` to enable method chaining.

#### Scenario: Successful pipeline execution

- GIVEN a Pipeline instance with CSVSource pointing to an existing file
- WHEN the user calls `pipeline.extract_from(src).transform(t).load_to(dst).run()`
- THEN the pipeline extracts data, applies transformation, and saves output
- AND each method returns self enabling the chain

#### Scenario: Pipeline execution failure

- GIVEN a Pipeline instance with an invalid Source
- WHEN `run()` is invoked
- THEN the system **MUST** raise `PipelineError` with descriptive message
- AND execution halts without loading partial data

---

### Requirement: Source Interface

The system **MUST** define an abstract `Source` base class with an `extract() -> DataFrame` method.

The Source interface **SHALL** be extensible for multiple source types.

#### Scenario: Successful data extraction

- GIVEN a Source implementation initialized with valid configuration
- WHEN `extract()` is called
- THEN it returns a pandas DataFrame containing the source data

#### Scenario: Source extraction error

- GIVEN a CSVSource with a non-existent file path
- WHEN `extract()` is invoked
- THEN the system **MUST** raise `SourceError` indicating file not found

---

### Requirement: CSVSource Implementation

The system **MUST** provide a `CSVSource` class accepting a `path` parameter.

CSVSource **SHALL** parse the CSV file and return a DataFrame via `extract()`.

#### Scenario: CSVSource reads valid file

- GIVEN a valid CSV file at `/data/input.csv` with columns `id,name,value`
- WHEN `CSVSource('/data/input.csv').extract()` is called
- THEN it returns a DataFrame with 3 columns and matching row count

#### Scenario: CSVSource handles malformed CSV

- GIVEN a CSVSource with path to a malformed CSV file
- WHEN `extract()` is invoked
- THEN the system **MUST** raise `SourceError` with parsing details

---

### Requirement: Destination Interface

The system **MUST** define an abstract `Destination` base class with `save(data: DataFrame)` method.

The Destination interface **SHALL** be extensible for multiple destination types.

#### Scenario: Successful data saving

- GIVEN a Destination implementation initialized with valid configuration
- WHEN `save(data)` is called with a DataFrame
- THEN the data is persisted to the configured destination

---

### Requirement: CSVDestination Implementation

The system **MUST** provide a `CSVDestination` class accepting a `path` parameter.

CSVDestination **SHALL** create parent directories if they do not exist.

CSVDestination **SHALL** write the DataFrame to CSV format via `save()`.

#### Scenario: CSVDestination writes to new directory

- GIVEN a CSVDestination configured with path `/output/results/data.csv`
- AND the parent directory `/output/results/` does not exist
- WHEN `save(data)` is called
- THEN parent directories are created automatically
- AND the CSV file is written successfully

#### Scenario: CSVDestination overwrites existing file

- GIVEN a CSVDestination configured with path to an existing file
- WHEN `save(new_data)` is called
- THEN the existing file is overwritten with new_data

---

### Requirement: Transformer Interface

The system **MUST** define an abstract `Transformer` base class with `fit_transform(data: DataFrame) -> DataFrame` method.

The Transformer **MUST** store learned parameters during the fit phase.

Transformers **SHALL** be composable in a pipeline chain.

#### Scenario: Transformer processes data

- GIVEN a Transformer implementation with `fit_transform` method
- WHEN `fit_transform(input_data)` is called
- THEN it returns a transformed DataFrame
- AND internal state is updated with learned parameters

#### Scenario: Multiple transformers in chain

- GIVEN a Pipeline with multiple Transformer instances
- WHEN data flows through the chain
- THEN each transformer applies its `fit_transform` in sequence
- AND output of each transformer becomes input to the next

---

## Data Flow

```
Source → [Transformer₁ → Transformer₂ → ... → Transformerₙ] → Destination
```

All data transfer between stages **SHALL** use pandas DataFrames.

---

## Error Handling

| Error Type | Condition | Behavior |
|------------|-----------|----------|
| `SourceError` | Source extraction fails | Raised with descriptive message |
| `PipelineError` | Pipeline execution fails | Raised with descriptive message |

---

## Success Criteria

- [ ] `Pipeline().extract_from().transform().load_to().run()` executes without errors
- [ ] CSVSource reads a CSV file into a DataFrame
- [ ] CSVDestination writes a DataFrame to CSV with auto directory creation
- [ ] Multiple Transformers chain in sequence
- [ ] PipelineError raised on execution failure
- [ ] SourceError raised on source extraction failure
