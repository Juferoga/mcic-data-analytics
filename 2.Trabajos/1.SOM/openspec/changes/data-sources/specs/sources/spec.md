# Data Sources Specification

## Purpose

Define Excel and JSON source/destination implementations for the ETL pipeline, extending data ingestion and output capabilities beyond CSV format.

---

## ADDED Requirements

### Requirement: ExcelSource Implementation

The system **MUST** provide an `ExcelSource` class that reads Excel files into pandas DataFrames.

ExcelSource **SHALL** accept a `path` parameter and optional `sheet_name` parameter (default: 0).

ExcelSource **SHALL** support both `.xlsx` and `.xls` file formats.

#### Scenario: ExcelSource reads valid XLSX file

- GIVEN an Excel file at `/data/input.xlsx` with sheet containing columns `id,name,value`
- WHEN `ExcelSource('/data/input.xlsx').extract()` is called
- THEN it returns a DataFrame with 3 columns and matching row count

#### Scenario: ExcelSource reads specific sheet by index

- GIVEN an Excel file with multiple sheets at `/data/multisheet.xlsx`
- WHEN `ExcelSource('/data/multisheet.xlsx', sheet_name=1).extract()` is called
- THEN it returns a DataFrame from the second sheet (index 1)

#### Scenario: ExcelSource reads specific sheet by name

- GIVEN an Excel file with a sheet named "SalesData"
- WHEN `ExcelSource('/data/multisheet.xlsx', sheet_name='SalesData').extract()` is called
- THEN it returns a DataFrame from the "SalesData" sheet

#### Scenario: ExcelSource raises error on missing file

- GIVEN a path to a non-existent Excel file `/data/nonexistent.xlsx`
- WHEN `ExcelSource('/data/nonexistent.xlsx').extract()` is called
- THEN the system **MUST** raise `SourceError` with message indicating file not found

#### Scenario: ExcelSource raises error on corrupted file

- GIVEN a corrupted Excel file at `/data/corrupted.xlsx`
- WHEN `ExcelSource('/data/corrupted.xlsx').extract()` is called
- THEN the system **MUST** raise `SourceError` with parsing error details

---

### Requirement: ExcelDestination Implementation

The system **MUST** provide an `ExcelDestination` class that writes DataFrames to Excel files.

ExcelDestination **SHALL** accept a `path` parameter and optional `sheet_name` parameter (default: 'Sheet1').

ExcelDestination **SHALL** use the `openpyxl` engine for writing.

ExcelDestination **SHALL** create parent directories if they do not exist.

#### Scenario: ExcelDestination writes to new file

- GIVEN a DataFrame with columns `id,name,value`
- WHEN `ExcelDestination('/output/results.xlsx').save(data)` is called
- THEN the file `/output/results.xlsx` is created with the data in default sheet 'Sheet1'

#### Scenario: ExcelDestination writes to custom sheet

- GIVEN a DataFrame with data
- WHEN `ExcelDestination('/output/results.xlsx', sheet_name='Report').save(data)` is called
- THEN the file contains the data in a sheet named 'Report'

#### Scenario: ExcelDestination creates parent directories

- GIVEN a DataFrame
- AND destination path `/output/nested/dir/results.xlsx` where parent directories do not exist
- WHEN `ExcelDestination('/output/nested/dir/results.xlsx').save(data)` is called
- THEN parent directories are created automatically
- AND the Excel file is written successfully

#### Scenario: ExcelDestination overwrites existing file

- GIVEN an ExcelDestination configured with path to an existing file
- WHEN `save(new_data)` is called
- THEN the existing file is overwritten with new_data

---

### Requirement: JSONSource Implementation

The system **MUST** provide a `JSONSource` class that reads JSON files into pandas DataFrames.

JSONSource **SHALL** accept a `path` parameter.

JSONSource **SHALL** support JSON arrays of objects using `pandas.read_json` with `orient='records'`.

JSONSource **SHALL** optionally support nested JSON structures.

#### Scenario: JSONSource reads valid JSON array

- GIVEN a JSON file at `/data/input.json` containing `[{"id":1,"name":"Alice"},{"id":2,"name":"Bob"}]`
- WHEN `JSONSource('/data/input.json').extract()` is called
- THEN it returns a DataFrame with columns `id,name` and 2 rows

#### Scenario: JSONSource reads nested JSON (flat mode)

- GIVEN a JSON file with nested objects `[{"id":1,"address":{"city":"NYC"}}]`
- WHEN `JSONSource('/data/nested.json').extract()` is called
- THEN it returns a DataFrame (nested fields handled per pandas default behavior)

#### Scenario: JSONSource raises error on invalid JSON

- GIVEN a file containing invalid JSON `{broken json`
- WHEN `JSONSource('/data/invalid.json').extract()` is called
- THEN the system **MUST** raise `SourceError` with JSON parsing error details

#### Scenario: JSONSource raises error on missing file

- GIVEN a path to a non-existent JSON file `/data/nonexistent.json`
- WHEN `JSONSource('/data/nonexistent.json').extract()` is called
- THEN the system **MUST** raise `SourceError` with message indicating file not found

---

### Requirement: JSONDestination Implementation

The system **MUST** provide a `JSONDestination` class that writes DataFrames to JSON files.

JSONDestination **SHALL** accept a `path` parameter and optional `pretty_print` parameter (default: False).

JSONDestination **SHALL** write DataFrames as JSON arrays using `pandas.to_json` with `orient='records'`.

JSONDestination **SHALL** create parent directories if they do not exist.

#### Scenario: JSONDestination writes valid JSON array

- GIVEN a DataFrame with columns `id,name` and rows
- WHEN `JSONDestination('/output/data.json').save(data)` is called
- THEN the file contains a valid JSON array `[{...},{...}]`

#### Scenario: JSONDestination writes with pretty print

- GIVEN a DataFrame
- WHEN `JSONDestination('/output/formatted.json', pretty_print=True).save(data)` is called
- THEN the JSON file is written with indentation for readability

#### Scenario: JSONDestination creates parent directories

- GIVEN a DataFrame
- AND destination path `/output/nested/dir/data.json` where parent directories do not exist
- WHEN `JSONDestination('/output/nested/dir/data.json').save(data)` is called
- THEN parent directories are created automatically
- AND the JSON file is written successfully

#### Scenario: JSONDestination overwrites existing file

- GIVEN a JSONDestination configured with path to an existing file
- WHEN `save(new_data)` is called
- THEN the existing file is overwritten with new_data

---

## Error Handling

| Error Type | Condition | Behavior |
|------------|-----------|----------|
| `SourceError` | Source extraction fails | Raised with descriptive message |
| `SourceError` | File not found | Raised with "file not found" message |
| `SourceError` | Parse error | Raised with parsing details |

---

## Success Criteria

- [ ] ExcelSource reads .xlsx and .xls files into DataFrames
- [ ] ExcelSource supports sheet_name parameter (index and string)
- [ ] ExcelSource raises SourceError on file not found or parse error
- [ ] ExcelDestination writes to Excel files with openpyxl engine
- [ ] ExcelDestination supports custom sheet_name parameter
- [ ] ExcelDestination creates parent directories automatically
- [ ] JSONSource reads JSON arrays into DataFrames
- [ ] JSONSource raises SourceError on invalid JSON
- [ ] JSONDestination writes DataFrames as JSON arrays
- [ ] JSONDestination supports pretty_print option
- [ ] JSONDestination creates parent directories automatically

---

## Data Flow

```
ExcelSource → DataFrame → ExcelDestination
JSONSource  → DataFrame → JSONDestination
```

All sources **SHALL** return pandas DataFrames.

All destinations **SHALL** accept pandas DataFrames as input.
