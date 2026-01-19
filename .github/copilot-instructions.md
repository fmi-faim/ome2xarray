# Copilot Instructions for ome2xarray

## Project Overview

`ome2xarray` is a Python library that provides tools to load OME "companion.ome" metadata and associated TIFF data into xarray datasets with Dask-backed arrays. The library focuses on working with microscopy data stored in OME-TIFF format.

## Development Environment

### Package Manager
This project uses **Pixi** for environment and dependency management, not pip or conda directly.

### Python Versions
The project supports Python 3.11, 3.12, 3.13, and 3.14. Active CI testing is performed on Python 3.11, 3.12, and 3.13 (Python 3.14 support is configured but not yet included in CI).

### Setup
To set up the development environment:
```bash
# Pixi will automatically handle environment setup
pixi install
```

## Project Structure

```
ome2xarray/
├── src/
│   └── ome2xarray/
│       ├── __init__.py       # Package initialization
│       └── companion.py      # Main CompanionFile class and logic
├── tests/
│   ├── test_companion.py     # Test suite
│   └── resources/            # Test data files
├── pyproject.toml            # Project configuration and dependencies
└── pixi.lock                 # Lock file for reproducible builds
```

## Key Components

### CompanionFile Class
The main class for working with OME companion files:
- `__init__(path: Path, data_folder: Path | None = None) -> None`: Initialize with a companion.ome file path
- `get_dataset(image_index: int) -> xr.Dataset`: Get an xarray Dataset for a specific image index
- `get_datatree() -> xr.DataTree`: Get an xarray DataTree containing all images
- `get_ome_metadata() -> OME`: Get the raw OME metadata object

### Data Structure
- Datasets use dimensions: `(t, z, y, x)` for time, z-position, y-axis, and x-axis
- Each channel is stored as a separate DataArray within the Dataset
- Arrays are backed by Dask for lazy loading and efficient computation

## Testing

### Running Tests
```bash
# Run tests with coverage for the default Python environment
pixi run test

# Run tests for a specific Python version
pixi run --environment py311 test
pixi run --environment py312 test
pixi run --environment py313 test
pixi run --environment py314 test  # Available but not yet in CI

# Generate XML coverage report (used in CI)
pixi run cov-xml
```

### Test Guidelines
- Tests are located in `tests/test_companion.py`
- Use `pytest` for writing tests
- Parametrize tests when testing multiple similar scenarios
- Test files rely on sample data in `tests/resources/`
- Verify data integrity using checksums (sum of pixel values)
- Test edge cases like invalid indices and missing data files

## Code Conventions

### Style
- Follow standard Python conventions (PEP 8)
- Use type hints for function parameters and return values (e.g., `Path | None`)
- Use descriptive variable names

### Imports
- Standard library imports first
- Third-party imports second (grouped: dask, ome_types, numpy, tifffile, xarray)
- Local imports last

### Error Handling
- Raise `IndexError` for out-of-bounds image indices
- Use `warnings.warn()` with `UserWarning` for non-critical issues (e.g., missing data files)
- Validate metadata consistency and raise `ValueError` for inconsistent data

### Dependencies
Key dependencies:
- `xarray`: Multi-dimensional labeled arrays and datasets
- `dask`: Parallel computing and lazy evaluation
- `ome-types`: OME metadata parsing
- `tifffile`: TIFF file reading

## Building and Distribution

### Version Management
- Version is managed via `hatch-vcs` from git tags
- No manual version updates in code

### Build System
- Uses `hatchling` as the build backend
- Editable install configured via `pyproject.toml`

## CI/CD

### Continuous Integration
- Tests run on Ubuntu, Windows, and macOS
- Tests run against all supported Python versions
- Coverage reports are uploaded to Codecov

### Workflows
- `test.yml`: Runs test suite on push and pull requests
- `publish.yml`: Handles package publishing

## Common Tasks

### Adding a New Feature
1. Implement the feature in `src/ome2xarray/companion.py`
2. Add type hints for all new functions/methods
3. Add tests in `tests/test_companion.py` with appropriate test data
4. Run tests: `pixi run test`
5. Ensure tests pass on all supported Python versions

### Fixing a Bug
1. Add a test that reproduces the bug
2. Fix the bug in the source code
3. Verify the test now passes: `pixi run test`
4. Check edge cases

### Updating Dependencies
1. Modify version constraints in `pyproject.toml`
2. Run `pixi install` to update lock file
3. Test thoroughly across all Python versions

## Additional Notes

- Always use `Path` objects from `pathlib` for file paths, not strings
- Leverage Dask's lazy evaluation for efficient memory usage
- When working with OME metadata, handle missing optional fields gracefully
- Coordinate systems follow OME conventions (position values in micrometers)
