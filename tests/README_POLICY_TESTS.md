# Policy System Tests

This directory contains tests for the SNN Policy Service.

## Quick Start

### No Dependencies Required

These tests can run immediately without any setup:

```bash
# Check syntax
python3 tests/test_policy_syntax.py

# Check structure
python3 tests/test_policy_quick_check.py
```

### With Dependencies

For full functionality tests, you need the proper environment:

```bash
# Install dependencies (if not already installed)
pip install numpy pytest

# Run integration tests
pytest tests/test_policy_integration.py -v

# Run SNN tests (requires PyTorch/snnTorch)
pytest tests/test_snn_components.py -v
```

## Test Files

| File | Purpose | Dependencies |
|------|---------|--------------|
| `test_policy_syntax.py` | Syntax validation | None |
| `test_policy_quick_check.py` | Structure validation | None |
| `test_policy_type_hints.py` | Type hint checking | None |
| `test_policy_integration.py` | Integration tests | pytest, numpy |
| `test_snn_components.py` | SNN component tests | pytest, torch, snntorch |
| `test_policy_sanity.py` | Comprehensive checks | numpy |
| `test_policy_validation.py` | Basic validation | numpy |
| `test_policy_edge_cases.py` | Edge case tests | numpy |

## Test Categories

### 1. Syntax & Structure (No Dependencies)
- ✅ Python syntax validation
- ✅ File structure checks
- ✅ Class definition checks
- ✅ Interface compliance

### 2. Unit Tests (Require Dependencies)
- Data structure validation
- Feature computation
- Policy decisions
- Safety filtering

### 3. Integration Tests (Require Dependencies)
- End-to-end pipeline
- Component integration
- Real-world scenarios

### 4. SNN Tests (Require PyTorch/snnTorch)
- Spike encoding
- SNN inference
- Decision decoding
- Temporal context

## Running All Tests

```bash
# Quick checks (no dependencies)
python3 tests/test_policy_syntax.py
python3 tests/test_policy_quick_check.py

# Full test suite (with dependencies)
pytest tests/ -v
```

## Expected Results

### Syntax Tests
- ✅ All files syntactically valid
- ✅ All imports correct
- ✅ All classes defined

### Integration Tests
- ✅ Topology service wraps graph correctly
- ✅ Feature service builds features
- ✅ Policy service makes decisions
- ✅ Safety arbitration filters correctly
- ✅ End-to-end pipeline works

### SNN Tests (if available)
- ✅ Spike encoding works
- ✅ SNN forward pass works
- ✅ Decision decoding works
- ✅ SNN policy integration works

## Troubleshooting

### Import Errors
If you see `ModuleNotFoundError`, install dependencies:
```bash
pip install numpy pytest
# For SNN tests:
pip install torch snntorch
```

### Syntax Errors
If syntax tests fail, check Python version (requires 3.8+)

### Test Failures
Check that:
1. All files are present in `src/hippocampus_core/policy/`
2. Dependencies are installed
3. Environment is set up correctly

## CI Integration

These tests are designed to run in CI:
- Syntax tests run without dependencies
- Integration tests require test environment
- SNN tests are optional (skip if PyTorch unavailable)

