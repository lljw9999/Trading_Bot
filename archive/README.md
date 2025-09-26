# Archive Directory

This directory contains legacy files and configurations that have been archived during the codebase modernization process.

## Directory Structure

### `backup_files/`
- `.env.backup` - Old environment configuration backup
- `Makefile.backup` - Previous Makefile backup

### `legacy_configs/`
- `kaggle.json` - Legacy Kaggle API configuration (replaced by environment variables)
- `model_registry.json` - Old model registry configuration (superseded by dynamic model management)

### `legacy_scripts/`
- `Pull_Data.py` - Legacy data pulling script with hardcoded 2025 dates (replaced by dynamic data ingestion)
- `test_simple_dashboard.py` - Basic dashboard test (superseded by comprehensive dashboard tests)
- `test_dashboard_data.py` - Duplicate dashboard data test (consolidated into main test suite)

## Why Files Were Archived

1. **Backup Files**: Old backup files that are no longer needed with proper version control
2. **Legacy Configurations**: Static config files replaced by environment-based configuration
3. **Duplicate Tests**: Test files that overlap with more comprehensive test suites
4. **Hardcoded Scripts**: Scripts with hardcoded values replaced by parameterized versions

## Recovery

If any archived file is needed, it can be restored by copying it back to the root directory or appropriate location. All files maintain their original functionality.

## Modernization Benefits

- Cleaner root directory structure
- Reduced confusion from duplicate files
- Better separation of current vs. legacy code
- Improved maintainability

## Next Steps

Consider reviewing the archived files periodically and permanently removing those that are no longer needed after sufficient testing of the modernized codebase.