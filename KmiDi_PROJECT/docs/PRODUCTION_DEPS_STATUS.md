# Production Dependencies Status

**Date**: 2025-01-02  
**Status**: ✅ **COMPLETE** - Production requirements file created

## Summary

Created a curated `requirements-production.txt` file with only production dependencies, excluding development tools.

## File Location

```
requirements-production.txt
```

## Contents

### Core Dependencies (from pyproject.toml)
- `numpy>=1.21.0` - Numerical operations
- `torch>=2.0.0` - PyTorch for ML models
- `librosa>=0.10.0` - Audio analysis
- `pyyaml>=6.0.0` - YAML configuration
- `scipy>=1.8.0` - Scientific computing

### Audio Processing
- `soundfile>=0.12.0` - Audio I/O
- `pydub>=0.25.0` - Audio manipulation

### FastAPI Production Service
- `fastapi>=0.125.0` - Web framework
- `uvicorn[standard]>=0.30.0` - ASGI server
- `slowapi>=0.1.9` - Rate limiting
- `pydantic>=2.0.0` - Data validation
- `python-multipart>=0.0.6` - Form data parsing

### Optional UI Components
- `streamlit>=1.52.0` - Streamlit demo app
- `PySide6>=6.5.0` - Qt GUI application

### MIDI Processing
- `mido>=1.2.10` - MIDI I/O
- `pretty_midi>=0.2.10` - MIDI manipulation

### HTTP Clients
- `requests>=2.31.0` - HTTP library
- `httpx>=0.25.0` - Async HTTP client

### Data Processing
- `pandas>=1.4.0` - Data analysis

### MCP Servers (optional)
- `fastmcp>=2.0.0` - MCP protocol
- `python-dotenv>=1.0.0` - Environment variables

### Security & Utilities
- `certifi>=2024.0.0` - SSL certificates
- `cryptography>=41.0.0` - Cryptographic functions

## Excluded Dependencies

The following development dependencies are **NOT** included:

- `pytest`, `pytest-cov` - Testing frameworks
- `black`, `flake8`, `mypy` - Code formatting and linting
- `coverage` - Code coverage
- `ipython`, `jupyter`, `notebook` - Interactive development
- `sphinx`, `sphinx-rtd-theme` - Documentation generation

These should be installed separately for development:
```bash
pip install -e ".[dev]"
```

## Installation

### Production Installation
```bash
pip install -r requirements-production.txt
```

### Development Installation
```bash
# Install production dependencies
pip install -r requirements-production.txt

# Install development dependencies
pip install -e ".[dev]"
```

### Full Installation (including optional dependencies)
```bash
pip install -e ".[all]"
```

## Security Audit

To audit dependencies for security vulnerabilities:

```bash
# Install pip-audit
pip install pip-audit

# Run audit
pip-audit -r requirements-production.txt

# Or use safety (alternative)
pip install safety
safety check -r requirements-production.txt
```

## File Comparison

### Before (pip freeze - 225 packages)
- Included all installed packages
- Included development dependencies
- Included transitive dependencies
- Harder to maintain

### After (Curated - ~30 packages)
- Only production dependencies
- Excludes development tools
- Version constraints for flexibility
- Easier to maintain and audit

## Maintenance

1. **Update Core Dependencies**: Modify `pyproject.toml` first, then update `requirements-production.txt`
2. **Security Updates**: Run `pip-audit` regularly and update vulnerable packages
3. **Version Pinning**: Consider pinning versions in production for reproducibility
4. **Separate Files**: Keep dev and prod dependencies separate

## Notes

- Version constraints use `>=` for flexibility
- For production deployments, consider pinning exact versions for reproducibility
- Optional dependencies (Streamlit, Qt) can be excluded if not needed
- MCP server dependencies are optional and can be excluded if not using MCP

## Next Steps

1. ✅ **Production Requirements** - Created
2. ⏳ **Security Audit** - Run `pip-audit` regularly
3. ⏳ **Version Pinning** - Consider creating `requirements-production-locked.txt` with exact versions
4. ⏳ **CI/CD Integration** - Add security audit to CI pipeline
5. ⏳ **Dependency Updates** - Set up automated dependency update checks
