# Release Versioning Scheme

**Version Format**: `MAJOR.MINOR.PATCH` (Semantic Versioning)

## Version Numbering

### Format

```
MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
```

Examples:
- `1.0.0` - Initial release
- `1.1.0` - Minor feature release
- `1.1.1` - Patch release
- `2.0.0-beta.1` - Beta pre-release
- `1.0.0+20250102` - Build metadata

### Version Components

#### MAJOR Version

Increment when you make incompatible API changes or breaking changes:

- Breaking API changes
- Major architecture changes
- Incompatible configuration changes
- Removal of deprecated features

**Examples**:
- `1.0.0` → `2.0.0`: Complete API redesign
- `1.0.0` → `2.0.0`: Breaking configuration format change

#### MINOR Version

Increment when you add functionality in a backwards compatible manner:

- New features
- New API endpoints
- New CLI commands
- Backwards-compatible enhancements
- New optional dependencies

**Examples**:
- `1.0.0` → `1.1.0`: New emotion presets
- `1.0.0` → `1.1.0`: New analysis features
- `1.0.0` → `1.1.0`: New GUI features

#### PATCH Version

Increment when you make backwards-compatible bug fixes:

- Bug fixes
- Security patches
- Performance improvements
- Documentation updates
- Dependency updates

**Examples**:
- `1.0.0` → `1.0.1`: Fix generation bug
- `1.0.0` → `1.0.1`: Security patch
- `1.0.0` → `1.0.1`: Performance optimization

### Pre-release Versions

Use for alpha, beta, and release candidates:

- `-alpha.1`, `-alpha.2`, ... - Early development
- `-beta.1`, `-beta.2`, ... - Beta testing
- `-rc.1`, `-rc.2`, ... - Release candidates

**Examples**:
- `1.0.0-alpha.1` - First alpha release
- `1.0.0-beta.1` - First beta release
- `1.0.0-rc.1` - First release candidate

### Build Metadata

Optional build information:

- `+20250102` - Build date
- `+abc123` - Git commit hash
- `+build.42` - Build number

**Examples**:
- `1.0.0+20250102` - Build from January 2, 2025
- `1.0.0+abc123` - Build from commit abc123

## Version Management

### Current Version

**Current Version**: `1.0.0`

Location: `pyproject.toml`, `api/main.py`, `kmidi_gui/__init__.py`

### Version Update Process

1. **Update Version Numbers**:
   ```bash
   # Update in all locations
   - pyproject.toml
   - api/main.py
   - kmidi_gui/__init__.py
   - docs (if version-specific)
   ```

2. **Create Git Tag**:
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```

3. **Update Changelog**:
   - Add entry to `CHANGELOG.md`
   - Document changes, fixes, and new features

4. **Create Release Notes**:
   - Use `docs/RELEASE_NOTES_TEMPLATE.md`
   - Generate release notes

### Version Bump Guidelines

#### When to Bump MAJOR

- Breaking API changes
- Major architecture changes
- Incompatible configuration
- Removal of features

#### When to Bump MINOR

- New features
- New endpoints/commands
- New optional dependencies
- Backwards-compatible enhancements

#### When to Bump PATCH

- Bug fixes
- Security patches
- Performance improvements
- Documentation updates

## Release Process

### Pre-Release

1. **Update Version**: Bump version number
2. **Update Changelog**: Document all changes
3. **Run Tests**: Ensure all tests pass
4. **Update Documentation**: Update relevant docs
5. **Create Release Branch**: `release/v1.0.0`

### Release

1. **Final Testing**: Comprehensive testing
2. **Create Tag**: Git tag for version
3. **Build Artifacts**: Create distributions
4. **Release Notes**: Generate and publish
5. **Merge to Main**: Merge release branch

### Post-Release

1. **Announcement**: Announce release
2. **Monitor**: Monitor for issues
3. **Hotfixes**: Address critical issues
4. **Documentation**: Update user docs

## Version History

### 1.0.0 (2025-01-02)

**Initial Release**

- Core music generation functionality
- Emotion-to-music mapping
- CLI interface
- Qt GUI
- Streamlit demo
- FastAPI service
- Health monitoring
- Comprehensive documentation

## Future Versions

### Planned for 1.1.0

- Additional emotion presets
- Enhanced GUI features
- Performance improvements
- Additional analysis tools

### Planned for 2.0.0

- Major API redesign (if needed)
- Advanced AI features
- Plugin system
- Multi-user support

## Version Compatibility

### API Compatibility

- **Same MAJOR version**: Fully compatible
- **Different MAJOR version**: May have breaking changes
- **Different MINOR version**: Backwards compatible, new features available
- **Different PATCH version**: Bug fixes only, fully compatible

### CLI Compatibility

- CLI commands remain stable within MAJOR version
- New commands added in MINOR versions
- Command behavior changes only in MAJOR versions

### Configuration Compatibility

- Configuration format stable within MAJOR version
- New options added in MINOR versions
- Format changes only in MAJOR versions

## Best Practices

1. **Follow Semantic Versioning**: Strict adherence to semver
2. **Document Changes**: Always update CHANGELOG.md
3. **Tag Releases**: Create git tags for all releases
4. **Test Before Release**: Comprehensive testing before tagging
5. **Communicate Changes**: Clear release notes for users

## References

- [Semantic Versioning 2.0.0](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
