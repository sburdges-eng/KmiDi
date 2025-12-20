# Contributing to Kelly MIDI Companion

Thank you for your interest in contributing! This guide will help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Submission Guidelines](#submission-guidelines)
- [Code Review Process](#code-review-process)
- [Project Philosophy](#project-philosophy)

## Code of Conduct

This project follows a code of conduct. By participating, you agree to maintain a respectful and inclusive environment.

## Getting Started

### Prerequisites

1. Read the [BUILD.md](BUILD.md) guide
2. Set up development environment (see [DEVELOPMENT.md](DEVELOPMENT.md))
3. Familiarize yourself with the project structure
4. Review existing issues and discussions

### First Contribution

Good first issues are labeled with `good-first-issue`. These are typically:

- Documentation improvements
- Bug fixes
- Small feature additions
- Test improvements

## Development Process

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/yourusername/kelly-midi-companion.git
cd kelly-midi-companion
```

### 2. Create a Branch

```bash
git checkout -b feature/my-feature
# or
git checkout -b fix/bug-description
```

Branch naming:

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation
- `refactor/` - Code refactoring
- `test/` - Test improvements

### 3. Make Changes

- Follow code style guidelines (see [DEVELOPMENT.md](DEVELOPMENT.md))
- Write tests for new features
- Update documentation as needed
- Keep commits focused and atomic

### 4. Test Your Changes

```bash
# Build and test
./build_all.sh --clean --test

# Run specific tests
pytest tests_music-brain/ -v -k "test_name"
cd build && ctest --output-on-failure
```

### 5. Commit Changes

Write clear, descriptive commit messages:

```
Short summary (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.
Explain what and why, not how.

- Bullet points for multiple changes
- Reference issues: Fixes #123
```

Example:

```
Add groove quantization to RhythmEngine

Implements quantize_to_grid() method that snaps MIDI notes
to the nearest grid position based on time signature.

Fixes #456
```

### 6. Push and Create Pull Request

```bash
git push origin feature/my-feature
```

Then create a Pull Request on GitHub with:

- Clear title and description
- Reference related issues
- List changes made
- Screenshots/videos if UI changes

## Submission Guidelines

### Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] Commit messages are clear
- [ ] Code is commented where needed

### Code Quality

**C++ Code**:

- Follows C++20 standard
- Uses consistent naming conventions
- Includes appropriate comments
- No memory leaks (check with Valgrind)
- Real-time safe where required

**Python Code**:

- Follows PEP 8
- Includes type hints
- Has docstrings
- Passes linting (ruff, black)

**Tests**:

- New features have tests
- Tests are clear and focused
- Edge cases are covered
- Tests are fast and reliable

### Documentation

Update documentation for:

- New features
- API changes
- Configuration changes
- Build system changes

Files to update:

- `README.md` - Overview and quick start
- `BUILD.md` - Build instructions
- `DEVELOPMENT.md` - Development guide
- Code comments - Inline documentation
- API docs - If adding public APIs

## Code Review Process

### What Reviewers Look For

1. **Correctness**: Does it work as intended?
2. **Style**: Follows project conventions?
3. **Tests**: Adequate test coverage?
4. **Documentation**: Is it documented?
5. **Performance**: Any performance concerns?
6. **Security**: Any security issues?

### Responding to Feedback

- Be open to suggestions
- Ask questions if unclear
- Make requested changes
- Discuss alternatives if needed
- Thank reviewers for their time

### Review Checklist for Authors

Before requesting review:

- [ ] All tests pass
- [ ] Code is formatted
- [ ] Documentation is updated
- [ ] PR description is complete
- [ ] Ready for review (not WIP)

## Project Philosophy

### Core Principles

1. **"Interrogate Before Generate"** - The tool shouldn't finish art for people; it should make them braver.

2. **Emotional Truth Over Technical Perfection** - Sometimes the "wrong" note is exactly right.

3. **Real-Time Safety First** - Audio processing must never glitch or drop.

4. **Documentation Matters** - Code should be self-documenting, but docs help.

### Design Guidelines

- **Simplicity**: Prefer simple solutions
- **Clarity**: Code should be readable
- **Performance**: Optimize where it matters (audio thread)
- **Maintainability**: Future developers should understand it

### Areas for Contribution

**High Priority**:

- Bug fixes
- Performance improvements
- Test coverage
- Documentation

**Medium Priority**:

- New engines/features
- UI improvements
- ML model improvements
- Cross-platform support

**Lower Priority**:

- Code style improvements
- Refactoring
- Additional examples

## Specific Contribution Areas

### C++ Development

- Audio processing engines
- MIDI generation
- Real-time performance
- JUCE plugin development

### Python Development

- ML framework
- Music analysis tools
- Training pipelines
- Example scripts

### Documentation

- User guides
- API documentation
- Tutorials
- Code comments

### Testing

- Unit tests
- Integration tests
- Performance tests
- Test infrastructure

## Getting Help

### Questions?

- Check existing documentation
- Search issues and discussions
- Ask in project discussions
- Create an issue for bugs

### Stuck?

- Review similar code in the project
- Check test examples
- Ask for help in discussions
- Pair with a maintainer if possible

## Recognition

Contributors will be:

- Listed in CONTRIBUTORS.md (if file exists)
- Credited in release notes
- Appreciated by the community

Thank you for contributing to Kelly MIDI Companion!
