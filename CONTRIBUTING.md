# Contributing to RecursionHub

We welcome contributions to RecursionHub! This document provides guidelines for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Quality](#code-quality)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Commit Message Guidelines](#commit-message-guidelines)

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/RecursionHub.git
   cd RecursionHub
   ```
3. Set up the development environment (see [Development Setup](#development-setup))
4. Create a new branch for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- A virtual environment tool (venv, conda, etc.)

### Local Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Upgrade pip and install the project in development mode:
   ```bash
   python -m pip install --upgrade pip
   pip install -e .[dev]
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

4. Verify the installation:
   ```bash
   python -c "import recursion_hub; print('Setup successful!')"
   ```

## Code Quality

We use several tools to maintain code quality. All of these are run automatically in CI and can be run locally.

### Running Quality Checks Locally

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Individual tools
ruff check .                    # Linting
ruff check . --fix             # Linting with auto-fix
black --check .                # Formatting check
black .                        # Auto-format
isort --check-only .           # Import sorting check
isort .                        # Fix import sorting
mypy .                         # Type checking
bandit -r .                    # Security scanning
```

### Code Style Guidelines

- **Line Length**: Maximum 88 characters (Black's default)
- **Imports**: Use isort with Black profile for import organization
- **Type Hints**: Use type hints for all public functions and methods
- **Docstrings**: Use Google-style docstrings for public APIs
- **Security**: Follow secure coding practices, use Bandit recommendations

### Pre-commit Hooks

Pre-commit hooks run automatically before each commit and will:
- Format code with Black
- Sort imports with isort
- Lint code with Ruff
- Check types with mypy
- Scan for security issues with Bandit
- Validate YAML, JSON, and TOML files
- Remove trailing whitespace and fix line endings

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov

# Run tests in verbose mode
pytest -xvs

# Run specific test file
pytest tests/test_specific.py

# Run tests matching a pattern
pytest -k "test_pattern"
```

### Test Guidelines

- Write tests for all new features and bug fixes
- Maintain test coverage above 80%
- Use descriptive test names that explain what is being tested
- Follow the AAA pattern (Arrange, Act, Assert)
- Use fixtures for common test setup
- Mock external dependencies

### Test Structure

```
tests/
├── conftest.py          # Shared fixtures
├── unit/               # Unit tests
│   ├── test_module1.py
│   └── test_module2.py
├── integration/        # Integration tests
│   └── test_workflows.py
└── fixtures/           # Test data files
    └── sample_data.json
```

## Pull Request Process

1. **Ensure CI passes**: All checks must pass before review
2. **Write clear descriptions**: Explain what changes you made and why
3. **Link related issues**: Reference any relevant issue numbers
4. **Update documentation**: Include any necessary documentation updates
5. **Add tests**: Include tests for new functionality
6. **Review checklist**: Complete the PR template checklist

### PR Title Format

Use conventional commit format:
- `feat: add new feature`
- `fix: resolve bug in component`
- `docs: update contributing guidelines`
- `refactor: improve code structure`
- `test: add missing tests`
- `ci: update workflow configuration`

### Review Process

1. Automated checks run on your PR
2. Maintainers review your code
3. Address any requested changes
4. Once approved, your PR will be merged

## Issue Guidelines

### Before Opening an Issue

- Search existing issues to avoid duplicates
- Check if your issue is already addressed in the documentation
- For bugs, try to reproduce with the latest version

### Bug Reports

Include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)
- Relevant error messages or logs

### Feature Requests

Include:
- Clear description of the desired feature
- Use case or motivation for the feature
- Possible implementation suggestions
- Consider if this fits the project scope

### Security Issues

**Do not open public issues for security vulnerabilities.**
Please follow our [Security Policy](SECURITY.md) for responsible disclosure.

## Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, missing semi-colons, etc.)
- `refactor`: Code changes that neither fixes a bug nor adds a feature
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools
- `ci`: Changes to CI configuration files and scripts

### Scopes
- `core`: Core functionality
- `scripts`: GitHub scripts
- `ci`: CI/CD workflows
- `docs`: Documentation
- `deps`: Dependencies

### Examples
```
feat(core): add recursive pattern detection
fix(scripts): resolve knot detection edge case
docs: update API documentation
ci: add security scanning workflow
```

## Development Workflow

1. **Pull latest changes**:
   ```bash
   git checkout main
   git pull upstream main
   ```

2. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature
   ```

3. **Make changes and commit**:
   ```bash
   git add .
   git commit -m "feat: your feature description"
   ```

4. **Run tests and quality checks**:
   ```bash
   pytest
   pre-commit run --all-files
   ```

5. **Push and create PR**:
   ```bash
   git push origin feature/your-feature
   # Create PR via GitHub UI
   ```

## Getting Help

- **Documentation**: Check our [docs](docs/) directory
- **Issues**: Open an issue for bugs or feature requests  
- **Discussions**: Use GitHub Discussions for questions
- **CI Documentation**: See [docs/ci.md](docs/ci.md) for CI-specific help

## Recognition

Contributors who make significant improvements to the project will be recognized in our changelog and README.

Thank you for contributing to RecursionHub! 🚀