# CI/CD Documentation

This document provides comprehensive information about our Continuous Integration and Continuous Deployment (CI/CD) setup for RecursionHub.

## Overview

RecursionHub uses GitHub Actions for automated testing, linting, security scanning, and maintenance. Our CI/CD pipeline ensures code quality, security, and reliability across all contributions.

## Workflows

### 1. Main CI Pipeline (`ci.yml`)

**Trigger**: Push to `main`, Pull Requests to `main`  
**Purpose**: Comprehensive testing and quality checks

#### Jobs:
- **Test**: Multi-version Python testing (3.10, 3.11, 3.12)
- **Security**: CodeQL security analysis
- **Lint**: Code quality and formatting checks
- **Build**: Package build verification

#### Key Features:
- Parallel execution for faster feedback
- Dependency caching for improved performance
- Comprehensive test coverage reporting
- Security scanning with Bandit
- Type checking with mypy
- Code quality checks with Ruff and Black

### 2. Lint Workflow (`lint.yml`)

**Trigger**: Pull Requests  
**Purpose**: Fast feedback on code quality

#### Checks:
- Ruff linting
- Black formatting
- isort import sorting
- mypy type checking
- Bandit security linting
- Pre-commit hooks

### 3. Scheduled Maintenance (`scheduled-maintenance.yml`)

**Trigger**: Weekly (Mondays at 6 AM UTC), Manual  
**Purpose**: Repository health monitoring

#### Activities:
- Full test suite execution
- Security audit with pip-audit
- Dependency update checks
- Artifact cleanup
- Stale issue/PR management

### 4. Dependency Updates (`dependency-updates.yml`)

**Trigger**: Weekly (Mondays at 2 AM UTC), Manual  
**Purpose**: Automated dependency security monitoring

#### Features:
- Security vulnerability scanning
- Automated issue creation for critical vulnerabilities
- Integration with Dependabot for updates

## Local Development

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/recursionlab/RecursionHub.git
cd RecursionHub

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
python -m pip install --upgrade pip
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### Running Tests Locally

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov

# Run tests in verbose mode with output
pytest -xvs

# Run specific test file
pytest tests/test_specific.py

# Generate coverage report
pytest --cov --cov-report=html
open htmlcov/index.html  # View coverage report
```

### Code Quality Checks

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Individual tools
ruff check .                    # Linting
ruff check . --fix             # Auto-fix issues
black --check .                # Formatting check
black .                        # Auto-format
isort --check-only .           # Import sorting check
isort .                        # Fix import sorting
mypy .                         # Type checking
bandit -r .                    # Security scanning
```

### Development Workflow

```bash
# Start development
git checkout main
git pull origin main
git checkout -b feature/your-feature

# Make changes and run checks
# ... edit files ...
pytest                         # Run tests
pre-commit run --all-files     # Run quality checks

# Commit and push
git add .
git commit -m "feat: your feature description"
git push origin feature/your-feature

# Create pull request via GitHub UI
```

## CI Configuration

### Environment Variables

Our CI workflows use the following environment variables:

#### Secrets (configured in GitHub repository settings):
- `CODECOV_TOKEN`: Token for Codecov coverage reporting (optional)
- `GITHUB_TOKEN`: Automatically provided by GitHub Actions

#### Configuration:
- Python versions: 3.10, 3.11, 3.12 (defined in ci.yml matrix)
- Test coverage threshold: 80% (configurable in pyproject.toml)
- Security scan level: High (configurable in workflow files)

### Caching Strategy

We use GitHub Actions caching to improve performance:
- **pip cache**: Caches Python package installations
- **pre-commit cache**: Caches pre-commit hook installations
- Cache keys include Python version and dependency file hashes

### Artifact Management

Our workflows generate and store the following artifacts:
- **Test Results**: JUnit XML and coverage reports
- **Security Reports**: Bandit and pip-audit JSON reports
- **Build Artifacts**: Python wheel and source distributions

Artifacts are retained for 90 days and automatically cleaned up by scheduled maintenance.

## Required Secrets

Configure these secrets in your GitHub repository settings:

### Optional Secrets:
- `CODECOV_TOKEN`: For uploading coverage reports to Codecov
  - Go to [Codecov.io](https://codecov.io/)
  - Sign up/login with GitHub
  - Add the RecursionHub repository
  - Copy the upload token
  - Add as repository secret

### Automatic Secrets:
- `GITHUB_TOKEN`: Automatically provided (no configuration needed)

## Branch Protection

We recommend configuring the following branch protection rules for `main`:

### Required Status Checks:
- `test (3.10)`
- `test (3.11)`
- `test (3.12)`
- `lint`
- `security`
- `build`

### Settings:
- ✅ Require a pull request before merging
- ✅ Require approvals (1 minimum)
- ✅ Dismiss stale PR approvals when new commits are pushed
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Include administrators

## Dependabot Configuration

Dependabot is configured via `.github/dependabot.yml`:

### Update Schedule:
- **Python dependencies**: Weekly on Monday at 2 AM UTC
- **GitHub Actions**: Weekly on Monday at 3 AM UTC

### Settings:
- Maximum 10 open PRs for Python dependencies
- Maximum 5 open PRs for GitHub Actions
- Automatic assignment to maintainers
- Proper labeling for easy identification

## Security Scanning

### Tools Used:
- **CodeQL**: GitHub's semantic code analysis
- **Bandit**: Python security linter
- **pip-audit**: Python dependency vulnerability scanner
- **Dependabot**: Automated dependency security alerts

### Security Workflow:
1. Automated scanning on every PR and push
2. Security issues create automated GitHub issues
3. Critical vulnerabilities trigger immediate notifications
4. Regular security audits via scheduled maintenance

## Performance Optimization

### CI Performance Tips:
- Use dependency caching
- Run linting before expensive tests
- Use matrix builds for parallel execution
- Cleanup old workflow runs automatically

### Local Performance:
- Use virtual environments
- Install only development dependencies locally
- Use `pytest -x` to stop on first failure during development
- Use `--lf` flag to run only previously failed tests

## Troubleshooting

### Common Issues:

#### "Tests Failed"
```bash
# Run locally to debug
pytest -xvs
# Check specific test
pytest tests/test_name.py::test_function -v
```

#### "Linting Failed"
```bash
# Run and fix automatically
ruff check . --fix
black .
isort .
```

#### "Type Checking Failed"
```bash
# Run mypy locally
mypy .
# Check specific file
mypy path/to/file.py
```

#### "Security Scan Failed"
```bash
# Run bandit locally
bandit -r .
# Check specific issue
bandit -r . -f json | jq
```

#### "Pre-commit Hook Failed"
```bash
# Update hooks
pre-commit autoupdate
# Run specific hook
pre-commit run black --all-files
```

### Getting Help:

1. **Check workflow logs**: Go to Actions tab in GitHub repository
2. **Review error messages**: Look for specific failure reasons
3. **Run locally**: Reproduce the issue on your development machine
4. **Ask for help**: Create an issue or discussion in the repository

## Best Practices

### For Contributors:
- Run tests locally before pushing
- Use pre-commit hooks to catch issues early
- Write descriptive commit messages
- Keep PRs focused and reasonably sized
- Update documentation when needed

### For Maintainers:
- Review CI logs for security issues
- Monitor dependency update PRs
- Keep workflows updated with latest Actions versions
- Respond to automated security issues promptly
- Maintain branch protection rules

## Metrics and Monitoring

### Available Metrics:
- **Test Coverage**: Target >80%, displayed in PR comments
- **Build Success Rate**: Monitored via GitHub Insights
- **Security Issues**: Tracked via automated issues
- **Dependency Freshness**: Monitored weekly

### Accessing Metrics:
- **Coverage**: View in PR comments and artifacts
- **Build Status**: GitHub Actions tab
- **Security**: Security tab in repository
- **Dependencies**: Dependabot tab in repository

## Migration Guide

### From Other CI Systems:

If migrating from other CI systems:
1. Review existing test commands and configurations
2. Adapt build matrices for GitHub Actions
3. Migrate environment variables to GitHub Secrets
4. Update documentation and README badges
5. Configure branch protection rules
6. Test thoroughly with sample PRs

## Updates and Maintenance

### Keeping CI Updated:
- **Monthly**: Review and update action versions
- **Quarterly**: Review Python versions and dependencies
- **As Needed**: Update security scanning rules
- **When Issues Occur**: Debug and fix workflow problems

### Changelog:
- All CI changes are documented in commit messages
- Major changes are announced in repository discussions
- Breaking changes require maintainer approval

---

## Quick Reference

### Essential Commands:
```bash
# Setup
pip install -e .[dev]
pre-commit install

# Testing
pytest
pytest --cov

# Quality
pre-commit run --all-files
ruff check . --fix
black .
mypy .

# Development
git checkout -b feature/name
# ... make changes ...
git commit -m "feat: description"
git push origin feature/name
```

### Useful Links:
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [pytest Documentation](https://docs.pytest.org/)
- [pre-commit Documentation](https://pre-commit.com/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Black Documentation](https://black.readthedocs.io/)

For questions about CI/CD, create an issue with the `ci` label.