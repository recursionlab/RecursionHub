# RecursionHub

[![CI](https://github.com/recursionlab/RecursionHub/workflows/CI/badge.svg)](https://github.com/recursionlab/RecursionHub/actions/workflows/ci.yml)
[![Lint](https://github.com/recursionlab/RecursionHub/workflows/Lint/badge.svg)](https://github.com/recursionlab/RecursionHub/actions/workflows/lint.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Advanced AI/ASI research hub with automation and recursive processes for responsible AI development.

## 🎯 Overview

RecursionHub is a research-oriented project focused on advanced AI and ASI (Artificial Superintelligence) research with an emphasis on:

- **Safety First**: Responsible AI development with built-in safety measures
- **Automation**: GitHub-based automation for code quality and maintenance  
- **Research Excellence**: Tools and utilities for advanced AI research
- **Collaboration**: Open research with proper oversight and ethics

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/recursionlab/RecursionHub.git
cd RecursionHub

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run specific tests
pytest tests/test_knot_detector.py
```

### Code Quality

```bash
# Run all quality checks
pre-commit run --all-files

# Individual tools
ruff check .      # Linting
black .           # Formatting
mypy .            # Type checking
bandit -r .       # Security scanning
```

## 📚 Documentation

- [Contributing Guide](CONTRIBUTING.md) - How to contribute to the project
- [CI/CD Documentation](docs/ci.md) - Continuous integration setup and usage
- [Security Policy](SECURITY.md) - Security guidelines and reporting
- [Code of Conduct](CODE_OF_CONDUCT.md) - Community guidelines

## 🔧 Features

### Automation Scripts

Located in `.github/scripts/`:

- **knot_detector.py**: Detects duplicate PR titles to identify workflow loops
- **seal_on_close.py**: Automated actions when PRs are closed  
- **compute_metrics.py**: Repository and development metrics computation

### CI/CD Pipeline

- **Comprehensive Testing**: Multi-version Python testing (3.10, 3.11, 3.12)
- **Code Quality**: Automated linting with Ruff, formatting with Black
- **Security**: Bandit security scanning, CodeQL analysis
- **Dependencies**: Automated dependency updates with Dependabot
- **Maintenance**: Scheduled repository health checks

### Development Tools

- **Pre-commit Hooks**: Automated code quality checks before commits
- **Type Safety**: Full mypy type checking
- **Testing**: Comprehensive test suite with pytest
- **Documentation**: Automated documentation generation

## 🛡️ Security

This project deals with advanced AI research and emphasizes security:

- No secrets committed to the repository
- Automated security scanning with multiple tools
- Regular dependency vulnerability checks
- Responsible disclosure procedures
- Access control and review requirements

See our [Security Policy](SECURITY.md) for details on reporting vulnerabilities.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for:

- Development setup instructions
- Code quality requirements
- Testing guidelines
- Pull request process
- Issue reporting

### Quick Contribution Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and add tests
4. Run quality checks: `pre-commit run --all-files`
5. Commit and push changes
6. Create a pull request

## 📊 Project Status

- **Version**: 0.1.0 (Alpha)
- **Python**: 3.10+ supported
- **License**: MIT
- **Status**: Active development

## 🏗️ Architecture

```
RecursionHub/
├── .github/
│   ├── workflows/          # GitHub Actions CI/CD
│   ├── scripts/           # Automation scripts
│   └── ISSUE_TEMPLATE/    # Issue templates
├── recursion_hub/         # Main Python package
├── tests/                # Test suite
├── docs/                 # Documentation
└── pyproject.toml        # Project configuration
```

## 📈 Development Workflow

Our development process emphasizes quality and security:

1. **Branch Protection**: Main branch requires reviews and passing CI
2. **Automated Testing**: All PRs run comprehensive test suites
3. **Code Quality**: Automated linting, formatting, and type checking
4. **Security Scanning**: Multiple security tools protect against vulnerabilities
5. **Regular Maintenance**: Weekly repository health checks

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/recursionlab/RecursionHub/issues)
- **Discussions**: [GitHub Discussions](https://github.com/recursionlab/RecursionHub/discussions)
- **Security**: security@recursionlab.org
- **General**: info@recursionlab.org

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with modern Python development best practices
- Automated CI/CD with GitHub Actions
- Security-first approach to AI research
- Community-driven development

---

**Note**: This project is in active development. APIs and interfaces may change as we refine the research tools and automation systems.
