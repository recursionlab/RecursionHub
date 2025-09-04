# RecursionHub - Research Repository for Recursion Theory and Mathematical Frameworks

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

RecursionHub is an academic research repository focused on recursion theory, category theory, higher-order mathematics, and AI-optimized learning frameworks. The repository contains extensive theoretical documentation but minimal working code - the main Python files are currently placeholders.

## Working Effectively

### Environment Setup
- Verify Python environment: `python --version` (should show Python 3.12.3)
- Check pip: `pip --version`
- Install core development tools (takes 2-3 minutes): `pip install matplotlib pandas pytest flake8 black`
  - NEVER CANCEL: Package installation can take up to 5 minutes due to large dependencies like matplotlib and pandas
  - Set timeout to 10+ minutes when installing packages
- The repository includes a dependency "na" which requires matplotlib and pandas to function

### Repository Structure
- **Main Python files**: `app.py`, `knowledge_builder.py` (currently contain placeholder "delete" - do not run these)
- **Documentation**: `docs/` contains extensive academic PDFs on recursion theory, category theory, and mathematical frameworks
  - `docs/core/` - 20 main theoretical PDFs 
  - `docs/python/` - Python development resources
  - `docs/AI/`, `docs/strategy/`, `docs/theories/` - specialized research materials
- **Configuration**: 
  - `requirements.txt` contains "na" (installs successfully but requires matplotlib/pandas)
  - `.env.example` contains OPENROUTER_API_KEY placeholder
  - No build system, CI/CD, or traditional project structure

### Development Workflow
- **DO NOT** try to run `python app.py` or `python knowledge_builder.py` - they will fail with NameError
- Create new Python files for development work instead of modifying the placeholder files
- Basic Python development works normally:
  - Create scripts: `python your_script.py` (< 0.1 seconds)
  - Linting: `python -m flake8 file.py` (< 0.2 seconds)  
  - Formatting: `python -m black file.py` (< 0.2 seconds)
  - Testing: `python -m pytest test_file.py` (< 1 second for simple tests)

### Validation Steps
- Always run linting before committing: `python -m flake8 .`
- Format code: `python -m black .`
- Run tests if created: `python -m pytest -v`
- Validate Python syntax: `python -m py_compile your_file.py`

### Package Installation Issues
- The "na" package from requirements.txt has dependencies (matplotlib, pandas) not listed
- NEVER CANCEL package installations - they may take 5+ minutes due to large scientific libraries
- If pip install times out, this is normal - wait and retry
- Set explicit timeouts of 10+ minutes for any pip install commands

### Known Limitations and Workarounds
- **Main application files DO NOT WORK**: app.py and knowledge_builder.py contain only "delete"
- **No test suite exists**: pytest collects 0 items when run
- **No CI/CD pipeline**: No .github/workflows or build automation
- **Package installation can be slow**: Scientific packages take time, NEVER CANCEL
- **This is a research repository**: Focus is on theoretical documentation rather than running code

### Common Tasks and Expected Times
- **Python script execution**: < 0.1 seconds
- **Linting with flake8**: < 0.2 seconds  
- **Code formatting with black**: < 0.2 seconds
- **Running tests**: < 1 second (when tests exist)
- **Package installation**: 2-10 minutes, NEVER CANCEL, set timeout to 15+ minutes
- **Exploring documentation**: Use file browser for PDF resources in docs/

## Manual Validation Scenarios

Since this is a research repository with placeholder code, validation focuses on development environment functionality:

1. **Environment Validation**: Create a simple "Hello World" script, run it, lint it, format it, and test it
2. **Package Validation**: Verify that scientific packages (numpy, pandas, matplotlib) can be imported
3. **Documentation Access**: Confirm that PDFs in docs/ can be accessed for research

## Research Focus Areas

Based on the documentation, the repository covers:
- Recursion theory and higher-order recursive schemes
- Category theory and infinity-topologies  
- Mathematical frameworks for AI and consciousness
- Topos theory and mathematical structures in language
- Meta-recursive intelligence and autopoietic systems

## Quick Reference Commands

```bash
# Validate environment
python --version
pip --version

# Install development tools (NEVER CANCEL, 10+ minute timeout)
pip install matplotlib pandas pytest flake8 black

# Development workflow
python -m flake8 .                 # Lint code
python -m black .                  # Format code  
python -m pytest -v               # Run tests (when they exist)

# DO NOT RUN (will fail):
python app.py                      # NameError: name 'delete' is not defined
python knowledge_builder.py       # NameError: name 'delete' is not defined
```

## Directory Listing Reference
```
.
├── .env.example                   # API key template
├── .gitignore                    # Git ignore rules
├── README.md                     # Minimal description
├── app.py                        # PLACEHOLDER - contains "delete"
├── knowledge_builder.py          # PLACEHOLDER - contains "delete"  
├── requirements.txt              # Contains "na" package
├── first file                    # Empty file
├── Full Process for Coding...    # Mathematical operators checklist
└── docs/                         # Academic research documentation
    ├── core/                     # 20 main theoretical PDFs
    ├── python/                   # Python development books
    ├── AI/                       # AI research materials
    ├── strategy/                 # Strategic frameworks
    └── theories/                 # Theoretical foundations
```