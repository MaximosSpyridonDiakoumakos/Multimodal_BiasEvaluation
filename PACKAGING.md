# Modern Python Packaging Guide

This project uses modern Python packaging standards with `pyproject.toml` as the primary configuration file.

## 🚀 Quick Start

### Prerequisites
```bash
pip install build twine
```

### Build Package
```bash
python -m build
```

### Upload to PyPI
```bash
# First, upload to TestPyPI to test
python -m twine upload --repository testpypi dist/*

# Then upload to PyPI
python -m twine upload dist/*
```

## 📦 Package Configuration

### Key Files
- **`pyproject.toml`**: Main package configuration (replaces `setup.py`)
- **`requirements.txt`**: Development dependencies (not used for package)

### Modern vs Legacy Approach

| Modern (pyproject.toml) | Legacy (setup.py) |
|------------------------|-------------------|
| ✅ Single source of truth | ❌ Duplicate configuration |
| ✅ Standardized format | ❌ Python code execution |
| ✅ Better dependency resolution | ❌ Potential security issues |
| ✅ Tool integration | ❌ Limited tool support |
| ✅ No build scripts needed | ❌ Requires setup.py script |

## 🔧 Configuration Details

### Build System
```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"
```

### Project Metadata
```toml
[project]
name = "multimodal-bias-evaluation"
version = "1.0.0"
description = "A comprehensive toolkit for evaluating bias in multimodal AI systems"
readme = "README.md"
license = {text = "Apache"}
```

### Dependencies
```toml
dependencies = [
    "torch>=1.9.0",
    "transformers>=4.20.0",
    # ... other dependencies
]
```

### Optional Dependencies
```toml
[project.optional-dependencies]
dev = [
    "pytest>=6.0",
    "black>=21.0",
    # ... development tools
]
```

## 🛠️ Development Workflow

### 1. Install in Development Mode
```bash
pip install -e .
```

### 2. Install with Development Dependencies
```bash
pip install -e ".[dev]"
```

### 3. Build Package
```bash
python -m build
```

### 4. Check Package
```bash
python -m twine check dist/*
```

### 5. Test Package Locally
```bash
pip install dist/multimodal_bias_evaluation-1.0.0.tar.gz
```

### 6. Upload to PyPI
```bash
# Test on TestPyPI first
python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*
```

## 📋 Package Structure

```
multimodal-bias-evaluation/
├── pyproject.toml          # Main configuration
├── requirements.txt         # Dev dependencies
├── README.md              # Package description
├── LICENSE                # License file
├── main.py                # Main entry point
├── __init__.py            # Package initialization
├── text_to_image.py       # Text-to-image evaluation
├── image_to_text.py       # Image-to-text evaluation
├── prompts_config.py      # Configuration
└── evaluationFunctions/   # Evaluation modules
    ├── __init__.py
    ├── evaluation_functions.py
    ├── gender_config.py
    └── visualization.py
```

## 🔍 Package Discovery

The package uses setuptools for discovery:
```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["*"]
exclude = ["tests*", "docs*"]
```

## 📦 Package Data

Include additional files:
```toml
[tool.setuptools.package-data]
"*" = ["*.md", "*.txt", "*.yml", "*.yaml"]
```

## 🚀 Entry Points

Console script entry point:
```toml
[project.scripts]
multimodal-bias-eval = "main:main"
```

## 🔧 Tools Configuration

### Code Formatting (Black)
```toml
[tool.black]
line-length = 88
target-version = ['py38']
```

### Type Checking (MyPy)
```toml
[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

## 🎯 Complete Workflow Example

```bash
# 1. Clean previous builds
rm -rf dist/ build/ *.egg-info/

# 2. Build package
python -m build

# 3. Check package
python -m twine check dist/*

# 4. Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# 5. Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ multimodal-bias-evaluation

# 6. Upload to PyPI
python -m twine upload dist/*
```

## 📚 Additional Resources

- [PEP 518](https://www.python.org/dev/peps/pep-0518/) - Build system requirements
- [PEP 621](https://www.python.org/dev/peps/pep-0621/) - Storing project metadata in pyproject.toml
- [Python Packaging User Guide](https://packaging.python.org/)
- [setuptools documentation](https://setuptools.pypa.io/)

## ⚠️ Migration Notes

- ✅ Removed `setup.py` (no longer needed)
- ✅ Removed build scripts (use standard tools instead)
- ✅ All configuration moved to `pyproject.toml`
- ✅ Modern build system using `setuptools.build_meta`
- ✅ Standardized dependency specification
- ✅ Better tool integration support
- ✅ **No Python scripts needed for building/uploading** 