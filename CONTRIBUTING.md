# Contributing to SCRUMBENCH

Thank you for your interest in contributing to SCRUMBENCH! This document provides guidelines and instructions for contributing.

## 🌟 Ways to Contribute

- **Bug Reports**: Found a bug? Open an issue with detailed reproduction steps
- **Feature Requests**: Have an idea? Propose it in GitHub Discussions first
- **Code Contributions**: Submit pull requests for bug fixes or new features
- **Documentation**: Improve docs, add examples, fix typos
- **Templates**: Add new benchmark templates or domains
- **Testing**: Write tests, improve coverage

## 🚀 Getting Started

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/FedEUAI/SCRUMBENCH.git
cd scrumbench

# Create a virtual environment
uv sync

# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Run linter
uv run ruff check .
uv run black --check .
```

### Project Structure

```
scrumbench/
├── core/           # Core benchmark logic
│   ├── benchmark.py    # Task templates and domains
│   ├── runner.py       # Experiment execution
│   ├── harness.py      # Agent interaction framework
│   └── integration.py  # Testing and scoring
├── api.py          # FastAPI dashboard backend
├── index.html      # Dashboard frontend
├── docs/           # Documentation
├── examples/       # Example configurations
└── tests/          # Test suite
```

## 📝 Code Style

### Python

- **Formatter**: Black (line length: 100)
- **Linter**: Ruff
- **Type Hints**: Required for all public APIs
- **Docstrings**: Google style for all public functions/classes

```python
def calculate_ics(unit_pass: int, unit_total: int, 
                  int_pass: int, int_total: int) -> float:
    """Calculate Integration Competence Score.
    
    Args:
        unit_pass: Number of passing unit tests
        unit_total: Total number of unit tests
        int_pass: Number of passing integration tests
        int_total: Total number of integration tests
        
    Returns:
        ICS score between 0.0 and 1.0
        
    Example:
        >>> calculate_ics(8, 10, 3, 4)
        0.85
    """
    # Implementation...
```

### JavaScript/HTML

- **Formatter**: Prettier
- **Style**: Modern ES6+, functional components for React
- **Comments**: JSDoc for complex functions

## 🧪 Testing

### Writing Tests

```python
# tests/test_benchmark.py
import pytest
from core.benchmark import ServiceClientTemplate

def test_service_client_generation():
    """Test that ServiceClientTemplate generates valid tasks."""
    template = ServiceClientTemplate()
    task = template.generate({"domain": "user_cache"}, seed=42)
    
    assert task.instance_id.startswith("service_client")
    assert len(task.starter_files) > 0
    assert len(task.unit_tests) > 0
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=core --cov-report=html

# Run specific test file
uv run pytest tests/test_benchmark.py

# Run with verbose output
uv run pytest -v
```

## 🎨 Adding New Templates

### 1. Create Template Class

```python
# core/benchmark.py
class MyCustomTemplate(TaskTemplate):
    """Template for testing [specific scenario].
    
    This template evaluates agents on [key capability].
    """
    
    name = "my_custom"
    description = "Brief description"
    
    def get_default_params(self) -> dict:
        return {"param1": "default_value"}
    
    def get_param_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string"}
            }
        }
    
    def generate(self, params: dict, seed: int) -> TaskInstance:
        # Generate task instance
        pass
```

### 2. Register Template

```python
# core/benchmark.py - TaskGenerator class
def __init__(self):
    self.templates = {
        # ... existing templates
        "my_custom": MyCustomTemplate(),
    }
```

### 3. Add Tests

```python
# tests/test_my_custom.py
def test_my_custom_template():
    template = MyCustomTemplate()
    task = template.generate({}, seed=42)
    # Assertions...
```

### 4. Document Template

Add to `docs/templates.md` with:
- Purpose and use case
- Parameters and options
- Example configuration
- Expected agent behavior

## 🌍 Adding New Domains

### 1. Define Domain Specification

```python
# core/benchmark.py - DOMAINS dict
DOMAINS["my_domain"] = DomainSpec(
    name="my_domain",
    data_type_name="MyData",
    fields=[
        ("id", "str", "Unique identifier"),
        ("value", "int", "Numeric value"),
    ],
    operations=[
        {
            "name": "create",
            "params": [("value", "int")],
            "returns": "MyData",
            "description": "Create new instance"
        },
    ]
)
```

### 2. Add Domain-Specific Tests

Generate appropriate unit and integration tests for the domain's operations.

### 3. Document Domain

Add to `docs/domains.md` with domain semantics and constraints.

## 📥 Pull Request Process

### Before Submitting

1. **Create an issue** describing the change (for non-trivial changes)
2. **Fork the repository** and create a feature branch
3. **Write tests** for new functionality
4. **Update documentation** if adding features
5. **Run linters** and ensure tests pass
6. **Write clear commit messages**

### Commit Message Format

```
type(scope): brief description

Detailed explanation of what changed and why.

Fixes #123
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Examples**:
```
feat(benchmark): add graph traversal template
fix(runner): correct checkpoint resume logic
docs(readme): update installation instructions
```

### PR Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (for significant changes)
- [ ] PR description explains changes clearly
- [ ] Linked to related issue(s)

### Review Process

1. Maintainers will review within 3-5 business days
2. Address review feedback
3. Once approved, maintainer will merge
4. Your contribution will be acknowledged in release notes!

## 🐛 Reporting Bugs

### Bug Report Template

```markdown
**Description**
Clear description of the bug

**To Reproduce**
1. Step 1
2. Step 2
3. See error

**Expected Behavior**
What should happen

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.11.5]
- SCRUMBENCH version: [e.g., 0.1.0]

**Additional Context**
Logs, screenshots, etc.
```

## 💡 Feature Requests

Use GitHub Discussions for feature requests. Include:
- **Use case**: Why is this needed?
- **Proposed solution**: How should it work?
- **Alternatives**: Other approaches considered
- **Impact**: Who benefits from this?

## 📜 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors.

### Our Standards

**Positive behavior**:
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community

**Unacceptable behavior**:
- Harassment, trolling, or discriminatory comments
- Publishing others' private information
- Other conduct inappropriate in a professional setting

### Enforcement

Violations may result in temporary or permanent ban from the project.

## 🎓 Learning Resources

- **Python Best Practices**: [Real Python](https://realpython.com/)
- **Testing**: [pytest documentation](https://docs.pytest.org/)
- **Type Hints**: [mypy documentation](https://mypy.readthedocs.io/)
- **FastAPI**: [FastAPI documentation](https://fastapi.tiangolo.com/)

## 📞 Getting Help

- **Questions**: Use [GitHub Discussions](https://github.com/FedEUAI/SCRUMBENCH/discussions)
- **Chat**: Join our [Discord server](https://discord.gg/scrumbench)
- **Email**: maintainers@scrumbench.org

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Acknowledged in academic papers (for significant contributions)

---

Thank you for contributing to SCRUMBENCH! 🚀
