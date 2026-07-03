# Contributing to Face Mask Detection

Thank you for your interest in contributing! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs
- Check if the bug has already been reported in the Issues
- Create a clear issue with a descriptive title and detailed description
- Include code samples, error messages, and steps to reproduce

### Suggesting Enhancements
- Use the Issues tab to suggest new features
- Be descriptive about the use case and expected behavior
- Include references or examples if applicable

### Pull Requests
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes with clear, descriptive commits
4. Ensure all tests pass: `python -m unittest discover -s tests`
5. Follow the project's coding style (use `black` for formatting)
6. Submit a pull request with a clear description of your changes

## Code Style

- Use **black** for code formatting: `black src tests`
- Use **flake8** for linting: `flake8 src tests`
- Add type hints where possible
- Write docstrings for functions and classes

## Testing

- Add tests for any new functionality
- Run the full test suite before submitting: `python -m unittest discover -s tests -p 'test_*.py'`
- Aim for high test coverage

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Questions?

Feel free to open an issue or discussion if you have questions about the project or development process.

Thank you for contributing!
