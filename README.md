# Instagram Message Analyzer

[![CI](https://github.com/peppapig450/insta-message-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/peppapig450/insta-message-analyzer/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/insta-message-analyzer.svg)](https://pypi.org/project/insta-message-analyzer/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/insta-message-analyzer.svg)](https://pypi.org/project/insta-message-analyzer)

Instagram Message Analyzer is a Python toolkit for exploring your archived Instagram conversations. It loads message exports from Meta, processes them into structured data, and runs several analysis strategies to reveal messaging patterns and network relationships. Optional visualizations are produced using Plotly.

## Features

- **Data loading and preprocessing** – Converts Meta JSON exports into a single DataFrame with normalized sender names, timestamps and reactions.
- **Activity analysis** – Computes time‑series metrics, burst detection, top senders and chat lifecycle statistics.
- **Network analysis** – Builds a bipartite graph of senders and chats to measure centrality, communities, influence and reactions.
- **Visualization** – Generates interactive HTML plots for message frequency, active hours, top senders and more.
- **Pipeline orchestration** – Modular `AnalysisPipeline` class runs multiple strategies and saves results to disk.
- **Command‑line interface** – `insta-analyze` entry point for quick analysis of a folder containing exported messages.

## Installation

Install from PyPi:

```bash
pip install insta-message-analyzer
```

Python 3.12 or 3.13 is required. The project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
poetry install --with dev
```

To install from source without the development tools, run `poetry install` without the `--with dev` flag. You may also install the package with pip:

```bash
pip install .
```

## Usage

1. Export your Instagram data from Meta and extract the archive so that your message JSON files reside under `data/your_instagram_activity/messages`.
2. Run the analysis pipeline:

```bash
poetry run insta-analyze
```

Results are written to the `output` directory. Logs are stored in `output/logs/insta_analyzer.log`.

You can also run the main module directly:

```bash
python -m insta_message_analyzer.main
```

## Running Tests

The repository contains a test suite for the network analysis component. Execute it with [pytest](https://pytest.readthedocs.io/):

```bash
poetry run pytest
```

Linting and type checking are performed via [Ruff](https://docs.astral.sh/ruff/) and [MyPy](https://mypy-lang.org/). A pre‑commit configuration is provided and can be enabled with:

```bash
pre-commit install
```

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub. Before submitting code, run the pre‑commit hooks and ensure all tests pass.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
