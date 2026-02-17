# Git-Derived Software Knowledge Graph (GDSKG)

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tree-sitter](https://img.shields.io/badge/parsing-tree--sitter-orange.svg)](https://tree-sitter.github.io/tree-sitter/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GDSKG** is a high-fidelity intelligence tool designed to map the deep evolutionary relationships within software repositories. By combining Git history with AST-level code analysis via Tree-sitter, GDSKG builds a navigable knowledge graph that transcends simple text search.

---

## Key Capabilities

### Semantic Code Intelligence
- **Intelligent Symbol Extraction**: Automatically identifies function definitions, class structures, and imports across your codebase.
- **Context-Aware Diffing**: Uses a two-pass analysis to determine which symbols were modified in a commit and which were merely referenced, providing precise impact mapping.
- **Language Support**: First-class support for **Python**, **TypeScript**, and **TSX/JSX**.

### Graph-Based Exploration
- **Recursive Traversal**: Query connections at depth (N) to find indirect relationships (e.g., "Find all commits that modified symbols used by this bug fix").
- **Smart Noise Filtering**: Automatically blocks "mega-node" hubs (like Author or Repository) during depth searches to keep results relevant.
- **Whitelisted Traversal**: Use `--dfs-attribute` to strictly control the dimensions of your search (e.g., traverse only through `File` or `Symbol` associations).

### Security & Operations
- **Secret Detection**: Scans Git history for exposed environment variables and secrets.
- **Temporal Analysis**: Groups activity into "Time Buckets" to identify development hotspots and architectural trends.
- **Deduplicated Messages**: Commits sharing exact messages are linked to a single `COMMIT_MESSAGE` node, highlighting recurring patterns.

---

## Installation

GDSKG requires Python 3.9+ and uses Tree-sitter for high-performance parsing.

```bash
# Clone the repository
git clone https://github.com/your-username/gdskg.git
cd gdskg

# Install dependencies
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Build the Intelligence Graph
Analyze a repository to generate a local SQLite-backed knowledge graph.

```bash
python main.py build \
  --repository /path/to/local/project \
  --graph ./graph_output \
  --overwrite
```

### 2. Query for Insights
Search the graph using natural language keywords and explore relationships.

```bash
# Find commits modifying 'rateLimit' logic and show immediate neighbors
python main.py query "rateLimit" -G ./graph_output -D 1

# Deep dive (Depth 2) to see other commits that touched the same files
python main.py query "auth" -G ./graph_output -D 2 -T File

# Multi-attribute traversal
python main.py query "database" -G ./graph_output -D 3 -T Symbol -T File
```

---

## Schema Architecture

### Node Dictionary
| Node Type | Description |
| :--- | :--- |
| **`COMMIT`** | A Git commit containing SHA, author info, and timestamp. |
| **`AUTHOR`** | Unique developer entities based on email. |
| **`FILE`** | Filesystem paths touched across the history. |
| **`SYMBOL`** | Logical code entities (Functions, Classes, Variables). |
| **`COMMIT_MESSAGE`** | The raw text of commit messages, deduplicated across the graph. |
| **`SECRET`** | Potential secrets or environment variables detected in code. |
| **`TIME_BUCKET`** | Temporal nodes (YYYY-MM) for time-based correlation. |
| **`REPOSITORY`** | The root repository metadata. |

### Relationship Mapping (Edges)
- **`AUTHORED_BY`**: Links `COMMIT` → `AUTHOR`.
- **`MODIFIED_FILE`**: Links `COMMIT` → `FILE`.
- **`MODIFIED_SYMBOL`**: Links `COMMIT` → `SYMBOL`.
- **`HAS_MESSAGE`**: Links `COMMIT` → `COMMIT_MESSAGE`.
- **`REFERENCES_ENV`**: Links `COMMIT` → `SECRET`.
- **`OCCURRED_IN`**: Links `COMMIT` → `TIME_BUCKET`.
- **`PART_OF_REPO`**: Links `COMMIT` → `REPOSITORY`.

---

## Development & Testing

We use `pytest` for unit testing the extraction and search logic.

```bash
# Run all tests
pytest

# Test specific import resolution logic
python debug_ts_import.py
```

---

## Project Structure

- `core/`: Graph schema, SQLite storage, and extraction logic.
- `analysis/`: Tree-sitter parsers, symbol resolution, and the BFS query engine.
- `scripts/`: Development utilities and test repository generators.
- `tests/`: Comprehensive test suite for graph correctness.
- `main.py`: Unified CLI entry point.
