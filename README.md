# Git-Derived Software Knowledge Graph (GDSKG)

<p align="center">
  <img src="docs/assets/gadskig_mascot.png" width="300" alt="Gadskig Mascot">
  <br>
  <i>Meet Gadskig, who is always hungry for more knowledge</i>
</p>


[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tree-sitter](https://img.shields.io/badge/parsing-tree--sitter-orange.svg)](https://tree-sitter.github.io/tree-sitter/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Meet **Gadskig**, the cute little monster guardian of your codebase. **GDSKG** is a high-fidelity intelligence tool designed to map the deep evolutionary relationships within software repositories. By combining Git history with AST-level code analysis via Tree-sitter, GDSKG builds a navigable knowledge graph that transcends simple text search.

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

## Environment Variables

Some GDSKG plugins require environment variables for authentication with external services:

| Variable | Required By | Description |
| :--- | :--- | :--- |
| **`GITHUB_PAT`** | `GitHubPR` | A GitHub Personal Access Token with `repo` scope to fetch Pull Request metadata. |
| **`CLICKUP_API_KEY`** | `ClickUpTask` | Your ClickUp API Token (found in User Settings > Apps) to fetch Task metadata. |

> [!NOTE]
> These variables are only required if you enable the corresponding plugins during indexing.

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

## MCP Server

GDSKG provides an MCP (Model Context Protocol) server to allow AI assistants to directly interact with the knowledge graph.

### Tools

#### `index_folder`
Indexes a local git repository into the knowledge graph. Supports all functionality available in the CLI.

**Parameters:**
- `path` (str): Absolute path to the local git repository.
- `graph_path` (str, optional): Directory to store the graph DB. Defaults to `./gdskg_graph`.
- `overwrite` (bool, optional): Overwrite existing database. Default `False`.
- `plugins` (List[str], optional): List of plugins to enable (e.g., `["GitHubPR", "ClickUpTask"]`).
- `parameters` (List[str], optional): Plugin configuration in format `["Plugin:Key=Value"]`.

#### `query_knowledge_graph`
Queries the knowledge graph using natural language.

**Parameters:**
- `query` (str): Search query or question.
- `graph_path` (str, optional): Path to the graph DB.
- `limit` (int, optional): Max results. Default `10`.

### Running the Server
```bash
# Run server using the provided wrapper or directly
python -m gdskg.mcp_server.server
```

### Tray Icon Application

GDSKG includes a tray icon application that manages the MCP server in the background.

**Features:**
- Start/Stop the MCP server from the system tray.
- Cross-platform support (Windows, macOS, Linux).
- Status indicator in the tray menu.

**Running the Tray App:**
```bash
python -m gdskg.mcp_server.tray
```

## Installation

### Option 1: Docker (Recommended)
Run the MCP server in a container without installing dependencies locally. This avoids macOS Gatekeeper issues.

```bash
# Pull the latest image
docker pull ghcr.io/gdskg/gdskg:latest

# Run the server in network mode (SSE) - Default behavior
# This exposes port 8015 for local connections
docker run -p 8015:8015 -v $(pwd):/data ghcr.io/gdskg/gdskg:latest

# Run the server in interactive mode (Stdio)
# Note: '-i' is required for the MCP server to communicate via stdin/stdout
docker run -i -v $(pwd):/data ghcr.io/gdskg/gdskg:latest serve --transport stdio
```

### Option 2: Pre-built Executables
Download the latest release for your OS from the [Releases page](https://github.com/gdskg/gdskg/releases).

**Note for macOS users:** You may need to bypass Gatekeeper by right-clicking the app and selecting "Open", or running `xattr -d com.apple.quarantine gdskg-mcp-macos`.
### Analyzing Remote Repositories
You can analyze remote repositories directly by providing a URL. If the repository is private, ensure `GITHUB_TOKEN` is set in your environment.

```bash
export GITHUB_TOKEN=your_token
docker run -e GITHUB_TOKEN=$GITHUB_TOKEN -v $(pwd):/data ghcr.io/gdskg/gdskg:latest build --repository https://github.com/owner/repo --graph /data/graph
```
The repository will be cloned to a local cache within the container (or your mounted volume).

### Option 3: From Source
You can build the standalone executable locally using the provided `Makefile`:

```bash
# Build the executable
make build

# Clean build artifacts
make clean
```

---

## Plugins

GDSKG supports a plugin system to enrich the graph with external data. Plugins are located in the `plugins/` directory.

### Available Plugins

- **`GitHubPR`**: Enriches the graph with GitHub Pull Request data.
- **`ClickUpTask`**: Enriches the graph with ClickUp task data.

### Using Plugins via MCP
To enable plugins when indexing via MCP, provide the `plugins` and `parameters` arguments:
```json
{
  "path": "/absolute/path/to/repo",
  "plugins": ["GitHubPR"],
  "parameters": ["GitHubPR:token=ghp_your_token_here"]
}
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
