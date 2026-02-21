# Git-Derived Software Knowledge Graph (GDSKG)

<p align="center">
  <img src="docs/assets/gadskig_mascot.png" width="300" alt="Gadskig Mascot">
  <br>
  <i>Meet Gadskig, who is always hungry for more knowledge</i>
</p>

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tree-sitter](https://img.shields.io/badge/parsing-tree--sitter-orange.svg)](https://tree-sitter.github.io/tree-sitter/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GDSKG** gives your AI Agents and developers a deep understanding of your codebase. 

Instead of searching through flat text or raw git logs, GDSKG parses your commits at the AST (Abstract Syntax Tree) level and builds an interconnected **knowledge graph**. It tracks exactly *why* functions changed, *who* touched what symbols, and *how* different pieces of your architecture evolved over time—then makes all of this queryable via natural language, vector search, and graph traversal.

---

## Why Use GDSKG?

If you've ever asked:
- *"When was this authentication bug introduced, and what other files were touched in that same commit?"*
- *"Show me the entire version history of `calculate_tax()`, ignoring unrelated file changes."*
- *"Who is the domain expert on our database connection pooling?"*

**Standard text search and basic vector databases fail here.** They lack the structural mapping of how files, functions, commits, and authors relate to one another. GDSKG bridges this gap perfectly, making it the ultimate tool for **AI coding assistants** (via Model Context Protocol) to orient themselves in new codebases, and **developers debugging complex legacy code**.

### Core Capabilities

- **Intelligent Graph Traversal**: Uses Graph + Vector Search to trace how concepts relate. Find a keyword, trace it to a commit, trace the commit to a PR, then trace it to the author—all automatically linked!
- **Deep Semantic Search**: AI-powered natural language queries to find concepts (e.g., "rate limiting logic") even if the exact keywords aren't present.
- **AST-Level Diffing**: Knows the difference between *modifying* a function, *calling* a function, and *deleting* a class.
- **Language Support**: Deep support for **Python**, **TypeScript**, and **TSX/JSX** (with more parser extensions coming), and fallback keyword detection for all other file types.

---

## How to Use It (Quickstart)

The absolute fastest way to get value from GDSKG is by running it as an **MCP (Model Context Protocol) Server**. This allows tools like Claude Desktop, Cursor, or autonomous AI agents to explore your repository's history as if they had written the code themselves.

### 1. Start the Server (Docker)
Run the server instantly without worrying about Python dependencies:

```bash
docker pull ghcr.io/gdskg/gdskg:latest

# Expose the MCP server locally and mount a volume for semantic search vectors
docker run -p 8015:8015 -v $(pwd)/vector_db:/app/vector_db ghcr.io/gdskg/gdskg:latest
```

*(You can also pass GitHub tokens during boot if you want your AI to index private remote repos directly!)*
`docker run -p 8015:8015 -v $(pwd)/vector_db:/app/vector_db -e GITHUB_TOKEN=your_token ghcr.io/gdskg/gdskg:latest`

### 2. Connect Your AI
Point your MCP-compatible client to `http://localhost:8015/mcp` (or hook up the Stdio transport). Your AI assistant will instantly gain access to specialized tools:
- `index_repository`: AI can dynamically clone and parse public/private repos.
- `query_knowledge_graph`: AI can search contexts, trace logic paths, and uncover architecture.
- `get_function_history`: AI can pull the exact historical transformation of any function block.

### 3. Ask Questions
Just ask your AI: *"Index the repository at `https://github.com/gdskg/gdskg` and then explain how the semantic search system evolved over the last year. Give me specific commits and related files."*

The AI will handle cloning, parsing, graph-building, and querying for you securely.

---

## CLI Usage (For Developers)

If you prefer to use GDSKG directly from your terminal to investigate your local projects:

### Installation
```bash
git clone https://github.com/gdskg/gdskg.git
cd gdskg
pip install -r requirements.txt
```

### 1. Build the Graph
Analyze your local repository to generate the SQLite-backed knowledge graph and vector DB.
```bash
python main.py build \
  --repository /path/to/local/project \
  --graph ./graph_output \
  --overwrite
```

### 2. Query for Insights
Search the graph using natural language and explore relationships.

```bash
# Semantic search for 'rateLimit' logic and show immediate neighboring nodes (Depth 1)
python main.py query "rateLimit" -G ./graph_output -D 1

# Deep dive (Depth 2) to see what other files were touched in the same commits
python main.py query "auth" -G ./graph_output -D 2 -T File

# Extract the entire evolution of a specific function
python main.py history "calculate_total" -G ./graph_output
```

---

## Plugins & Integrations

GDSKG gets even more powerful when you enrich the graph with external metadata. Plugins can be run dynamically via the CLI or MCP.

- **`GitHubPR`**: Pulls in PR descriptions and links them directly to the commits.
- **`ClickUpTask`**: Links commits to ticketing systems for full traceability.

*Require Auth?* Yes, you can pass these in via the CLI (`-X GitHubPR:token=...`) or your MCP client's configuration!

---

## Architecture & Schema

GDSKG creates a multi-dimensional map of your code:

| Node Type | Description |
| :--- | :--- |
| **`COMMIT`** | A Git commit containing SHA, author info, and timestamp. |
| **`AUTHOR`** | Unique developer entities based on email. |
| **`FILE`** | Filesystem paths touched across the history. |
| **`SYMBOL`** | Logical code entities (Functions, Classes). |
| **`COMMIT_MESSAGE`** | The raw text of commit messages, deduplicated across the graph. |

These are connected by edges like `AUTHORED_BY`, `MODIFIED_SYMBOL`, and `MODIFIED_FILE` allowing for powerful graph-traversal queries.

---

## Local Development & Testing

We use `pytest` for unit testing the extraction and search logic.

```bash
# Run all tests
pytest

# Test specific extraction logic
pytest tests/test_e2e.py
```

### Project Structure
- `core/`: Graph schema, SQLite storage, Vector DB, and extraction logic.
- `analysis/`: Tree-sitter parsers, semantic embedders, and the search engine.
- `mcp_server/`: The standard FastMCP interface exposing graph commands to AI.
- `plugins/`: Extensible modules for third-party integrations.
