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
- `query_knowledge_graph`: AI can search contexts, trace logic paths, and uncover architecture. Includes rich metadata for related nodes.
- `get_function_history`: AI can pull the exact historical transformation of any function block.
- `get_ast_nodes`: AI can retrieve the deep Abstract Syntax Tree linked directly to an active file or function version. Supports **semantic filtering** via the `query` parameter.
- `get_dependencies`: AI can discover what a node (file, function, symbol) uses and what uses it. Supports **fuzzy path resolution** (e.g., just passing "Auth.tsx" matches the full internal path).
- `list_node_types` / `list_edge_types`: AI can dynamically query available entity and relationship schema definitions.
- `amend_graph`: AI can actively update or correct the knowledge graph (using `add_node`, `update_edge`, etc.), producing a replayable changelog in the process.

### 3. Ask Questions
Just ask your AI: *"Index the repository at `https://github.com/gdskg/gdskg` and then explain how the semantic search system evolved over the last year. Give me specific commits and related files."*

The AI will handle cloning, parsing, graph-building, and querying for you securely.

---

## Build Your Own GDSKG Image (GitHub Action)

If you want to pre-compute the knowledge graph for your repositories and bake it directly into a Docker image (perfect for team environments or faster AI startup times), you can use our built-in GitHub Action.

Create a `.github/workflows/gdskg.yml` file in your repository:

The template is found in [gdskg/examples/github-workflow.yml](examples/github-workflow.yml)

---

## Local Interactive Setup (Manual Bundle)

If you want to build a bundled image locally without setting up a GitHub Action, you can use our interactive setup script. This is the easiest way to get an "all-in-one" image for local testing or sharing.

```bash
# From the root of the gdskg repository
make setup
```

The script will interactively guide you through:
1.  Entering the repositories you want to index.
2.  Providing an optional GitHub Token (PAT) for private repos.
3.  Naming your final Docker image.

Once finished, you will have a local Docker image (e.g., `gdskg-bundled`) that contains the pre-indexed knowledge graph for all specified repositories. You can use it in your MCP config immediately!

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

*Note: GDSKG also natively supports PostgreSQL! Set the `GDSKG_DB_URL` environment variable to a valid Postgres connection string (requires `pgvector` to be installed on the database instance).*
```bash
export GDSKG_DB_URL="postgresql://user:password@localhost:5432/gdskg"
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

# View the full AST structure of a specific file or function node
python main.py ast "analysis/search_engine.py" -G ./graph_output --depth 3

# Semantic filtering within the AST (find 'query' logic inside the file)
python main.py ast "analysis/search_engine.py" -G ./graph_output --query "parse query"

# Explore dependencies and dependents for a specific node (fuzzy matching supported)
python main.py dependencies "SearchEngine" -G ./graph_output

### 3. Administrative Commands
Manage the local environment and the server lifecycle.

```bash
# Pre-download semantic search models (recommended before first run)
python main.py prepare

# Start the MCP server manually (Stdio transport)
python main.py serve --transport stdio --graph ./graph_output

# Start the MCP server manually (SSE transport on port 8015)
python main.py serve --transport sse --port 8015
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
| **`FUNCTION_VERSION`** | Exact content of a function at a point in time. |
| **`COMMIT_MESSAGE`** | The raw text of commit messages, deduplicated across the graph. |
| **`AST_NODE`** | Tree-sitter AST nodes comprising the syntax structure of the *current HEAD* state. |

These are connected by edges like `AUTHORED_BY`, `MODIFIED_SYMBOL`, `MODIFIED_FILE`, and `CURRENT_VERSION` allowing for powerful graph-traversal queries.

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
