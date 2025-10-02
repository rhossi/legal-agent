# Legal Agent

This project provides a LangGraph-powered agent for legal-domain data exploration with two primary tools:

- `nl2sql`: Convert natural language into safe SQLite `SELECT` queries against `data/legal.db` and preview results.
- `rag_search`: Retrieve semantically similar documents from a Chroma DB persisted at `data/chroma/legal` with metadata filters.

## Setup

1. Python 3.12+
2. Install dependencies (using `uv` or pip):

```bash
uv sync
# or
pip install -e .
```

1. Configure environment variables in a `.env` file (OpenAI credentials):

```bash
OPENAI_API_KEY=sk-...
```

Alternatively, copy from a template:

```bash
cp .env.example .env
```

Then edit `.env` and set required/optional variables.

Suggested `.env.example`:

```bash
# Required
# OpenAI API key for embeddings and chat models
OPENAI_API_KEY=

# Optional: LangSmith observability (leave unset if not using)
#LANGSMITH_API_KEY=
```

## Build Vector Store

Create a Chroma vector database from the provided JSONL dataset:

```bash
python load_vectors.py --jsonl datasets/legal-dataset.jsonl \
  --persist-dir data/chroma/legal \
  --collection legal_posts \
  --embedding-model text-embedding-3-small
```

## Load Relational Database

Load the same dataset into SQLite for NL2SQL queries:

```bash
python load_relational.py --jsonl datasets/legal-dataset.jsonl \
  --db data/legal.db \
  --table posts \
  --analyze
```

## Using the Tools Programmatically

In `main.py` an agent is created with both tools. You can also call tools directly:

```python
from main import nl2sql_tool, rag_search_tool

print(nl2sql_tool("Top 10 posts by created_utc"))

print(rag_search_tool(
    query="contract breach remedies",
    top_k=5,
    created_utc_gte=1609459200,  # 2021-01-01
    created_utc_lte=1640995199,  # 2021-12-31
    text_label=["contract", "litigation"],
))
```

Notes on RAG filter syntax: multiple field filters are combined with a top-level `$and`. Ranges are expressed using two separate operator expressions for `$gte` and `$lte`.

## When to use `rag_search` vs `nl2sql`

- **Use `rag_search`**: for open-ended, qualitative, or exploratory questions where you want to pull exemplar passages and synthesize trends (e.g., "What were the trends in housing disputes in 2019?"). It retrieves text segments from Chroma based on semantic similarity, with optional filters:
  - **created_utc_gte/lte**: restrict by Unix epoch seconds (e.g., 2019 → 1546300800 to 1577836799)
  - **text_label**: focus on topical slices, e.g., `"housing"`, or multiple via `["housing", "contract"]`

- **Use `nl2sql`**: for precise counts, aggregations, and structured lookups over the SQLite database (e.g., "How many posts had label 'housing' in 2019?", "Top 10 labels by count"). It translates NL to a safe `SELECT` and previews results.

Tip for the agent: combine both—first use `rag_search` to gather context/evidence, then use `nl2sql` to compute exact statistics referenced in your summary.

## Lexical search with `nl2sql`

Use `nl2sql` to have the LLM generate keyword-oriented SQL over the relational dataset `data/legal.db` (table `posts`). Common patterns:

```sql
SELECT id, title
FROM posts
WHERE (
  LOWER(title) LIKE '%eviction%'
  OR LOWER(body) LIKE '%eviction%'
)
AND created_utc BETWEEN 1546300800 AND 1577836799 -- 2019
AND text_label IN ('housing');
```

Examples of suitable questions:

- "How many posts mention eviction in 2019?"
- "List top 10 titles about landlord disputes in 2021."
- "Counts by text_label for posts containing 'breach'."

## Serve the agent with LangGraph CLI

The project includes a `langgraph.json` mapping the graph name `agent` to `main.py:agent`. You can use the LangGraph CLI to run a local dev server.

### Install the CLI

- Using this project's dependencies (recommended):

```bash
uv sync  # or pip install -e .
```

The CLI (`langgraph`) is installed from `pyproject.toml`.

- Or install globally via pipx:

```bash
pipx install langgraph-cli
```

### Run the dev server

From the project root (where `langgraph.json` lives):

```bash
langgraph dev
```

Optional flags:

```bash
langgraph dev --host 0.0.0.0 --port 8123
```

Notes:

- The CLI will read environment variables from `.env` (as configured in `langgraph.json`). Ensure `OPENAI_API_KEY` is set.
- The dev server provides a local UI and HTTP API for the `agent` graph defined in `main.py`.
- Use the on-screen playground to chat with the agent, or see LangGraph docs for API endpoints.

## Optional: Run Agent Chat UI (LangChain)

Use the community Agent Chat UI to interact with your locally served graph.

Prerequisites:

- Node.js 18+ and npm (or pnpm)
- A running dev server from the previous step (`langgraph dev`)

Steps:

1. Start the LangGraph dev server in this repo’s root.

```bash
langgraph dev --port 8123
```

1. In a separate terminal, clone and install the Agent Chat UI.

```bash
git clone https://github.com/langchain-ai/agent-chat-ui
cd agent-chat-ui
npm install  # or: pnpm install
```

1. Create a `.env.local` file in the UI project and set the backend URL and graph id.

```bash
echo "NEXT_PUBLIC_LANGGRAPH_BASE_URL=http://localhost:8123" >> .env.local
echo "NEXT_PUBLIC_GRAPH_ID=agent" >> .env.local
# optional branding
echo "NEXT_PUBLIC_PROJECT_TITLE=Legal Agent" >> .env.local
```

1. Run the UI.

```bash
npm run dev  # or: pnpm dev
```

Open `http://localhost:3000` and chat with the `agent` graph. If you used a different port for `langgraph dev`, update `NEXT_PUBLIC_LANGGRAPH_BASE_URL` accordingly.

## Dataset

The project expects a JSONL file at `datasets/legal-dataset.jsonl` with fields:
`id`, `created_utc`, `full_link`, `title`, `body`, `text_label`, `flair_label`.
