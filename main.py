"""legal-agent main module

This module wires up the core agent and tools:

- `nl2sql_tool`: Safely converts natural language questions into SQLite `SELECT` queries
  against `data/legal.db`, executing them and returning a tabular preview of results.
- `rag_search_tool`: Performs semantic retrieval against a Chroma vector store persisted in
  `data/chroma/legal`, supporting metadata filters on `created_utc` and `text_label`.

The ReAct-style agent is instantiated with these tools via `create_react_agent` using
the `openai:gpt-4.1` chat model configured by environment variables.

Environment:
- Requires OpenAI credentials (e.g., `OPENAI_API_KEY`) loaded via `.env`.

"""

from typing import Annotated, List, Tuple

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

import sqlite3
from pathlib import Path
from collections import Counter
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

def _introspect_schema(conn: sqlite3.Connection) -> Tuple[str, List[str]]:
    """Return a printable schema summary and list of table names for a SQLite DB."""
    cur = conn.cursor()
    tables = [
        row[0]
        for row in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
        ).fetchall()
    ]
    schema_lines: List[str] = []
    for t in tables:
        cols = cur.execute(f"PRAGMA table_info({t});").fetchall()
        col_defs = ", ".join([f"{c[1]} {c[2]}" for c in cols])
        schema_lines.append(f"TABLE {t}({col_defs})")
    return "\n".join(schema_lines), tables


def _ensure_select_only(sql: str) -> str:
    """Normalize an LLM-produced SQL string and assert it is a SELECT.

    Strips code fences and `sql` language tags if present, enforces the statement
    starts with `SELECT`, and removes a trailing semicolon.
    """
    s = sql.strip().strip(";")
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1] if not parts[0] else parts[1]
            s = s.strip()
    if s.lower().startswith("sql\n"):
        s = s[4:].strip()
    if not s.lower().lstrip().startswith("select"):
        raise ValueError("Generated SQL is not a SELECT statement.")
    return s


@tool("nl2sql")
def nl2sql_tool(question: str, db_path: str = str(Path("data") / "legal.db"), limit: int = 50) -> str:
    """Translate NL to a safe SQLite SELECT and preview results.

    Dataset and schema:
    - Table: `posts(id, created_utc, full_link, title, body, text_label, flair_label)`.
    - Typical lexical targets: `title`, `body`. Typical filters: `created_utc`, `text_label`.

    How this helps users and the LLM for lexical search:
    - Allows keyword-based lookups over `title`/`body` via SQL (e.g., `LOWER(title) LIKE '%eviction%'`).
    - Supports time windows using epoch seconds on `created_utc` (e.g., 2019 → 1546300800..1577836799).
    - Supports topical filtering using `text_label` (single label or `IN (...)`).
    - Ideal for counts, aggregates, top-N, and precise tabular answers grounded in the relational data.

    When to use vs. semantic RAG (`rag_search`):
    - Use `nl2sql` for exact numbers, lists, or filters by keywords/labels/dates.
    - Use `rag_search` for open-ended synthesis or when you want relevant passages by meaning.

    Tips for lexical queries the LLM can generate:
    - Use `LOWER(title) LIKE '%term%' OR LOWER(body) LIKE '%term%'` for case-insensitive keyword search.
    - Combine with `created_utc BETWEEN <start_epoch> AND <end_epoch>` for date ranges.
    - Add `AND text_label = 'housing'` or `AND text_label IN ('housing','contract')` for topical focus.

    Parameters:
    - question: Natural language query to convert to SQL.
    - db_path: Path to the SQLite database file.
    - limit: Row cap for the results preview and a default LIMIT if absent.
    """
    local_llm = init_chat_model("openai:gpt-4o-mini")
    conn = sqlite3.connect(db_path)
    try:
        schema_text, _ = _introspect_schema(conn)
        sys_msg = (
            "You are a careful SQLite expert. Given a user question and the database schema, "
            "produce exactly ONE SQLite SELECT query that answers it. Use only available tables/columns. "
            f"If no LIMIT appears, append LIMIT {limit}. Do not include commentary."
        )
        user_msg = (
            f"SCHEMA:\n{schema_text}\n\n"
            f"QUESTION: {question}\n\n"
            "Return only the SQL."
        )
        resp = local_llm.invoke([
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ])
        sql = _ensure_select_only(resp.content)

        cur = conn.cursor()
        try:
            rows = cur.execute(sql).fetchmany(limit)
            colnames = [d[0] for d in cur.description] if cur.description else []
        except Exception as e:
            return f"SQL:\n{sql}\n\nERROR: {e}"

        lines = []
        if colnames:
            lines.append("\t".join(colnames))
        for r in rows:
            lines.append("\t".join(["" if v is None else str(v) for v in r]))
        result_text = "\n".join(lines)
        return f"SQL:\n{sql}\n\nRESULTS (up to {limit} rows):\n{result_text}"
    finally:
        conn.close()


@tool("rag_search")
def rag_search_tool(
    query: str,
    top_k: int = 5,
    persist_dir: str = str(Path("data") / "chroma" / "legal"),
    collection: str = "legal_posts",
    embedding_model: str = "text-embedding-3-small",
    created_utc_gte: int | None = None,
    created_utc_lte: int | None = None,
    text_label: str | List[str] | None = None,
) -> str:
    """Semantic RAG over the legal posts corpus with metadata filters.

    Corpus and fields:
    - Source: `datasets/legal-dataset.jsonl` loaded into Chroma at `data/chroma/legal`.
    - Content: `page_content` is the concatenation of `title` + two newlines + `body`.
    - Metadata: `created_utc` (Unix epoch seconds), `text_label` (string category).

    What this tool helps with:
    - Retrieves semantically similar posts for open-ended questions (e.g., "trends in housing disputes").
    - Narrows results by time window using `created_utc_gte`/`created_utc_lte`.
    - Focuses on topical slices via `text_label` (single label or list using `$in`).
    - Enables the LLM to ground summaries/answers in concrete retrieved evidence.

    When to use vs. `nl2sql`:
    - Use `rag_search` for unstructured, qualitative, or example-driven queries; when you want passages to quote or summarize; or when you don't know precise columns.
    - Use `nl2sql` for exact counts, aggregations, or tabular lookups over the relational `data/legal.db`.

    Parameters:
    - query: Free-text query to search for.
    - top_k: Number of results to return.
    - persist_dir: Directory containing the Chroma DB.
    - collection: Collection name inside Chroma.
    - embedding_model: OpenAI embedding model name.
    - created_utc_gte: Only include docs with `created_utc` >= this value (e.g., 2019-01-01 00:00:00 UTC → 1546300800).
    - created_utc_lte: Only include docs with `created_utc` <= this value (e.g., 2019-12-31 23:59:59 UTC → 1577836799).
    - text_label: Single label (string) or list of labels (uses `$in`).

    Filter semantics:
    - Multiple field conditions are combined under a top-level `$and`.
    - Ranges are expressed as two separate operator expressions: `{created_utc: {$gte: ...}}` and `{created_utc: {$lte: ...}}`.
    """
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    # Build Chroma-compatible filter for metadata fields.
    # Notes for future readers:
    # - Chroma expects each field's operator expression to contain exactly one operator.
    #   Therefore, for a range we MUST split into two separate clauses:
    #     {"created_utc": {"$gte": ...}} and {"created_utc": {"$lte": ...}}
    # - When multiple field conditions are present, they must be wrapped under a
    #   single top-level logical operator. We use "$and" here to intersect filters.
    clauses: List[dict] = []
    # Split range into two separate operator expressions to satisfy Chroma's filter grammar
    if created_utc_gte is not None:
        clauses.append({"created_utc": {"$gte": int(created_utc_gte)}})
    if created_utc_lte is not None:
        clauses.append({"created_utc": {"$lte": int(created_utc_lte)}})
    if text_label is not None:
        if isinstance(text_label, list):
            clauses.append({"text_label": {"$in": text_label}})
        else:
            clauses.append({"text_label": text_label})

    # Collapse the clauses into the final "where" filter Chroma expects.
    # - 0 clauses → no filter (None)
    # - 1 clause  → pass the single dict directly
    # - >1 clauses → wrap with top-level "$and"
    where = None
    if len(clauses) == 1:
        where = clauses[0]
    elif len(clauses) > 1:
        where = {"$and": clauses}

    # Perform semantic search with optional metadata filter.
    # Prefer the API that returns (Document, relevance_score) pairs. If that
    # method is not available in the environment, fall back to plain similarity
    # search and synthesize a (doc, None) tuple for a consistent downstream shape.
    try:
        results = vectorstore.similarity_search_with_relevance_scores(
            query, k=top_k, filter=where
        )
    except Exception:
        # Fallback if relevance scores API is unavailable
        docs = vectorstore.similarity_search(query, k=top_k, filter=where)
        results = [(d, None) for d in docs]

    # Format a compact, readable preview:
    # - Header shows k, applied filters, and collection
    # - Each entry prints score (if available), key metadata, and a truncated snippet
    lines: List[str] = []
    lines.append(
        f"RESULTS (k={top_k}, filters={where if where is not None else {}}, collection='{collection}')"
    )
    for idx, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata or {}
        created = meta.get("created_utc")
        label = meta.get("text_label")
        snippet = (doc.page_content or "").strip()
        # Truncate long content to keep terminal output skimmable
        if len(snippet) > 600:
            snippet = snippet[:600] + "..."
        lines.append(
            f"{idx}. score={score if score is not None else 'n/a'} | created_utc={created} | text_label={label}\n{snippet}"
        )
    return "\n\n".join(lines)

llm_agent = init_chat_model("openai:gpt-4.1")
tools = [nl2sql_tool, rag_search_tool]
agent = create_react_agent(llm_agent, tools)