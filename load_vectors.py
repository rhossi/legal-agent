import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Generator, List, Tuple

from dotenv import load_dotenv

# Vector store + embeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


def iter_jsonl(
    jsonl_path: Path,
) -> Generator[Tuple[str, str, Dict], None, None]:
    """Yield `(id, text, metadata)` tuples from a JSONL dataset.

    Text is composed as "title\n\nbody" when both fields exist. The metadata contains
    `created_utc` and `text_label` so that they can be used later for filtering in
    retrieval.
    """
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            doc_id = str(obj.get("id")) if obj.get("id") is not None else None
            if not doc_id:
                # Skip rows without id
                continue

            title = obj.get("title") or ""
            body = obj.get("body") or ""
            text = (title + "\n\n" + body).strip()
            if not text:
                # Skip empty content
                continue

            metadata = {
                "created_utc": obj.get("created_utc"),
                "text_label": obj.get("text_label"),
            }
            yield doc_id, text, metadata


def embed_to_chroma(
    jsonl_path: Path,
    persist_dir: Path,
    collection_name: str,
    batch_size: int = 500,
    embedding_model: str = "text-embedding-3-small",
) -> None:
    """Create and persist a Chroma collection from the JSONL dataset.

    If `persist_dir` exists, it will be recreated to ensure a clean index build.
    """
    # Fresh start if exists
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv()

    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict] = []

    total = 0
    for doc_id, text, meta in iter_jsonl(jsonl_path):
        ids.append(doc_id)
        texts.append(text)
        metadatas.append(meta)
        if len(ids) >= batch_size:
            vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            vectorstore.persist()
            total += len(ids)
            print(f"Indexed {total} documents...")
            ids.clear()
            texts.clear()
            metadatas.clear()

    if ids:
        vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        vectorstore.persist()
        total += len(ids)

    print(f"Done. Indexed {total} documents into '{collection_name}' at {persist_dir}.")


def main():
    """CLI entry point to build a Chroma DB from a JSONL dataset."""
    parser = argparse.ArgumentParser(description="Embed JSONL dataset into a Chroma vector DB")
    parser.add_argument(
        "--jsonl",
        type=str,
        default=str(Path("datasets") / "legal-dataset.jsonl"),
        help="Path to JSONL dataset",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=str(Path("data") / "chroma" / "legal"),
        help="Directory to persist Chroma database",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="legal_posts",
        help="Chroma collection name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for upserts",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model",
    )

    args = parser.parse_args()
    embed_to_chroma(
        jsonl_path=Path(args.jsonl),
        persist_dir=Path(args.persist_dir),
        collection_name=args.collection,
        batch_size=args.batch_size,
        embedding_model=args.embedding_model,
    )


if __name__ == "__main__":
    main()


