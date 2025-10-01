import argparse
import json
import sqlite3
from pathlib import Path
from collections import Counter
from typing import List


def load_jsonl_to_sqlite(
    jsonl_path: str,
    db_path: str,
    table_name: str = "posts",
    analyze: bool = False,
    batch_size: int = 1000,
):
    """Load a JSONL dataset into a SQLite table, creating it if necessary.

    Columns created: id, created_utc, full_link, title, body, text_label, flair_label.
    Use `--analyze` to print a small summary of label distributions.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id TEXT PRIMARY KEY,
                created_utc INTEGER,
                full_link TEXT,
                title TEXT,
                body TEXT,
                text_label TEXT,
                flair_label INTEGER
            )
            """
        )

        labels_counter: Counter[str] = Counter()
        flair_counter: Counter[int] = Counter()

        with open(jsonl_path, "r", encoding="utf-8") as f:
            batch: List[tuple] = []
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                row = (
                    obj.get("id"),
                    obj.get("created_utc"),
                    obj.get("full_link"),
                    obj.get("title"),
                    obj.get("body"),
                    obj.get("text_label"),
                    obj.get("flair_label"),
                )
                batch.append(row)

                if analyze:
                    if obj.get("text_label") is not None:
                        labels_counter[str(obj.get("text_label"))] += 1
                    if obj.get("flair_label") is not None:
                        flair_counter[int(obj.get("flair_label"))] += 1

                if len(batch) >= batch_size:
                    conn.executemany(
                        f"""
                        INSERT OR REPLACE INTO {table_name}
                        (id, created_utc, full_link, title, body, text_label, flair_label)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        batch,
                    )
                    conn.commit()
                    batch = []

            if batch:
                conn.executemany(
                    f"""
                    INSERT OR REPLACE INTO {table_name}
                    (id, created_utc, full_link, title, body, text_label, flair_label)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                conn.commit()

        if analyze:
            print("text_label counts:")
            for label, count in labels_counter.most_common():
                print(f"  {label}: {count}")

            print("flair_label counts:")
            for label, count in flair_counter.most_common():
                print(f"  {label}: {count}")
    finally:
        conn.close()


def main():
    """CLI for loading the JSONL dataset into a SQLite database."""
    parser = argparse.ArgumentParser(description="Load JSONL dataset into SQLite")
    parser.add_argument(
        "--jsonl",
        type=str,
        default=str(Path("datasets") / "legal-dataset.jsonl"),
        help="Path to JSONL dataset",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(Path("data") / "legal.db"),
        help="Path to SQLite database file",
    )
    parser.add_argument("--table", type=str, default="posts", help="SQLite table name")
    parser.add_argument("--analyze", action="store_true", help="Print label stats")
    parser.add_argument("--batch-size", type=int, default=1000, help="Upsert batch size")

    args = parser.parse_args()
    load_jsonl_to_sqlite(
        jsonl_path=args.jsonl,
        db_path=args.db,
        table_name=args.table,
        analyze=args.analyze,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()


