import os
import sqlite3
from pathlib import Path


def get_connection():
    db_path = os.environ.get("EVALFORGE_DB_PATH")

    if not db_path:
        raise RuntimeError("EVALFORGE_DB_PATH is not configured.")

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def initialize_database():
    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT NOT NULL,
                model TEXT NOT NULL,
                total_items INTEGER NOT NULL,
                passed INTEGER NOT NULL,
                failed INTEGER NOT NULL,
                average_token_f1 REAL,
                average_semantic_similarity REAL,
                average_judge_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS experiment_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                token_f1_score REAL,
                semantic_similarity_score REAL,
                judge_score INTEGER,
                judge_reason TEXT,
                verdict TEXT,
                actual_answer TEXT,
                FOREIGN KEY (experiment_id)
                    REFERENCES experiments(id)
                    ON DELETE CASCADE
            )
        """)

        # Safely migrate existing databases.
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(experiment_results)"
            ).fetchall()
        }

        if "actual_answer" not in columns:
            connection.execute("""
                ALTER TABLE experiment_results
                ADD COLUMN actual_answer TEXT
            """)


def save_experiment(experiment):
    with get_connection() as connection:

        cursor = connection.execute("""
            INSERT INTO experiments (
                experiment_name,
                model,
                total_items,
                passed,
                failed,
                average_token_f1,
                average_semantic_similarity,
                average_judge_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment["experiment_name"],
            experiment["model"],
            experiment["total_items"],
            experiment["passed"],
            experiment["failed"],
            experiment["average_token_f1"],
            experiment["average_semantic_similarity"],
            experiment["average_judge_score"]
        ))

        experiment_id = cursor.lastrowid

        for result in experiment["results"]:
            connection.execute("""
                INSERT INTO experiment_results (
                    experiment_id,
                    question,
                    token_f1_score,
                    semantic_similarity_score,
                    judge_score,
                    judge_reason,
                    verdict,
                    actual_answer
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id,
                result["question"],
                result["token_f1_score"],
                result["semantic_similarity_score"],
                result["judge_score"],
                result["judge_reason"],
                result["verdict"],
                result.get("actual_answer", "")
            ))

        return experiment_id


def get_all_experiments():
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT * FROM experiments ORDER BY id DESC
        """).fetchall()

        return [dict(row) for row in rows]


def get_experiment_by_id(experiment_id):
    with get_connection() as connection:

        experiment = connection.execute("""
            SELECT * FROM experiments WHERE id = ?
        """, (experiment_id,)).fetchone()

        if experiment is None:
            return None

        experiment = dict(experiment)

        results = connection.execute("""
            SELECT * FROM experiment_results
            WHERE experiment_id = ?
            ORDER BY id
        """, (experiment_id,)).fetchall()

        experiment["results"] = [
            dict(row) for row in results
        ]

        return experiment
