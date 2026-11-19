
import json
import shutil
import uuid

import pandas as pd

from datetime import datetime, timezone
from pathlib import Path


def register_experiment(
    experiment_id,
    experiment_name,
    comparison_report,
    models,
    dataset,
    response_files,
    evaluation_files,
    dataset_size,
    metrics,
    provider="Groq",
    judge_model=None,
    temperature=0
):
    """Register a completed model evaluation experiment."""

    comparison_path = Path(comparison_report)

    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Comparison report not found: {comparison_path}"
        )

    required_files = [
        Path(dataset),
        *[Path(p) for p in response_files.values()],
        *[Path(p) for p in evaluation_files.values()]
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    comparison_df = pd.read_csv(comparison_path)

    required_columns = [
        "model_name",
        "total_questions",
        "judge_pass",
        "judge_fail",
        "judge_errors",
        "avg_token_f1",
        "avg_semantic_similarity",
        "avg_judge_score",
        "avg_latency_seconds",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "avg_factuality_score",
        "factuality_supported",
        "factuality_partially_supported",
        "factuality_unsupported",
        "factuality_errors",
        "avg_reasoning_score",
        "reasoning_strong",
        "reasoning_acceptable",
        "reasoning_weak",
        "reasoning_errors"
    ]

    missing = [
        col for col in required_columns
        if col not in comparison_df.columns
    ]

    if missing:
        raise ValueError(
            f"Comparison report is missing columns: {missing}"
        )

    if comparison_df["model_name"].duplicated().any():
        raise ValueError("Duplicate model names in comparison report.")

    run_dir = Path("experiments/runs") / experiment_id
    config_dir = Path("experiments/configs")

    if run_dir.exists():
        raise FileExistsError(
            f"{experiment_id} already exists. Choose a new experiment ID."
        )

    run_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    model_results = {}

    for _, row in comparison_df.iterrows():
        model_name = row["model_name"]

        model_results[model_name] = {
            "total_questions": int(row["total_questions"]),
            "judge_pass": int(row["judge_pass"]),
            "judge_fail": int(row["judge_fail"]),
            "judge_errors": int(row["judge_errors"]),
            "avg_token_f1": float(row["avg_token_f1"]),
            "avg_semantic_similarity": float(
                row["avg_semantic_similarity"]
            ),
            "avg_judge_score": float(row["avg_judge_score"]),
            "avg_latency_seconds": float(
                row["avg_latency_seconds"]
            ),
            "avg_prompt_tokens": float(row["avg_prompt_tokens"]),
            "avg_completion_tokens": float(
                row["avg_completion_tokens"]
            ),
            "avg_factuality_score": float(row["avg_factuality_score"]),
            "factuality_supported": int(row["factuality_supported"]),
            "factuality_partially_supported": int(
                row["factuality_partially_supported"]
            ),
            "factuality_unsupported": int(row["factuality_unsupported"]),
            "factuality_errors": int(row["factuality_errors"]),
            "avg_reasoning_score": float(row["avg_reasoning_score"]),
            "reasoning_strong": int(row["reasoning_strong"]),
            "reasoning_acceptable": int(row["reasoning_acceptable"]),
            "reasoning_weak": int(row["reasoning_weak"]),
            "reasoning_errors": int(row["reasoning_errors"])
        }

    config = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "models": models,
        "provider": provider,
        "dataset": str(dataset),
        "dataset_size": dataset_size,
        "temperature": temperature,
        "judge_model": judge_model,
        "response_files": response_files,
        "evaluation_files": evaluation_files,
        "metrics": metrics,
        "comparison_report": str(comparison_path)
    }

    metadata = {
        "experiment_id": experiment_id,
        "run_id": str(uuid.uuid4()),
        "experiment_name": experiment_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "provider": provider,
        "models_evaluated": models,
        "dataset_size": dataset_size,
        "evaluation_metrics": metrics,
        "model_results": model_results,
        "reports": {
            "comparison_report": str(comparison_path),
            "experiment_comparison_copy": "model_comparison.csv",
            "evaluation_files": evaluation_files
        }
    }

    config_path = config_dir / f"{experiment_id}_config.json"
    metadata_path = run_dir / "metadata.json"
    report_copy = run_dir / "model_comparison.csv"

    with open(config_path, "w") as file:
        json.dump(config, file, indent=4)

    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=4)

    shutil.copy2(comparison_path, report_copy)

    return {
        "experiment_id": experiment_id,
        "config_path": str(config_path),
        "metadata_path": str(metadata_path),
        "report_path": str(report_copy),
        "model_results": model_results
    }
