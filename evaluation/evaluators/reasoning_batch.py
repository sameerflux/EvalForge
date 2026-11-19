import json
import time
import pandas as pd
from pathlib import Path

from evaluation.evaluators.reasoning import evaluate_reasoning


def evaluate_reasoning_batch(
    input_file,
    output_file,
    client,
    model="openai/gpt-oss-20b",
    max_rows=None,
    delay_seconds=1.5
):
    """Evaluate reasoning quality across a CSV dataset with resume support."""

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    required = ["id", "question", "expected_answer", "actual_response"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing input columns: {missing}")

    if max_rows is not None:
        df = df.head(max_rows).copy()

    df = df.reset_index(drop=True)

    result_columns = [
        "reasoning_score",
        "reasoning_verdict",
        "reasoning_strengths",
        "reasoning_issues",
        "reasoning_explanation"
    ]

    if output_path.exists():
        results = pd.read_csv(output_path)

        if (
            len(results) != len(df)
            or "id" not in results.columns
            or list(results["id"].astype(str)) != list(df["id"].astype(str))
        ):
            raise ValueError(
                "Existing output does not match the input IDs. "
                "Use a different output filename."
            )

        for col in df.columns:
            if col not in results.columns:
                results[col] = df[col]

        for col in result_columns:
            if col not in results.columns:
                results[col] = None

        print("Resuming existing reasoning evaluation.")
    else:
        results = df.copy()

        for col in result_columns:
            results[col] = None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print("Created new reasoning evaluation report.")

    total = len(results)

    for i in range(total):
        current = results.at[i, "reasoning_verdict"]

        if pd.notna(current) and str(current) not in ("", "ERROR", "None"):
            continue

        print(f"Evaluating question {i + 1}/{total}...")

        evaluation = evaluate_reasoning(
            client=client,
            question=str(results.at[i, "question"]),
            expected_answer=str(results.at[i, "expected_answer"]),
            actual_answer=str(results.at[i, "actual_response"]),
            model=model
        )

        results.at[i, "reasoning_score"] = evaluation["reasoning_score"]
        results.at[i, "reasoning_verdict"] = evaluation["reasoning_verdict"]
        results.at[i, "reasoning_strengths"] = json.dumps(
            evaluation["reasoning_strengths"]
        )
        results.at[i, "reasoning_issues"] = json.dumps(
            evaluation["reasoning_issues"]
        )
        results.at[i, "reasoning_explanation"] = (
            evaluation["reasoning_explanation"]
        )

        # Save progress after every question
        results.to_csv(output_path, index=False)

        if i < total - 1:
            time.sleep(delay_seconds)

    successful = results["reasoning_verdict"].isin(
        ["STRONG", "ACCEPTABLE", "WEAK"]
    ).sum()

    errors = (results["reasoning_verdict"] == "ERROR").sum()

    valid_scores = pd.to_numeric(
        results["reasoning_score"], errors="coerce"
    ).dropna()

    print("\nREASONING BATCH SUMMARY")
    print("Total questions:", total)
    print("Successful evaluations:", successful)
    print("Errors:", errors)
    print(
        "Average reasoning score:",
        round(valid_scores.mean(), 3) if len(valid_scores) else "N/A"
    )
    print("Report saved:", output_path)

    return results
