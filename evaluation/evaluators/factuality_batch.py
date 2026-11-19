
import json
import time
import pandas as pd

from pathlib import Path
from evaluation.evaluators.factuality import evaluate_factuality


def evaluate_factuality_batch(
    input_file,
    output_file,
    client,
    model="openai/gpt-oss-20b",
    max_rows=None,
    delay_seconds=1
):
    """Evaluate factuality with incremental saving and resume support."""

    input_path = Path(input_file)
    output_path = Path(output_file)

    df = pd.read_csv(input_path)

    required_columns = [
        "id",
        "question",
        "expected_answer",
        "actual_response"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if max_rows is not None:
        df = df.head(max_rows).copy()
    else:
        df = df.copy()

    result_columns = [
        "factuality_score",
        "factuality_verdict",
        "supported_claims",
        "unsupported_claims",
        "incorrect_claims",
        "factuality_explanation"
    ]

    # Resume from an existing report if it matches the input dataset.
    if output_path.exists():
        saved = pd.read_csv(output_path)

        if (
            len(saved) == len(df)
            and saved["id"].astype(str).tolist()
            == df["id"].astype(str).tolist()
        ):
            df = saved.copy()
            print("Existing report found. Resuming unfinished rows.")

        else:
            print("Existing report does not match input. Starting fresh.")

    for column in result_columns:
        if column not in df.columns:
            df[column] = None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for index, row in df.iterrows():

        # Skip completed rows; retry ERROR rows.
        existing_score = row["factuality_score"]
        existing_verdict = row["factuality_verdict"]

        if (
            pd.notna(existing_score)
            and str(existing_verdict).upper() != "ERROR"
        ):
            continue

        if any(
            pd.isna(row[col]) or not str(row[col]).strip()
            for col in required_columns
        ):
            result = {
                "score": None,
                "verdict": "ERROR",
                "supported_claims": [],
                "unsupported_claims": [],
                "incorrect_claims": [],
                "explanation": "Missing input data."
            }

        else:
            print(
                f"Evaluating question {index + 1}/{len(df)}: "
                f"{row['question']}"
            )

            result = evaluate_factuality(
                client=client,
                question=str(row["question"]),
                reference_answer=str(row["expected_answer"]),
                actual_answer=str(row["actual_response"]),
                model=model
            )

        df.at[index, "factuality_score"] = result["score"]
        df.at[index, "factuality_verdict"] = result["verdict"]

        for field in [
            "supported_claims",
            "unsupported_claims",
            "incorrect_claims"
        ]:
            df.at[index, field] = json.dumps(result[field])

        df.at[index, "factuality_explanation"] = result["explanation"]

        # Save after every question.
        df.to_csv(output_path, index=False)

        if delay_seconds:
            time.sleep(delay_seconds)

    print("\nFACTUALITY BATCH COMPLETED")
    print("Total questions:", len(df))
    print("Successful:", df["factuality_score"].notna().sum())
    print(
        "Errors:",
        (df["factuality_verdict"] == "ERROR").sum()
    )
    print("Report:", output_path)

    return df
