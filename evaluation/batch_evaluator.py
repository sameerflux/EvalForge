
import pandas as pd
from pathlib import Path

from evaluation.evaluation_engine import calculate_f1
from evaluation.metrics.semantic_similarity import calculate_similarity
from evaluation.judges.llm_judge import evaluate_answer


def evaluate_batch(
    input_file,
    output_file,
    client,
    judge_model="openai/gpt-oss-20b",
    max_rows=None
):
    df = pd.read_csv(input_file)

    required_columns = [
        "question",
        "expected_answer",
        "actual_response"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if max_rows is not None:
        df = df.head(max_rows).copy()
    else:
        df = df.copy()

    token_scores = []
    semantic_scores = []
    judge_scores = []
    judge_reasons = []
    judge_verdicts = []

    for index, row in df.iterrows():

        question = str(row["question"])
        expected = str(row["expected_answer"])
        actual = str(row["actual_response"])

        if any(
            pd.isna(row[col]) or not str(row[col]).strip()
            for col in required_columns
        ):
            token_scores.append(None)
            semantic_scores.append(None)
            judge_scores.append(None)
            judge_reasons.append("Missing input data")
            judge_verdicts.append("ERROR")
            continue

        print(f"Evaluating question {index + 1}...")

        # 1. Token F1
        token_score = calculate_f1(expected, actual)

        # 2. Semantic Similarity
        semantic_score = calculate_similarity(expected, actual)

        # 3. LLM Judge
        judge_result = evaluate_answer(
            client=client,
            question=question,
            expected_answer=expected,
            actual_answer=actual,
            model=judge_model
        )

        token_scores.append(float(token_score))
        semantic_scores.append(float(semantic_score))
        judge_scores.append(judge_result["score"])
        judge_reasons.append(judge_result["reason"])
        judge_verdicts.append(judge_result["verdict"])

    df["token_f1_score"] = token_scores
    df["semantic_similarity_score"] = semantic_scores
    df["judge_score"] = judge_scores
    df["judge_reason"] = judge_reasons
    df["judge_verdict"] = judge_verdicts

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("\nBatch evaluation completed!")
    print("Questions evaluated:", len(df))
    print("Report saved to:", output_path)

    return df
