
import pandas as pd
import re

def tokenize(text):
    """Convert text into normalized tokens."""
    text = str(text).lower()
    return re.findall(r"\b\w+\b", text)


def calculate_f1(expected, actual):
    """Calculate token-based F1 similarity."""

    expected_tokens = tokenize(expected)
    actual_tokens = tokenize(actual)

    if not expected_tokens or not actual_tokens:
        return 0.0

    expected_counts = {}
    actual_counts = {}

    for token in expected_tokens:
        expected_counts[token] = expected_counts.get(token, 0) + 1

    for token in actual_tokens:
        actual_counts[token] = actual_counts.get(token, 0) + 1

    common = sum(
        min(count, actual_counts.get(token, 0))
        for token, count in expected_counts.items()
    )

    precision = common / len(actual_tokens)
    recall = common / len(expected_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def evaluate_dataset(input_file, output_file, threshold=0.5):

    df = pd.read_csv(input_file)

    scores = []
    results = []

    for _, row in df.iterrows():

        expected = row["expected_answer"]
        actual = row["actual_response"]

        if pd.isna(actual) or not str(actual).strip():
            score = 0.0
        else:
            score = calculate_f1(expected, actual)

        passed = score >= threshold

        scores.append(round(score, 4))
        results.append("PASS" if passed else "FAIL")

    df["score"] = scores
    df["passed"] = results

    df.to_csv(output_file, index=False)

    print("Evaluation completed!")
    print("------------------------------")
    print("Total questions:", len(df))
    print("Passed:", results.count("PASS"))
    print("Failed:", results.count("FAIL"))
    print("Average F1 score:", round(sum(scores) / len(scores), 4))
    print("Pass threshold:", threshold)
    print("Report saved:", output_file)

    return df
