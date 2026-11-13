
import json


def evaluate_answer(
    client,
    question,
    expected_answer,
    actual_answer,
    model="openai/gpt-oss-20b"
):
    """Evaluate an answer using a strict LLM-as-a-Judge rubric."""

    if not str(actual_answer).strip():
        return {
            "score": 1,
            "reason": "The actual answer is empty.",
            "verdict": "FAIL"
        }

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": """
You are a strict and unbiased AI evaluation judge.

Evaluate the actual answer against the question
and expected answer.

Evaluate these four dimensions:

1. Factual correctness:
   Are the claims factually correct?

2. Relevance:
   Does the answer address the question directly?

3. Coverage:
   Does it include the important information
   present in the expected answer?

4. Unsupported claims:
   Does it contain misleading, contradictory,
   or unsupported factual claims?

IMPORTANT RULES:

- Judge meaning, not exact wording.
- Do not penalize correct explanations merely
  because they are longer than the expected answer.
- Do not reward length or technical vocabulary.
- Penalize incorrect or contradictory claims.
- Do not assume that an answer is correct simply
  because it sounds confident.
- If the core answer is missing, do not give 4 or 5.
- Evaluate the complete answer, not just its first sentence.

SCORING RUBRIC:

1 = Completely incorrect or irrelevant.
2 = Mostly incorrect, with major factual problems.
3 = Partially correct, but has important omissions
    or errors.
4 = Correct and relevant, with minor omissions.
5 = Fully correct, relevant, and sufficiently complete.

VERDICT:

PASS = Score 4 or 5.
FAIL = Score 1, 2, or 3.

Return valid JSON only:

{
    "score": 1,
    "reason": "Specific explanation of the score",
    "verdict": "FAIL"
}
"""
                },
                {
                    "role": "user",
                    "content": f"""
Question:
{question}

Expected answer:
{expected_answer}

Actual answer:
{actual_answer}
"""
                }
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        result = json.loads(
            response.choices[0].message.content
        )

        score = int(result["score"])

        if score not in [1, 2, 3, 4, 5]:
            raise ValueError("Invalid judge score.")

        if not isinstance(result.get("reason"), str):
            raise ValueError("Invalid judge reason.")

        return {
            "score": score,
            "reason": result["reason"],
            "verdict": "PASS" if score >= 4 else "FAIL"
        }

    except Exception as e:
        return {
            "score": None,
            "reason": f"Evaluation error: {str(e)}",
            "verdict": "ERROR"
        }
