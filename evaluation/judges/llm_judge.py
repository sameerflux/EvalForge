
import json


def evaluate_answer(
    client,
    question,
    expected_answer,
    actual_answer,
    model="openai/gpt-oss-20b"
):
    """Evaluate answers using a calibrated LLM-as-a-Judge rubric."""

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
You are an objective and calibrated AI evaluation judge.

Evaluate factual correctness, relevance, and completeness.

IMPORTANT PRINCIPLES:
- Judge meaning, not exact wording.
- Accept factually correct alternative explanations.
- The expected answer is a reference, not the only
  acceptable wording.
- Do not penalize correct answers for being longer
  or shorter than the expected answer.
- Distinguish minor omissions from major errors.
- Penalize false, contradictory, or misleading claims.
- Evaluate the complete answer in context.
- Never invent errors that are not present.

SCORING RUBRIC:

5 = Fully correct, relevant, and sufficiently complete.
4 = Correct overall, with minor omissions.
3 = Partially correct, with meaningful omissions or errors.
2 = Mostly incorrect, with major factual errors.
1 = Completely incorrect, irrelevant, or fundamentally wrong.

CALIBRATION:
- Correct answers expressed in different words must not
  receive low scores merely due to wording differences.
- Missing a minor detail should generally result in 4.
- Missing essential information should generally result in 3.
- A fundamentally wrong central claim should receive 1 or 2.
- Do not award 5 when important information is missing.

VERDICT:
PASS = Score 4 or 5.
FAIL = Score 1, 2, or 3.

Return valid JSON only:
{
    "score": 4,
    "reason": "Specific explanation of the evaluation.",
    "verdict": "PASS"
}
"""
                },
                {
                    "role": "user",
                    "content": f"""
Question: {question}

Expected answer: {expected_answer}

Actual answer: {actual_answer}
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
