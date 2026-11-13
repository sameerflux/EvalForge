import json


def evaluate_relevance(
    client,
    question,
    actual_answer,
    model="openai/gpt-oss-20b"
):
    """Evaluate how relevant an answer is to the question."""

    if not str(actual_answer).strip():
        return {
            "relevance_score": 1,
            "relevance_verdict": "IRRELEVANT",
            "relevance_reason": "The answer is empty."
        }

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": """
You are a strict answer relevance evaluator.

Evaluate whether the actual answer addresses the question.

Evaluate relevance only, not factual correctness.

Scoring:
1 = Completely unrelated.
2 = Mostly unrelated.
3 = Partially relevant.
4 = Clearly relevant.
5 = Directly and completely relevant.

Do not reward unrelated extra information.
Do not reward technical words without relevance.

Return valid JSON only:
{
    "score": 1,
    "reason": "Explanation"
}
"""
                },
                {
                    "role": "user",
                    "content": f"""
Question: {question}

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
            raise ValueError("Invalid relevance score.")

        return {
            "relevance_score": score,
            "relevance_verdict": (
                "RELEVANT" if score >= 4 else "IRRELEVANT"
            ),
            "relevance_reason": str(result["reason"])
        }

    except Exception as e:
        return {
            "relevance_score": None,
            "relevance_verdict": "ERROR",
            "relevance_reason": f"Evaluation error: {str(e)}"
        }
