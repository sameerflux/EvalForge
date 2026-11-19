import json


def evaluate_reasoning(
    client,
    question,
    expected_answer,
    actual_answer,
    model="openai/gpt-oss-20b"
):
    """Evaluate the observable reasoning quality of an LLM answer."""

    prompt = f"""
You are an expert AI evaluation judge.

Evaluate the observable reasoning quality of this answer.

QUESTION:
{question}

REFERENCE ANSWER:
{expected_answer}

MODEL ANSWER:
{actual_answer}

Evaluate:
1. Logical consistency.
2. Correctness of the approach.
3. Relevance to the question.
4. Whether the explanation supports its conclusion.
5. Contradictions or reasoning errors.

Rules:
- Evaluate only the explanation visible in the answer.
- Do not request or infer hidden chain-of-thought.
- Do not penalize concise but correct answers.
- Allow alternative valid approaches.

Score:
5 = Excellent reasoning.
4 = Good reasoning with minor omissions.
3 = Acceptable reasoning with noticeable gaps.
2 = Weak reasoning with significant gaps.
1 = Fundamentally flawed or contradictory reasoning.

Return only valid JSON:
{{
    "score": 1,
    "strengths": ["strength 1"],
    "issues": ["issue 1"],
    "explanation": "Brief justification."
}}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        score = result.get("score")

        if (
            isinstance(score, bool)
            or not isinstance(score, int)
            or not 1 <= score <= 5
        ):
            raise ValueError("Score must be an integer from 1 to 5.")

        strengths = result.get("strengths", [])
        issues = result.get("issues", [])

        # Normalize LLM responses into lists of strings.
        def normalize_items(value):
            if value is None:
                return []
            if isinstance(value, str):
                return [value] if value.strip() else []
            if isinstance(value, list):
                return [str(item) for item in value if item is not None]
            return [str(value)]

        strengths = normalize_items(strengths)
        issues = normalize_items(issues)

        return {
            "reasoning_score": score,
            "reasoning_verdict": (
                "STRONG" if score == 5
                else "ACCEPTABLE" if score in (3, 4)
                else "WEAK"
            ),
            "reasoning_strengths": strengths,
            "reasoning_issues": issues,
            "reasoning_explanation": str(
                result.get("explanation", "")
            )
        }

    except Exception as e:
        return {
            "reasoning_score": None,
            "reasoning_verdict": "ERROR",
            "reasoning_strengths": [],
            "reasoning_issues": [],
            "reasoning_explanation": f"Reasoning evaluation error: {e}"
        }
