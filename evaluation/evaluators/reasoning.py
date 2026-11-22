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

Evaluate the observable reasoning quality and correctness of the model answer.

QUESTION:
{question}

REFERENCE ANSWER:
{expected_answer}

MODEL ANSWER:
{actual_answer}

Evaluation criteria:
1. Does the answer correctly address the question?
2. Are its main statements factually correct?
3. Is the explanation logically consistent?
4. Does the explanation support its conclusion?
5. Are there important omissions, contradictions, or reasoning errors?

Scoring rules:
- Compare the meaning of the model answer with the reference answer, not just the wording.
- Accept alternative explanations and valid approaches when they correctly answer the question.
- Do not require every detail from the reference answer if the model answer is still correct.
- Distinguish missing information from incorrect information.
- A correct central idea with an important missing detail should generally receive partial credit, not the lowest score.
- Do not treat uncertainty alone as proof that the entire answer is incorrect. Evaluate the correctness of the information actually provided.
- Do not reward an answer that merely repeats related terminology without answering the question.
- Evaluate only the explanation visible in the answer. Do not request or infer hidden chain-of-thought.
- Do not penalize concise but correct answers.

Score:
5 = Correct answer with sound, relevant reasoning and no significant gaps.
4 = Correct answer with only minor omissions or weaknesses.
3 = Partially correct answer with a correct central idea but an important omission or gap.
2 = Mostly incorrect answer with limited correct information.
1 = Fundamentally incorrect, irrelevant, or logically contradictory answer.

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
