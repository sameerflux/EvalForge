
import json
import re


def evaluate_factuality(
    client,
    question,
    reference_answer,
    actual_answer,
    model="openai/gpt-oss-20b"
):
    """
    Evaluate factual consistency between an AI answer
    and a reference answer.

    Scores:
        5 = Fully supported
        4 = Mostly supported
        3 = Partially supported
        2 = Mostly unsupported
        1 = Contradicts the reference
    """

    prompt = f"""
You are an expert factuality evaluator for an AI evaluation platform.

Evaluate the factual consistency of the actual answer against
the reference answer.

IMPORTANT RULES:
1. Do not penalize valid additional information just because it
   is absent from the reference.
2. Identify claims that are unsupported by the reference.
3. Identify claims that directly contradict the reference.
4. Distinguish unsupported claims from demonstrably false claims.
5. Evaluate the meaning, not just matching words.

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

ACTUAL ANSWER:
{actual_answer}

Return ONLY valid JSON using this structure:

{{
    "score": 1,
    "verdict": "unsupported",
    "supported_claims": [],
    "unsupported_claims": [],
    "incorrect_claims": [],
    "explanation": "Explain your evaluation."
}}

SCORING:
5 = Fully supported and factually consistent.
4 = Mostly supported, with minor unsupported details.
3 = Partially supported, with some unsupported claims.
2 = Mostly unsupported or containing significant inaccuracies.
1 = Directly contradicts the reference or is fundamentally incorrect.

Allowed verdicts:
supported, partially_supported, unsupported.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a precise factuality evaluator. Return valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    # Extract JSON if the response contains extra formatting
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)

        if not match:
            raise ValueError(
                f"Could not extract JSON from evaluator response: {content}"
            )

        result = json.loads(match.group())

    required_fields = [
        "score",
        "verdict",
        "supported_claims",
        "unsupported_claims",
        "incorrect_claims",
        "explanation"
    ]

    for field in required_fields:
        if field not in result:
            raise ValueError(f"Missing evaluator field: {field}")

    score = int(result["score"])

    if score < 1 or score > 5:
        raise ValueError("Factuality score must be between 1 and 5.")

    if result["verdict"] not in [
        "supported",
        "partially_supported",
        "unsupported"
    ]:
        raise ValueError("Invalid factuality verdict.")

    return result
