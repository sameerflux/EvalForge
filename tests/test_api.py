
from fastapi.testclient import TestClient
from unittest.mock import patch

from backend.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["project"] == "EvalForge"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_token_f1_endpoint():
    response = client.post(
        "/api/evaluation/token-f1",
        json={
            "expected_answer": "Python is a programming language",
            "actual_answer": "Python is a programming language"
        }
    )

    assert response.status_code == 200
    assert response.json()["f1_score"] == 1.0


def test_semantic_similarity_endpoint():
    with patch(
        "backend.main.calculate_similarity",
        return_value=0.95
    ):
        response = client.post(
            "/api/evaluation/semantic-similarity",
            json={
                "expected_answer": "Python is a programming language",
                "actual_answer": "Python is a programming language"
            }
        )

    assert response.status_code == 200
    assert response.json()["similarity_score"] == 0.95


def test_llm_judge_endpoint():
    mock_result = {
        "score": 5,
        "reason": "The answer is correct.",
        "verdict": "PASS"
    }

    with patch(
        "backend.main.evaluate_answer",
        return_value=mock_result
    ), patch(
        "backend.main.Groq"
    ), patch.dict(
        "os.environ",
        {"GROQ_API_KEY": "test-key"}
    ):
        response = client.post(
            "/api/evaluation/llm-judge",
            json={
                "question": "What is Python?",
                "expected_answer": "Python is a programming language.",
                "actual_answer": "Python is a programming language."
            }
        )

    assert response.status_code == 200
    assert response.json()["score"] == 5
    assert response.json()["verdict"] == "PASS"


def test_unified_evaluation_endpoint():
    mock_result = {
        "score": 5,
        "reason": "The answer is correct.",
        "verdict": "PASS"
    }

    with patch(
        "backend.main.calculate_similarity",
        return_value=0.95
    ), patch(
        "backend.main.evaluate_answer",
        return_value=mock_result
    ), patch(
        "backend.main.Groq"
    ), patch.dict(
        "os.environ",
        {"GROQ_API_KEY": "test-key"}
    ):
        response = client.post(
            "/api/evaluation/unified",
            json={
                "question": "What is Python?",
                "expected_answer": "Python is a programming language.",
                "actual_answer": "Python is a programming language."
            }
        )

    assert response.status_code == 200
    assert response.json()["judge_score"] == 5
    assert response.json()["verdict"] == "PASS"


def test_llm_judge_missing_api_key():
    with patch.dict("os.environ", {}, clear=True):
        response = client.post(
            "/api/evaluation/llm-judge",
            json={
                "question": "What is Python?",
                "expected_answer": "A programming language.",
                "actual_answer": "A programming language."
            }
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "GROQ_API_KEY is not configured."


def test_token_f1_missing_field():
    response = client.post(
        "/api/evaluation/token-f1",
        json={
            "expected_answer": "Python is a programming language."
        }
    )

    assert response.status_code == 422


def test_token_f1_empty_answer():
    response = client.post(
        "/api/evaluation/token-f1",
        json={
            "expected_answer": "Python is a programming language.",
            "actual_answer": ""
        }
    )

    assert response.status_code == 422
