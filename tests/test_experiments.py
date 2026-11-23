from fastapi.testclient import TestClient
from unittest.mock import patch

from backend.main import app

client = TestClient(app)


def test_experiment_execution_success():

    mock_result = {
        "score": 5,
        "reason": "The answer is correct.",
        "verdict": "PASS"
    }

    with patch(
        "backend.routes.experiments.Groq"
    ), patch(
        "backend.routes.experiments.calculate_similarity",
        return_value=0.95
    ), patch(
        "backend.routes.experiments.evaluate_answer",
        return_value=mock_result
    ), patch.dict(
        "os.environ",
        {"GROQ_API_KEY": "test-key"}
    ):

        response = client.post(
            "/api/experiments/run",
            json={
                "experiment_name": "Automated Experiment Test",
                "items": [
                    {
                        "question": "What is Python?",
                        "expected_answer": "A programming language.",
                        "actual_answer": "A programming language."
                    },
                    {
                        "question": "What is AI?",
                        "expected_answer": "Artificial Intelligence.",
                        "actual_answer": "Artificial Intelligence."
                    }
                ]
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert data["total_items"] == 2
    assert data["passed"] == 2
    assert data["failed"] == 0
    assert len(data["results"]) == 2


def test_experiment_missing_api_key():

    with patch.dict("os.environ", {}, clear=True):

        response = client.post(
            "/api/experiments/run",
            json={
                "experiment_name": "Missing Key Test",
                "items": [
                    {
                        "question": "What is Python?",
                        "expected_answer": "A programming language.",
                        "actual_answer": "A programming language."
                    }
                ]
            }
        )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "GROQ_API_KEY is not configured."
    )


def test_experiment_empty_items():

    response = client.post(
        "/api/experiments/run",
        json={
            "experiment_name": "Empty Dataset Test",
            "items": []
        }
    )

    assert response.status_code == 422


def test_get_experiment_history():

    mock_history = [
        {
            "id": 101,
            "experiment_name": "History API Test"
        }
    ]

    with patch(
        "backend.routes.experiments.initialize_database"
    ), patch(
        "backend.routes.experiments.get_all_experiments",
        return_value=mock_history
    ):

        response = client.get("/api/experiments")

    assert response.status_code == 200
    assert response.json() == mock_history


def test_get_experiment_by_id():

    mock_experiment = {
        "id": 101,
        "experiment_name": "History API Test",
        "results": []
    }

    with patch(
        "backend.routes.experiments.initialize_database"
    ), patch(
        "backend.routes.experiments.get_experiment_by_id",
        return_value=mock_experiment
    ):

        response = client.get("/api/experiments/101")

    assert response.status_code == 200
    assert response.json()["id"] == 101
    assert response.json()["experiment_name"] == "History API Test"


def test_get_missing_experiment():

    with patch(
        "backend.routes.experiments.initialize_database"
    ), patch(
        "backend.routes.experiments.get_experiment_by_id",
        return_value=None
    ):

        response = client.get("/api/experiments/99999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Experiment not found."
