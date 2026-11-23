
import os

from fastapi import APIRouter, HTTPException
from groq import Groq

from backend.schemas.experiment import (
    ExperimentRunRequest,
    ExperimentRunResponse,
    ExperimentItemResult
)

from backend.database import (
    initialize_database,
    save_experiment,
    get_all_experiments,
    get_experiment_by_id
)

from evaluation.evaluation_engine import calculate_f1
from evaluation.metrics.semantic_similarity import calculate_similarity
from evaluation.judges.llm_judge import evaluate_answer


router = APIRouter()


@router.post(
    "/api/experiments/run",
    response_model=ExperimentRunResponse
)
def run_experiment(request: ExperimentRunRequest):

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured."
        )

    client = Groq(api_key=api_key)

    results = []

    for item in request.items:

        token_f1 = calculate_f1(
            item.expected_answer,
            item.actual_answer
        )

        semantic_score = calculate_similarity(
            item.expected_answer,
            item.actual_answer
        )

        judge_result = evaluate_answer(
            client=client,
            question=item.question,
            expected_answer=item.expected_answer,
            actual_answer=item.actual_answer,
            model=request.model
        )

        results.append(
            ExperimentItemResult(
                question=item.question,
                token_f1_score=float(token_f1),
                semantic_similarity_score=float(semantic_score),
                judge_score=judge_result["score"],
                judge_reason=judge_result["reason"],
                verdict=judge_result["verdict"]
            )
        )

    total = len(results)

    passed = sum(
        1 for result in results
        if result.verdict == "PASS"
    )

    failed = total - passed

    average_token_f1 = sum(
        result.token_f1_score for result in results
    ) / total

    average_semantic = sum(
        result.semantic_similarity_score for result in results
    ) / total

    valid_judge_scores = [
        result.judge_score
        for result in results
        if result.judge_score is not None
    ]

    average_judge = (
        sum(valid_judge_scores) / len(valid_judge_scores)
        if valid_judge_scores else None
    )

    response = ExperimentRunResponse(
        experiment_name=request.experiment_name,
        model=request.model,
        total_items=total,
        passed=passed,
        failed=failed,
        average_token_f1=average_token_f1,
        average_semantic_similarity=average_semantic,
        average_judge_score=average_judge,
        results=results
    )

    try:
        initialize_database()

        experiment_id = save_experiment(
            response.model_dump(exclude={"experiment_id"})
        )

        response.experiment_id = experiment_id

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save experiment: {str(error)}"
        )

    return response



@router.get("/api/experiments")
def list_experiments():

    try:
        initialize_database()
        return get_all_experiments()

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve experiments: {str(error)}"
        )


@router.get("/api/experiments/{experiment_id}")
def retrieve_experiment(experiment_id: int):

    try:
        initialize_database()
        experiment = get_experiment_by_id(experiment_id)

        if experiment is None:
            raise HTTPException(
                status_code=404,
                detail="Experiment not found."
            )

        return experiment

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve experiment: {str(error)}"
        )
