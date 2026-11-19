
import os

from fastapi import FastAPI, HTTPException
from groq import Groq

from backend.schemas.evaluation import (
    TokenF1Request,
    TokenF1Response,
    SemanticSimilarityRequest,
    SemanticSimilarityResponse,
    LLMJudgeRequest,
    LLMJudgeResponse,
    UnifiedEvaluationRequest,
    UnifiedEvaluationResponse
)

from evaluation.evaluation_engine import calculate_f1
from evaluation.metrics.semantic_similarity import calculate_similarity
from evaluation.judges.llm_judge import evaluate_answer


app = FastAPI(
    title="EvalForge API",
    description="LLM Evaluation and Benchmarking Platform",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "project": "EvalForge",
        "message": "Welcome to EvalForge API",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "EvalForge Backend"
    }


@app.post(
    "/api/evaluation/token-f1",
    response_model=TokenF1Response
)
def calculate_token_f1(request: TokenF1Request):
    score = calculate_f1(
        request.expected_answer,
        request.actual_answer
    )

    return {
        "f1_score": float(score),
        "metric": "Token F1"
    }


@app.post(
    "/api/evaluation/semantic-similarity",
    response_model=SemanticSimilarityResponse
)
def calculate_semantic_similarity(request: SemanticSimilarityRequest):
    score = calculate_similarity(
        request.expected_answer,
        request.actual_answer
    )

    return {
        "similarity_score": score,
        "metric": "Semantic Similarity"
    }


@app.post(
    "/api/evaluation/llm-judge",
    response_model=LLMJudgeResponse
)
def judge_answer(request: LLMJudgeRequest):

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured."
        )

    client = Groq(api_key=api_key)

    return evaluate_answer(
        client=client,
        question=request.question,
        expected_answer=request.expected_answer,
        actual_answer=request.actual_answer,
        model=request.model
    )


@app.post(
    "/api/evaluation/unified",
    response_model=UnifiedEvaluationResponse
)
def unified_evaluation(request: UnifiedEvaluationRequest):

    # 1. Token F1
    token_f1 = calculate_f1(
        request.expected_answer,
        request.actual_answer
    )

    # 2. Semantic Similarity
    semantic_score = calculate_similarity(
        request.expected_answer,
        request.actual_answer
    )

    # 3. LLM Judge
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured."
        )

    client = Groq(api_key=api_key)

    judge_result = evaluate_answer(
        client=client,
        question=request.question,
        expected_answer=request.expected_answer,
        actual_answer=request.actual_answer,
        model=request.model
    )

    return {
        "token_f1_score": float(token_f1),
        "semantic_similarity_score": float(semantic_score),
        "judge_score": judge_result["score"],
        "judge_reason": judge_result["reason"],
        "verdict": judge_result["verdict"]
    }
