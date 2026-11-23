
from pydantic import BaseModel, Field


class ExperimentItem(BaseModel):
    question: str = Field(..., min_length=1)
    expected_answer: str = Field(..., min_length=1)
    actual_answer: str = Field(..., min_length=1)


class ExperimentRunRequest(BaseModel):
    experiment_name: str = Field(..., min_length=1)
    model: str = "openai/gpt-oss-20b"
    items: list[ExperimentItem] = Field(
        ...,
        min_length=1,
        max_length=100
    )


class ExperimentItemResult(BaseModel):
    question: str
    token_f1_score: float
    semantic_similarity_score: float
    judge_score: int | None
    judge_reason: str
    verdict: str


class ExperimentRunResponse(BaseModel):
    experiment_id: int | None = None
    experiment_name: str
    model: str
    total_items: int
    passed: int
    failed: int
    average_token_f1: float
    average_semantic_similarity: float
    average_judge_score: float | None
    results: list[ExperimentItemResult]
