
from pydantic import BaseModel, Field, model_validator


class ExperimentItem(BaseModel):
    question: str = Field(..., min_length=1)
    expected_answer: str = Field(..., min_length=1)
    actual_answer: str | None = None


class ExperimentRunRequest(BaseModel):
    experiment_name: str = Field(..., min_length=1)
    model: str = "openai/gpt-oss-20b"
    generate_answer: bool = False

    items: list[ExperimentItem] = Field(
        ...,
        min_length=1,
        max_length=100
    )

    @model_validator(mode="after")
    def validate_answers(self):
        if not self.generate_answer:
            for item in self.items:
                if not item.actual_answer or not item.actual_answer.strip():
                    raise ValueError(
                        "actual_answer is required when generate_answer is False."
                    )
        return self


class ExperimentItemResult(BaseModel):
    question: str
    actual_answer: str
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
