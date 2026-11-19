
from pydantic import BaseModel, Field

class TokenF1Request(BaseModel):
    expected_answer: str = Field(..., min_length=1)
    actual_answer: str = Field(..., min_length=1)

class TokenF1Response(BaseModel):
    f1_score: float
    metric: str


class SemanticSimilarityRequest(BaseModel):
    expected_answer: str = Field(..., min_length=1)
    actual_answer: str = Field(..., min_length=1)


class SemanticSimilarityResponse(BaseModel):
    similarity_score: float
    metric: str


class LLMJudgeRequest(BaseModel):
    question: str = Field(..., min_length=1)
    expected_answer: str = Field(..., min_length=1)
    actual_answer: str = Field(..., min_length=1)
    model: str = "openai/gpt-oss-20b"


class LLMJudgeResponse(BaseModel):
    score: int | None
    reason: str
    verdict: str


class UnifiedEvaluationRequest(BaseModel):
    question: str = Field(..., min_length=1)
    expected_answer: str = Field(..., min_length=1)
    actual_answer: str = Field(..., min_length=1)
    model: str = "openai/gpt-oss-20b"


class UnifiedEvaluationResponse(BaseModel):
    token_f1_score: float
    semantic_similarity_score: float
    judge_score: int | None
    judge_reason: str
    verdict: str
