from pydantic import BaseModel, Field
from app.schemas.predict import PredictResponse


class CopilotRequest(BaseModel):
    question: str
    # Typed as PredictResponse so the copilot always receives a validated model output
    explanation_payload: PredictResponse


class CopilotResponse(BaseModel):
    summary: str
    model_rationale: str
    evidence: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    limitations: str = ""
    uncertainty: str = ""
