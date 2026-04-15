import json
from pydantic import ValidationError
from fastapi import APIRouter, HTTPException, Depends
from app.schemas.copilot import CopilotRequest, CopilotResponse
from app.services.copilot_service import CopilotService
from app.core.auth import require_api_key

router = APIRouter(prefix="/api", tags=["copilot"])


@router.post("/copilot", response_model=CopilotResponse, dependencies=[Depends(require_api_key)])
async def copilot_answer(req: CopilotRequest):
    service = CopilotService()
    try:
        raw, retrieved = await service.generate_answer(
            question=req.question,
            explanation_payload=req.explanation_payload.model_dump(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # FIX: strip markdown fences the LLM sometimes emits before JSON.loads
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = CopilotResponse.model_validate(json.loads(cleaned))
    except (ValidationError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {e}")

    return parsed
