# DEBUG ONLY — do NOT register this router in main.py for production deployments.
# Exposes raw LLM inference without authentication for local development only.
# To test locally: temporarily add `app.include_router(llm_debug.router)` in main.py.
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.llm_client import LLMClient

router = APIRouter(prefix="/api", tags=["llm-debug"])


class LLMTestRequest(BaseModel):
    prompt: str


@router.post("/llm/test")
def llm_test(payload: LLMTestRequest):
    try:
        client = LLMClient()
        result = client.chat(
            system_prompt="You are a concise assistant.",
            user_prompt=payload.prompt,
        )
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
