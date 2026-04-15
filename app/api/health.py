import torch
from fastapi import APIRouter

router = APIRouter(tags=["health"])


def _model_ready() -> bool:
    """Non-raising check — True only if artifacts are already loaded in memory."""
    try:
        from app.model_loader import _ARTIFACTS
        return _ARTIFACTS is not None
    except Exception:
        return False


@router.get("/health")
def health_check():
    checks: dict[str, str] = {
        "model": "ok" if _model_ready() else "not_loaded",
        "cuda": (
            f"ok ({torch.cuda.get_device_name(0)})"
            if torch.cuda.is_available() else "cpu_only"
        ),
    }

    try:
        from app.services.chroma_store import ChromaStore
        ChromaStore().client.heartbeat()
        checks["chromadb"] = "ok"
    except Exception as e:
        checks["chromadb"] = f"error: {e}"

    # FIX: also verify the LLM API key is configured
    try:
        from app.core.config import settings
        checks["llm_key"] = "ok" if settings.NVIDIA_API_KEY else "missing"
    except Exception:
        checks["llm_key"] = "missing"

    overall = "ok" if all(
        v.startswith("ok") or v == "cpu_only" for v in checks.values()
    ) else "degraded"
    return {"status": overall, "checks": checks}
