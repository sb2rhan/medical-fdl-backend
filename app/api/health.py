import torch
from fastapi import APIRouter
from app.model_loader import _ARTIFACTS

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check():
    checks: dict[str, str] = {"model": "ok" if _ARTIFACTS is not None else "not_loaded", "cuda": (
        f"ok ({torch.cuda.get_device_name(0)})"
        if torch.cuda.is_available() else "cpu_only"
    )}

    try:
        from app.services.chroma_store import ChromaStore
        ChromaStore().client.heartbeat()
        checks["chromadb"] = "ok"
    except Exception as e:
        checks["chromadb"] = f"error: {e}"

    overall = "ok" if all(
        v.startswith("ok") or v == "cpu_only" for v in checks.values()
    ) else "degraded"
    return {"status": overall, "checks": checks}
