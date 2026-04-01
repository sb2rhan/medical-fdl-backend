from fastapi import APIRouter
from app.schemas import PredictRequest
from app.model_loader import load_artifacts
from app.inference import run_single

router = APIRouter(prefix="/api", tags=["explain"])

@router.post("/rules")
async def explain_rules(req: PredictRequest):
    """Returns full fired fuzzy rule table for a single subject."""
    arts    = load_artifacts()
    result  = run_single(req, arts)
    return {
        "subject_id"     : result.subject_id,
        "anfis_rules"    : result.anfis_rules,
        "fusion_weights" : result.fusion_weights,
        "modality_status": result.modality_status,
        "clinical_features_used": arts["features"],
    }