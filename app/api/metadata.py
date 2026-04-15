from fastapi import APIRouter
from app.schemas.predict import MetadataResponse
from app.model_loader import load_artifacts

router = APIRouter(prefix="/api", tags=["metadata"])


@router.get("", response_model=MetadataResponse)
async def get_metadata():
    arts = load_artifacts()
    return MetadataResponse(
        dataset_name=arts["dataset"],
        modalities=[m.name for m in arts["modality_configs"]],
        clinical_features=arts["features"],
        n_fuzzy_sets=arts["n_fuzzy_sets"],
        threshold=arts["threshold"],
        target_names=arts["target_names"],
        device=str(arts["device"]),
        best_epoch=arts["best_epoch"],
        val_bacc=arts["val_bacc"],
        val_auc=arts["val_auc"],
        calibrator_loaded=arts["calibrator"] is not None,
    )
