import asyncio
import torch
import concurrent.futures
from fastapi import APIRouter, HTTPException, Depends
from app.schemas.predict import PredictRequest
from app.model_loader import load_artifacts
from app.inference import run_single
from app.core.auth import require_api_key

router = APIRouter(prefix="/api", tags=["explain"])
_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def _get_membership_params(artifacts: dict) -> list[dict]:
    """
    Extract ANFIS membership function parameters for each selected feature.
    Matches the actual ANFIS implementation in model.architecture.ANFIS.
    """
    try:
        anfis = artifacts["model"].anfis
        feature_names = artifacts["features"]
        n_fuzzy_sets = artifacts["n_fuzzy_sets"]

        # mf_mix is a single global scalar; convert to [0,1] blend weight
        global_blend = round(float(torch.sigmoid(anfis.mf_mix).item()), 4)

        mf_params = []
        labels = ["LOW", "MED", "HIGH"]

        for i, name in enumerate(feature_names):
            entry = {"feature": name, "sets": []}

            for s in range(n_fuzzy_sets):
                entry["sets"].append({
                    "set_index": s,
                    "label": labels[s] if s < len(labels) else f"SET_{s}",
                    "gaussian": {
                        "mu": round(float(anfis.mu_gauss[i, s].item()), 4),
                        "sigma": round(float(anfis.sigma_gauss[i, s].item()), 4),
                    },
                    "trapezoid": {
                        "a": round(float(anfis.trap_a[i, s].item()), 4),
                        "b": round(float(anfis.trap_b[i, s].item()), 4),
                        "c": round(float(anfis.trap_c[i, s].item()), 4),
                        "d": round(float(anfis.trap_d[i, s].item()), 4),
                    },
                    "blend_weight": global_blend,
                })

            mf_params.append(entry)

        return mf_params

    except Exception as e:
        return [{"error": f"membership extraction failed: {str(e)}"}]


@router.post("/rules", dependencies=[Depends(require_api_key)])
async def explain_rules(req: PredictRequest):
    """Returns fired fuzzy rules + full membership function parameters per feature."""
    arts = load_artifacts()
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_pool, run_single, req, arts),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timed out.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "subject_id":             result.subject_id,
        "prediction":             result.prediction,
        "probability":            result.probability,
        "anfis_rules":            result.anfis_rules,
        "fusion_weights":         result.fusion_weights,
        "modality_status":        result.modality_status,
        "clinical_features_used": arts["features"],
        "membership_functions":   _get_membership_params(arts),
    }
