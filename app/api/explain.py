import asyncio
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
    Returns Gaussian (mu, sigma), trapezoidal (a,b,c,d), and blend weight
    for every feature x fuzzy-set combination.
    """
    try:
        anfis = artifacts["model"].anfis
        feature_names = artifacts["features"]
        n_fuzzy_sets = artifacts["n_fuzzy_sets"]
        mf_params = []
        for i, name in enumerate(feature_names):
            entry = {"feature": name, "sets": []}
            for s in range(n_fuzzy_sets):
                try:
                    entry["sets"].append({
                        "set_index": s,
                        "label": ["LOW", "MED", "HIGH"][s],
                        "gaussian": {
                            "mu":    round(float(anfis.gauss_mu[i, s].item()), 4),
                            "sigma": round(float(anfis.gauss_sigma[i, s].item()), 4),
                        },
                        "trapezoid": {
                            "a": round(float(anfis.trap_params[i, s, 0].item()), 4),
                            "b": round(float(anfis.trap_params[i, s, 1].item()), 4),
                            "c": round(float(anfis.trap_params[i, s, 2].item()), 4),
                            "d": round(float(anfis.trap_params[i, s, 3].item()), 4),
                        },
                        "blend_weight": round(float(anfis.blend[i, s].item()), 4),
                    })
                except Exception:
                    entry["sets"].append({"set_index": s, "error": "params unavailable"})
            mf_params.append(entry)
        return mf_params
    except Exception:
        return []  # graceful degradation if ANFIS attribute path differs


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
