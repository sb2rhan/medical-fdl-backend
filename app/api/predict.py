import asyncio
import concurrent.futures
from fastapi import APIRouter, HTTPException, Depends
from app.schemas.predict import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
)
from app.model_loader import load_artifacts
from app.inference import run_single
from app.core.auth import require_api_key

router = APIRouter(prefix="/api", tags=["predict"])
# 1 worker: multiple parallel threads on 1 GPU cause contention, not speedup
_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)


@router.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
async def predict_single(req: PredictRequest):
    arts = load_artifacts()
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_pool, run_single, req, arts),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timed out.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-batch", response_model=BatchPredictResponse, dependencies=[Depends(require_api_key)])
async def predict_batch(req: BatchPredictRequest):
    if len(req.samples) > 64:
        raise HTTPException(status_code=400, detail="Max 64 samples per batch.")
    arts = load_artifacts()
    loop = asyncio.get_event_loop()
    results = []
    # Sequential on single GPU: parallel gather would serialize anyway
    # and create thread contention
    for sample in req.samples:
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(_pool, run_single, sample, arts),
                timeout=60.0,
            )
            results.append(result)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Inference timed out on sample '{sample.subject_id}'.",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return BatchPredictResponse(results=results)
