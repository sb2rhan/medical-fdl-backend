from fastapi import APIRouter, HTTPException
from app.schemas import PredictRequest, PredictResponse, BatchPredictRequest, BatchPredictResponse
from app.model_loader import load_artifacts
from app.inference import run_single
import asyncio, concurrent.futures

router = APIRouter(prefix="/api", tags=["predict"])
_pool  = concurrent.futures.ThreadPoolExecutor(max_workers=4)

@router.post("/predict", response_model=PredictResponse)
async def predict_single(req: PredictRequest):
    arts = load_artifacts()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_pool, run_single, req, arts)

@router.post("/predict-batch", response_model=BatchPredictResponse)
async def predict_batch(req: BatchPredictRequest):
    if len(req.samples) > 64:
        raise HTTPException(status_code=400, detail="Max 64 samples per batch.")
    arts = load_artifacts()
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(_pool, run_single, s, arts) for s in req.samples]
    results = await asyncio.gather(*tasks)
    return BatchPredictResponse(results=list(results))