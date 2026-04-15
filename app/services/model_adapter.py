from app.schemas.predict import PredictRequest, PredictResponse
from app.model_loader import load_artifacts
from app.inference import run_single


class ModelAdapter:
    """
    Adapter between PredictionService and the real inference pipeline.
    Replaces the old placeholder stub that returned hardcoded fake outputs.
    """

    def run_inference(self, payload: PredictRequest) -> PredictResponse:
        arts = load_artifacts()
        return run_single(payload, arts)
