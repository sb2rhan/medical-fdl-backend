from app.schemas.predict import PredictRequest, PredictResponse
from app.services.model_adapter import ModelAdapter


class PredictionService:
    def __init__(self):
        self.model_adapter = ModelAdapter()

    def predict(self, payload: PredictRequest) -> PredictResponse:
        # Adapter returns a real PredictResponse — no field remapping needed
        return self.model_adapter.run_inference(payload)
