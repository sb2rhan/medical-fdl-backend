from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api import predict, copilot, explain, health, predict_scan


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load main model + extractor backbone at startup
    from app.model_loader import load_artifacts
    from app.services.scan_extractor import get_backbone
    arts = load_artifacts()
    get_backbone(arts["device"])   # warm up MedicalResNet3D on the same device
    yield


app = FastAPI(
    title="Medical FDL Backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(predict_scan.router)
app.include_router(explain.router)
app.include_router(copilot.router)
