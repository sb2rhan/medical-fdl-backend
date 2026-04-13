import os

from fastapi import FastAPI
from contextlib import asynccontextmanager

from fastapi.middleware.cors import CORSMiddleware
from app.model_loader import load_artifacts
from app.api.health import router as health_router
from app.api.predict import router as predict_router
from app.api.copilot import router as copilot_router
from app.api.explain import router as explain_router
from app.api.metadata import router as metadata_router

_DEBUG = os.getenv("APP_ENV", "production").lower() != "production"

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()         # loads model + warm-up forward pass
    yield

app = FastAPI(
    title="Medical FDL API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if _DEBUG else None,   # hide Swagger in production
    redoc_url="/redoc" if _DEBUG else None,
)

_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(copilot_router)
app.include_router(explain_router)
app.include_router(metadata_router)

# FIX: debug routes only active when APP_ENV != "production"
if _DEBUG:
    from app.api.rag_debug import router as rag_debug_router
    from app.api.llm_debug import router as llm_debug_router
    app.include_router(rag_debug_router)
    app.include_router(llm_debug_router)

@app.get("/")
def root():
    return {"message": "Welcome to the Medical FDL API"}


