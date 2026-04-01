from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.model_loader import load_artifacts
from app.api.health import router as health_router
from app.api.predict import router as predict_router
from app.api.rag_debug import router as rag_debug_router
from app.api.llm_debug import router as llm_debug_router
from app.api.copilot import router as copilot_router
from app.api.explain import router as explain_router
from app.api.metadata import router as metadata_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()          # warm up model on startup
    yield

app = FastAPI(title="Medical FDL API", description="An API to get model predictions", version="1.0.0", lifespan=lifespan)

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(rag_debug_router)
app.include_router(llm_debug_router)
app.include_router(copilot_router)
app.include_router(explain_router)
app.include_router(metadata_router)

@app.get("/")
def root():
    return {"message": "Welcome to the Medical FDL API"}


