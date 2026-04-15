# DEBUG ONLY — do NOT register this router in main.py for production deployments.
# Exposes ChromaDB retrieval results without authentication for local development only.
# To test locally: temporarily add `app.include_router(rag_debug.router)` in main.py.
from fastapi import APIRouter, Query
from app.services.chroma_store import ChromaStore

router = APIRouter(prefix="/api", tags=["rag-debug"])


@router.post("/rag/retrieve")
def retrieve(question: str = Query(...), k: int = Query(3, ge=1, le=10)):
    store = ChromaStore()
    store.seed_if_empty()
    results = store.query(question=question, k=k)

    return {
        "question": question,
        "matches": results,
    }
