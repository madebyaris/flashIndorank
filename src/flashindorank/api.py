"""FastAPI application exposing the reranker over HTTP."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, List

from fastapi import FastAPI, HTTPException

from . import reranker
from .config import settings
from .models import list_models
from .schemas import (
    CascadeRequestBody,
    HealthResponse,
    ModelDescription,
    RerankRequestBody,
    RerankResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.preload_models:
        reranker.warmup(settings.preload_models)
    yield


app = FastAPI(
    title="flashIndorank",
    description="Lightweight, high-performance cross-encoder reranker built on FlashRank.",
    version="0.1.0",
    lifespan=lifespan,
)


def _passages_to_payload(passages: List[Any]) -> List[Any]:
    """Convert pydantic passage models / strings into plain inputs for the core."""
    payload: List[Any] = []
    for item in passages:
        if isinstance(item, str):
            payload.append(item)
        else:
            payload.append(item.model_dump())
    return payload


def _loaded_model_names() -> List[str]:
    return sorted({name for (name, _length) in reranker._ranker_cache})  # noqa: SLF001


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", loaded_models=_loaded_model_names())


@app.get("/models", response_model=List[ModelDescription])
def models() -> List[ModelDescription]:
    descriptions = [
        ModelDescription(
            name=m.name,
            size_mb=m.size_mb,
            description=m.description,
            multilingual=m.multilingual,
        )
        for m in list_models()
    ]
    for name, path in settings.custom_models.items():
        descriptions.append(
            ModelDescription(
                name=name,
                size_mb=0.0,
                description=f"Custom local ONNX reranker ({path})",
                multilingual=True,
            )
        )
    return descriptions


@app.post("/rerank", response_model=RerankResponse)
def rerank_endpoint(body: RerankRequestBody) -> RerankResponse:
    if body.model is not None and not reranker.is_known_model(body.model):
        raise HTTPException(status_code=400, detail=f"Unsupported model: {body.model}")

    start = time.perf_counter()
    results = reranker.rerank(
        query=body.query,
        passages=_passages_to_payload(body.passages),
        model_name=body.model,
        top_k=body.top_k,
        max_length=body.max_length,
    )
    took_ms = (time.perf_counter() - start) * 1000
    return RerankResponse(results=results, count=len(results), took_ms=round(took_ms, 3))


@app.post("/rerank/cascade", response_model=RerankResponse)
def rerank_cascade_endpoint(body: CascadeRequestBody) -> RerankResponse:
    for model_name in (body.fast_model, body.strong_model):
        if model_name is not None and not reranker.is_known_model(model_name):
            raise HTTPException(status_code=400, detail=f"Unsupported model: {model_name}")

    start = time.perf_counter()
    results = reranker.rerank_cascade(
        query=body.query,
        passages=_passages_to_payload(body.passages),
        fast_model=body.fast_model,
        strong_model=body.strong_model,
        prune_to=body.prune_to,
        top_k=body.top_k,
        max_length=body.max_length,
    )
    took_ms = (time.perf_counter() - start) * 1000
    return RerankResponse(results=results, count=len(results), took_ms=round(took_ms, 3))
