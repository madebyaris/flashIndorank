"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .models import DEFAULT_FAST_MODEL, DEFAULT_STRONG_MODEL


class Passage(BaseModel):
    id: Optional[Union[int, str]] = None
    text: str
    meta: Dict[str, Any] = Field(default_factory=dict)


# A passage may be sent as a plain string or as a structured object.
PassageField = Union[str, Passage]


class RerankRequestBody(BaseModel):
    query: str = Field(..., description="The search query.")
    passages: List[PassageField] = Field(..., description="Candidate passages to rerank.")
    model: str = Field(DEFAULT_FAST_MODEL, description="Model name from /models.")
    top_k: Optional[int] = Field(None, ge=1, description="Return only the top K results.")
    max_length: Optional[int] = Field(None, ge=16, le=512, description="Max token length.")


class CascadeRequestBody(BaseModel):
    query: str = Field(..., description="The search query.")
    passages: List[PassageField] = Field(..., description="Candidate passages to rerank.")
    fast_model: str = Field(DEFAULT_FAST_MODEL, description="Cheap first-stage model.")
    strong_model: str = Field(DEFAULT_STRONG_MODEL, description="Stronger second-stage model.")
    prune_to: int = Field(50, ge=1, description="Survivors kept after stage 1.")
    top_k: Optional[int] = Field(None, ge=1, description="Return only the top K results.")
    max_length: Optional[int] = Field(None, ge=16, le=512, description="Max token length.")


class RankedPassage(BaseModel):
    id: Union[int, str]
    text: str
    score: float
    meta: Dict[str, Any] = Field(default_factory=dict)


class RerankResponse(BaseModel):
    results: List[RankedPassage]
    count: int
    took_ms: float


class ModelDescription(BaseModel):
    name: str
    size_mb: float
    description: str
    multilingual: bool


class HealthResponse(BaseModel):
    status: str
    loaded_models: List[str]
