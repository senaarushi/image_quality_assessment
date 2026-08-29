"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class IssueSchema(BaseModel):
    type: str
    severity: str
    confidence: float


class AnalysisResponse(BaseModel):
    id: int
    filename: str
    image_url: str
    quality_score: int
    quality_label: str
    issues: list[IssueSchema]
    image_stats: dict
    model_breakdown: dict
    created_at: datetime
    gradcam_available: bool = False

    class Config:
        from_attributes = True


class HistoryItem(BaseModel):
    id: int
    filename: str
    quality_score: int
    quality_label: str
    created_at: datetime

    class Config:
        from_attributes = True


class HistoryResponse(BaseModel):
    total: int
    results: list[HistoryItem]


class ErrorResponse(BaseModel):
    detail: str