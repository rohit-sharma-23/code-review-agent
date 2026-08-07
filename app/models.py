"""Data models and schemas for the application."""

from pydantic import BaseModel
from typing import Optional, List


class CodeReviewRequest(BaseModel):
    repository_url: Optional[str] = None
    code_snippet: Optional[str] = None


class CodeReviewResponse(BaseModel):
    summary: str
    issues: List[dict] = []
