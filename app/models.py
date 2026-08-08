"""Data models and schemas for the application."""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class CodeChunk(BaseModel):
    """Schema representing an extracted code chunk."""
    file_path: str
    content: str
    chunk_type: Literal["function", "async_function", "class"]
    embedding: Optional[List[float]] = None


class ReviewFinding(BaseModel):
    """Schema representing an individual code review comment finding."""
    file_path: str
    line: int
    comment_type: Literal["bug_risk", "missing_test", "style_deviation"]
    body: str


class CodeReviewRequest(BaseModel):
    repository_url: Optional[str] = None
    code_snippet: Optional[str] = None


class CodeReviewResponse(BaseModel):
    summary: str
    issues: List[dict] = []


