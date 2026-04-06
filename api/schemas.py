"""
api/schemas.py
Pydantic models for all API request and response shapes.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Puzzle models ─────────────────────────────────────────────────────────────

class GroupSchema(BaseModel):
    level:      int             = Field(..., ge=0, le=3)
    color:      str
    hex:        str
    theme:      str
    members:    list[str]       = Field(..., min_length=4, max_length=4)
    theme_type: str
    coherence:  float

class PuzzleSchema(BaseModel):
    id:             str
    groups:         list[GroupSchema] = Field(..., min_length=4, max_length=4)
    words_shuffled: list[str]         = Field(..., min_length=16, max_length=16)
    red_herrings:   list[str]
    meta:           Optional[dict]    = None

class PuzzleListItem(BaseModel):
    id:     str
    themes: list[str]           # one theme label per group, ordered by level
    types:  list[str]           # one theme_type per group


# ── Generation request/response ───────────────────────────────────────────────

class GenerateRequest(BaseModel):
    seed:      Optional[int] = Field(None, description="Random seed for reproducibility")
    pool_size: int           = Field(24, ge=8, le=60,
                                    description="Candidate pool size (larger = better quality, slower)")

class GenerateResponse(BaseModel):
    puzzle:           PuzzleSchema
    validation_passed: bool
    quality_score:    float
    warnings:         list[str] = []


# ── Guess validation ──────────────────────────────────────────────────────────

class GuessRequest(BaseModel):
    puzzle_id: str
    words:     list[str] = Field(..., min_length=4, max_length=4)

class GuessResponse(BaseModel):
    correct:    bool
    one_away:   bool          # 3 of 4 words belong to a correct group
    group:      Optional[GroupSchema] = None   # populated on correct guess
    message:    str


# ── Error ─────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str