"""
api/routes/generate.py
POST /generate — runs the full generation + validation pipeline
and returns a ready-to-play puzzle.
"""

import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException

# Ensure ml/ is importable
ML_DIR = Path(__file__).parent.parent.parent / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from puzzle_assembler import generate_puzzle
from validator        import PuzzleValidator

from api.schemas import GenerateRequest, GenerateResponse, PuzzleSchema
from api.store   import PuzzleStore

router    = APIRouter()
_store    = PuzzleStore()
_validator = PuzzleValidator()


@router.post("/generate", response_model=GenerateResponse, tags=["generation"])
def generate(req: GenerateRequest = GenerateRequest()) -> GenerateResponse:
    """
    Generate a new NYT-style Connections puzzle.

    Runs the full ML pipeline:
      1. GroupGenerator builds a candidate pool
      2. PuzzleAssembler selects the best 4-group combination
      3. PuzzleValidator checks quality

    Returns the puzzle even if validation has warnings, but will
    raise 500 if generation fails entirely.
    """
    MAX_ATTEMPTS = 5

    for attempt in range(1, MAX_ATTEMPTS + 1):
        puzzle = generate_puzzle(seed=req.seed)
        if puzzle is None:
            continue

        data   = puzzle.to_dict()
        result = _validator.validate(data)

        if result.passed:
            _store.save(data)
            return GenerateResponse(
                puzzle=PuzzleSchema(**data),
                validation_passed=True,
                quality_score=result.quality_score,
                warnings=result.warnings,
            )

        # On last attempt, return best-effort even if warnings remain
        if attempt == MAX_ATTEMPTS:
            if not result.failures:
                _store.save(data)
                return GenerateResponse(
                    puzzle=PuzzleSchema(**data),
                    validation_passed=False,
                    quality_score=result.quality_score,
                    warnings=result.warnings + result.failures,
                )

    raise HTTPException(
        status_code=500,
        detail=f"Could not generate a valid puzzle after {MAX_ATTEMPTS} attempts."
    )