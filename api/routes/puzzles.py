"""
api/routes/puzzles.py
Puzzle retrieval and guess-validation endpoints.

  GET  /puzzles               → list all stored puzzles (id + themes)
  GET  /puzzle/{id}           → full puzzle by id
  GET  /puzzle/random         → pick a random stored puzzle
  POST /puzzle/{id}/guess     → validate a 4-word guess
"""

import random
from fastapi import APIRouter, HTTPException

from api.schemas import (
    PuzzleSchema, PuzzleListItem,
    GuessRequest, GuessResponse, GroupSchema,
)
from api.store import PuzzleStore

router = APIRouter()
_store = PuzzleStore()


@router.get("/puzzles", response_model=list[PuzzleListItem], tags=["puzzles"])
def list_puzzles() -> list[PuzzleListItem]:
    """Return a lightweight list of all stored puzzles."""
    all_puzzles = _store.all()
    return [
        PuzzleListItem(
            id=p["id"],
            themes=[g["theme"] for g in sorted(p["groups"], key=lambda g: g["level"])],
            types= [g["theme_type"] for g in sorted(p["groups"], key=lambda g: g["level"])],
        )
        for p in all_puzzles
    ]


@router.get("/puzzle/random", response_model=PuzzleSchema, tags=["puzzles"])
def random_puzzle() -> PuzzleSchema:
    """Return a randomly selected stored puzzle."""
    all_puzzles = _store.all()
    if not all_puzzles:
        raise HTTPException(status_code=404, detail="No puzzles stored yet. Call POST /generate first.")
    chosen = random.choice(all_puzzles)
    return PuzzleSchema(**chosen)


@router.get("/puzzle/{puzzle_id}", response_model=PuzzleSchema, tags=["puzzles"])
def get_puzzle(puzzle_id: str) -> PuzzleSchema:
    """Return a specific puzzle by its id."""
    puzzle = _store.get(puzzle_id)
    if puzzle is None:
        raise HTTPException(status_code=404, detail=f"Puzzle '{puzzle_id}' not found.")
    return PuzzleSchema(**puzzle)


@router.post("/puzzle/{puzzle_id}/guess", response_model=GuessResponse, tags=["puzzles"])
def validate_guess(puzzle_id: str, req: GuessRequest) -> GuessResponse:
    """
    Validate a player's 4-word guess against a stored puzzle.

    Returns:
      correct   — all 4 words belong to the same group
      one_away  — exactly 3 of 4 words belong to one group (but not correct)
      group     — the solved GroupSchema, only when correct=True
      message   — human-readable feedback
    """
    puzzle = _store.get(puzzle_id)
    if puzzle is None:
        raise HTTPException(status_code=404, detail=f"Puzzle '{puzzle_id}' not found.")

    guessed = {w.upper().strip() for w in req.words}
    if len(guessed) != 4:
        raise HTTPException(status_code=422, detail="Guess must contain exactly 4 unique words.")

    best_overlap = 0
    matched_group: dict | None = None

    for group in puzzle["groups"]:
        group_words = {m.upper() for m in group["members"]}
        overlap     = len(guessed & group_words)

        if overlap > best_overlap:
            best_overlap  = overlap
            matched_group = group

    if best_overlap == 4:
        return GuessResponse(
            correct=True,
            one_away=False,
            group=GroupSchema(**matched_group),
            message=f"Correct! The {matched_group['color'].upper()} group was: {matched_group['theme']}",
        )

    if best_overlap == 3:
        return GuessResponse(
            correct=False,
            one_away=True,
            group=None,
            message="One away! You're very close — one word doesn't belong.",
        )

    return GuessResponse(
        correct=False,
        one_away=False,
        group=None,
        message="Incorrect. Try a different combination.",
    )