"""
run_pipeline.py
Orchestrates the full pipeline from project root:

  Phase 2 → ml/pattern_analyzer.py   (mine NYT patterns)
  Phase 3 → ml/embeddings.py         (build embedding cache)
  Phase 4 → ml/difficulty_model.py   (train difficulty classifier)
  Phase 5+6 → generate + validate a test puzzle

All ml/ modules are imported by injecting the ml/ directory onto sys.path
before any import, which resolves the "Unresolved reference" errors that
occur when running from the project root.

Usage:
  python run_pipeline.py                     # full pipeline
  python run_pipeline.py --skip-embeddings   # skip Phase 3 if cache exists
  python run_pipeline.py --skip-training     # skip Phases 2-4
  python run_pipeline.py --generate-only     # only Phase 5+6
  python run_pipeline.py --no-generate       # only Phases 2-4
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ── Resolve ml/ on the path before any ml module is imported ─────────────────
ML_DIR = Path(__file__).parent / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))


# ── Phase runner ──────────────────────────────────────────────────────────────

def run_phase(label: str, module_path: str) -> None:
    print(f"\n{'▶' * 3}  {label}")
    print("─" * 60)
    t0 = time.time()

    import importlib.util
    spec = importlib.util.spec_from_file_location("_phase", module_path)
    mod  = importlib.util.module_from_spec(spec)
    # Ensure the module itself can import its siblings
    sys.path.insert(0, str(Path(module_path).parent))
    spec.loader.exec_module(mod)
    mod.main()

    print(f"   Completed in {time.time() - t0:.1f}s")


# ── Phase 5+6: generate + validate ───────────────────────────────────────────

def run_generation_test() -> bool:
    print(f"\n{'▶' * 3}  Phase 5+6 — Generate & Validate")
    print("─" * 60)
    t0 = time.time()

    # These imports work because ML_DIR is on sys.path (inserted at top of file)
    from puzzle_assembler import generate_puzzle
    from validator import PuzzleValidator

    puzzle = generate_puzzle()
    if puzzle is None:
        print("  ✗ Puzzle generation failed.")
        return False

    data      = puzzle.to_dict()
    validator = PuzzleValidator()
    result    = validator.validate(data)

    print(f"\n── Validation Result ──")
    print(result)

    # Persist to puzzle store (append, no duplicates by id)
    store_path = Path("api/puzzle_store.json")
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store: list = []
    if store_path.exists():
        with open(store_path) as f:
            store = json.load(f)

    # Only save passing puzzles
    if result.passed:
        store.append(data)
        with open(store_path, "w") as f:
            json.dump(store, f, indent=2)
        print(f"\n  ✓ Puzzle saved → {store_path}  (total: {len(store)})")
    else:
        print(f"\n  ✗ Puzzle did not pass validation — not saved.")

    print(f"   Completed in {time.time() - t0:.1f}s")
    return result.passed


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Connections Generator Pipeline")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip Phase 3 if embeddings_cache.npz already exists")
    parser.add_argument("--skip-patterns",   action="store_true",
                        help="Skip Phase 2 if pattern_clusters.json already exists")
    parser.add_argument("--skip-training",   action="store_true",
                        help="Skip Phases 2-4 entirely (all model files must exist)")
    parser.add_argument("--generate-only",   action="store_true",
                        help="Only run Phase 5+6")
    parser.add_argument("--no-generate",     action="store_true",
                        help="Run Phases 2-4 only, skip generation test")
    args = parser.parse_args()

    data_path = Path("data/connections.json")
    if not data_path.exists():
        print(f"✗  {data_path} not found. Place connections.json in data/ and retry.")
        sys.exit(1)

    print("╔" + "═" * 58 + "╗")
    print("║   Connections Generator — Full Pipeline                 ║")
    print("╚" + "═" * 58 + "╝")

    if not args.generate_only and not args.skip_training:
        pat_exists = Path("ml/models/pattern_clusters.json").exists()
        emb_exists = Path("ml/models/embeddings_cache.npz").exists()

        if args.skip_patterns and pat_exists:
            print("\n⏭  Skipping Phase 2 (pattern_clusters.json exists)")
        else:
            run_phase("Phase 2 — Pattern Analyzer", "ml/pattern_analyzer.py")

        if args.skip_embeddings and emb_exists:
            print("\n⏭  Skipping Phase 3 (embeddings_cache.npz exists)")
        else:
            run_phase("Phase 3 — Embeddings", "ml/embeddings.py")

        run_phase("Phase 4 — Difficulty Classifier", "ml/difficulty_model.py")

    passed = True
    if not args.no_generate:
        passed = run_generation_test()

    print("\n╔" + "═" * 58 + "╗")
    if passed:
        print("║   ✓  Pipeline complete. Ready for Phase 7 (API)        ║")
    else:
        print("║   ⚠  Puzzle failed validation. Re-run --generate-only  ║")
    print("╚" + "═" * 58 + "╝\n")


if __name__ == "__main__":
    main()