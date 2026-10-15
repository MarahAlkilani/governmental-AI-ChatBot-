#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Official entry point:  python main.py questions.json  ->  answers.json (same folder)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_engine import run_questions


def main() -> int:
    questions = sys.argv[1] if len(sys.argv) > 1 else "questions.json"
    out = Path(questions).resolve().parent / "answers.json"
    run_questions(questions, str(out))
    print(f"answers written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
