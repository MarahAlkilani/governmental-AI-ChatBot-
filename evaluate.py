#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py - 42-style auto-grader for Kashaf.

  python evaluate.py [questions.json] [answers.json]

Runs every question through rag_engine, writes answers.json, then audits the file:
structure vs answers_example.json, every message answered, contexts is a list,
refusals carry no contexts. Uses `rich` when installed, plain text otherwise.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_engine import REFUSAL, RAGEngine, run_questions

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    RICH = True
except Exception:  # pragma: no cover
    RICH = False

BANNER = r"""
 _  __    _    ____  _   _    _    _____
| |/ /   / \  / ___|| | | |  / \  |  ___|
| ' /   / _ \ \___ \| |_| | / _ \ | |_
| . \  / ___ \ ___) |  _  |/ ___ \|  _|
|_|\_\/_/   \_\____/|_| |_/_/   \_\_|    searchlight // auto-grader
"""


def count_messages(data) -> int:
    chats = data["chats"] if isinstance(data, dict) else data
    return sum(len(c.get("messages", [])) for c in chats)


def audit(questions: dict, answers: dict, example_path: Path):
    """Return a list of (check, ok, detail)."""
    checks = []
    q_chats = questions["chats"] if isinstance(questions, dict) else questions
    a_chats = answers["chats"] if isinstance(answers, dict) else answers
    checks.append(("same number of chats", len(q_chats) == len(a_chats), f"{len(a_chats)}/{len(q_chats)}"))
    order_ok, missing, bad_ctx, refusal_ctx, errors = True, 0, 0, 0, 0
    for qc, ac in zip(q_chats, a_chats):
        if qc.get("chat_id") != ac.get("chat_id"):
            order_ok = False
        for qm, am in zip(qc.get("messages", []), ac.get("messages", [])):
            if qm.get("message_id") != am.get("message_id") or qm.get("question") != am.get("question"):
                order_ok = False
            if not str(am.get("answer", "")).strip():
                missing += 1
            if am.get("answer") == "error":
                errors += 1
            if not isinstance(am.get("contexts"), list):
                bad_ctx += 1
            if am.get("answer") == REFUSAL and am.get("contexts"):
                refusal_ctx += 1
    checks += [
        ("ids / questions untouched, order kept", order_ok, ""),
        ("every message has an answer", missing == 0, f"{missing} missing"),
        ("contexts is a list everywhere", bad_ctx == 0, f"{bad_ctx} bad"),
        ("refusals carry no contexts", refusal_ctx == 0, f"{refusal_ctx} bad"),
        ("no engine errors", errors == 0, f"{errors} errors"),
    ]
    if example_path.exists():
        try:
            ex = json.loads(example_path.read_text(encoding="utf-8-sig"))
            ex_keys = set(ex["chats"][0]["messages"][0])
            got = set(a_chats[0]["messages"][0])
            checks.append(("keys match answers_example.json", ex_keys <= got, ",".join(sorted(ex_keys - got)) or "ok"))
        except Exception as exc:  # noqa: BLE001
            checks.append(("keys match answers_example.json", False, str(exc)))
    return checks


def main() -> int:
    qpath = Path(sys.argv[1] if len(sys.argv) > 1 else "questions.json")
    apath = Path(sys.argv[2] if len(sys.argv) > 2 else qpath.resolve().parent / "answers.json")
    if not qpath.exists():
        print(f"questions file not found: {qpath}")
        return 2
    questions = json.loads(qpath.read_text(encoding="utf-8-sig"))
    total = count_messages(questions)
    rows = []

    if RICH:
        console = Console()
        console.print(f"[bold #F59E0B]{BANNER}[/]")
        console.print(f"[#94A3B8]> loading engine (llama3 + bge-m3)...[/]")
        engine = RAGEngine()
        progress = Progress(
            SpinnerColumn(style="#F59E0B"),
            TextColumn("[bold #F59E0B]KASHAF[/] [#94A3B8]{task.description}"),
            BarColumn(complete_style="#F59E0B", finished_style="#22C55E", pulse_style="#B45309"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        with progress:
            task = progress.add_task("grading".ljust(10), total=total)

            def cb(msg, res, dt):
                rows.append((msg.get("message_id", "?"), msg.get("question", ""), res["mode"], dt))
                progress.update(task, advance=1, description=f"{msg.get('message_id', '')}".ljust(10))

            t0 = time.time()
            data = run_questions(str(qpath), str(apath), engine=engine, on_message=cb)
        table = Table(title="results", border_style="#F59E0B", header_style="bold #F59E0B")
        for col in ("message", "mode", "sec", "question"):
            table.add_column(col, overflow="fold")
        for mid, q, mode, dt in rows:
            style = {"answered": "green", "refused": "yellow", "out_of_scope": "yellow", "clarify": "cyan"}.get(mode, "red")
            table.add_row(mid, f"[{style}]{mode}[/]", f"{dt:.1f}", q[:70])
        console.print(table)
        for name, ok, detail in audit(questions, data, qpath.resolve().parent / "answers_example.json"):
            console.print(f"  {'[green]PASS[/]' if ok else '[red]FAIL[/]'}  {name} [#94A3B8]{detail}[/]")
        console.print(Panel.fit(f"[bold green]Evaluation Complete ✅[/]  {total} answers -> {apath}  ({time.time() - t0:.1f}s)",
                                border_style="#F59E0B"))
    else:
        print(BANNER)
        engine = RAGEngine()
        done = {"n": 0}
        t0 = time.time()

        def cb(msg, res, dt):
            done["n"] += 1
            filled = int(30 * done["n"] / max(total, 1))
            print(f"\r[{'#' * filled}{'.' * (30 - filled)}] {done['n']}/{total} {msg.get('message_id', '')} {res['mode']} {dt:.1f}s", end="", flush=True)

        data = run_questions(str(qpath), str(apath), engine=engine, on_message=cb)
        print()
        for name, ok, detail in audit(questions, data, qpath.resolve().parent / "answers_example.json"):
            print(f"  {'PASS' if ok else 'FAIL'}  {name} {detail}")
        print(f"Evaluation Complete ✅  {total} answers -> {apath} ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
