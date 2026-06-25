"""
Travel Assistant — CLI entry point.

Commands:
  /reset  — reset conversation and trip context
  /state  — show current structured trip context
  /eval   — run evaluator scenarios
  /save   — save current conversation transcript
  /exit   — quit
"""

from __future__ import annotations

import itertools
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from context import ContextManager, IntentExtractor, ResponseGenerator
from models import ConversationTurn


class _Thinking:
    """ASCII spinner displayed while the LLM processes a request."""

    def __init__(self, label: str = "Thinking") -> None:
        self._label = label
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self) -> None:
        frames = itertools.cycle("|/-\\")
        while not self._stop.is_set():
            sys.stdout.write(f"\r{self._label} {next(frames)} ")
            sys.stdout.flush()
            time.sleep(0.1)
        # Clear the spinner line before the response prints
        sys.stdout.write(f"\r{' ' * (len(self._label) + 3)}\r")
        sys.stdout.flush()

    def __enter__(self) -> "_Thinking":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()


_COMMANDS = {
    "/reset": "Reset conversation and trip context",
    "/state": "Show current trip context",
    "/eval":  "Run evaluation scenarios",
    "/save":  "Save conversation transcript",
    "/exit":  "Quit",
}


def _save_transcript(history: list[ConversationTurn]) -> str:
    Path("transcripts").mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"transcripts/transcript_{ts}.txt"
    with open(path, "w", encoding="utf-8") as f:
        for turn in history:
            label = "You" if turn.role == "user" else "Assistant"
            f.write(f"{label}: {turn.content}\n\n")
    return path


def main() -> None:
    ctx = ContextManager()
    extractor = IntentExtractor()
    generator = ResponseGenerator()

    print("Travel Assistant. Type /help for commands.")
    print()

    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not raw:
            continue

        # ── Slash commands ────────────────────────────────────────────────────
        if raw.startswith("/"):
            cmd = raw.lower().split()[0]

            if cmd == "/exit":
                print("Bye!")
                break

            elif cmd == "/reset":
                ctx.reset()
                print("[Context and history reset.]\n")

            elif cmd == "/state":
                print("\n[Current trip context]")
                print(ctx.trip_context.model_dump_json(indent=2))
                print()

            elif cmd == "/eval":
                # Import here to avoid loading evaluator/scenarios at startup
                from evaluator import run_evaluation
                run_evaluation()

            elif cmd == "/save":
                if not ctx.history:
                    print("[Nothing to save yet.]\n")
                else:
                    try:
                        path = _save_transcript(ctx.history)
                        print(f"[Transcript saved to {path}]\n")
                    except OSError as e:
                        print(f"[Could not save transcript: {e}. Check permissions on transcripts/.]\n")

            elif cmd == "/help":
                print("\n[Available commands]")
                for name, desc in _COMMANDS.items():
                    print(f"  {name:8s}  {desc}")
                print()

            else:
                cmds = ", ".join(_COMMANDS)
                print(f"[Unknown command. Available: {cmds}]\n")

            continue

        # ── Conversation turn ─────────────────────────────────────────────────
        try:
            with _Thinking():
                # 1. Extract intent + context updates
                extraction = extractor.extract(
                    user_message=raw,
                    history=ctx.history,
                    trip_context=ctx.trip_context,
                )

                # 2. Merge context updates, add user turn to history
                ctx.update(extraction, raw)

                # 3. Route tools (uses shared cache from ctx)
                weather, attractions = ctx.router.route(extraction, ctx.trip_context)

                # 4. Generate response
                response = generator.generate(
                    user_message=raw,
                    intent_extraction=extraction,
                    trip_context=ctx.trip_context,
                    history=ctx.history,
                    weather=weather,
                    attractions=attractions,
                )

            # 5. Print, record, update last_topic
            print(f"\nAssistant: {response.response_text}\n")
            ctx.add_assistant_turn(response.response_text)
            ctx.set_last_topic(extraction.intent)

        except Exception as e:
            print("\nAssistant: Something went wrong on my end. Could you rephrase?\n")
            sys.stderr.write(f"[error] {type(e).__name__}: {e}\n")
            if os.getenv("DEBUG"):
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
