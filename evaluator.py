"""
Evaluator: runs fixed scenario conversations, applies deterministic checks,
calls the LLM judge, and prints structured EvaluationResult objects.

Called via the /eval CLI command.
"""

from __future__ import annotations

import json
import os
import re
import traceback
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from context import ContextManager, IntentExtractor, ResponseGenerator, get_backend
from models import (
    AssistantResponse,
    DeterministicChecks,
    EvaluationResult,
    EvaluationTest,
    IntentExtraction,
    JudgeOutput,
    TripContext,
)
from prompts import EVALUATOR_SYSTEM_ADDENDUM, SYSTEM_PROMPT, format_evaluator_user_prompt
from tools import ToolRouter

_ALLOWED_INTENTS = frozenset({
    "destination_recommendation", "itinerary_planning", "packing_advice",
    "local_attractions", "weather_advice", "trip_refinement", "context_update",
    "clarification_needed", "general_travel_qa", "out_of_scope",
})

# Patterns used in the hallucination deterministic check
# Catches HH:MM (with optional 24h range or AM/PM) and compact forms like "9am"/"6pm"
_TIME_PATTERN = re.compile(
    r"\b\d{1,2}:\d{2}(?:\s*[-–]\s*\d{1,2}:\d{2})?\s*(?:AM|PM)?"
    r"|\b\d{1,2}\s*(?:am|pm)\b",
    re.IGNORECASE,
)
_PRICE_PATTERN = re.compile(r"\$\d+|\€\d+|£\d+|\d+\s*(euro|euros|USD|EUR|GBP)", re.IGNORECASE)

# Patterns used in incremental_info_gathering deterministic checks
_ITINERARY_MARKER_PATTERN = re.compile(r"\*\*Day \d+|Day 1[:\.]|day-by-day", re.IGNORECASE)
_DAY_HEADER_PATTERN = re.compile(r"\bDay\s+\d+\b", re.IGNORECASE)
_CONFIRMATION_PROMPT_PATTERN = re.compile(
    r"want me to (draft|put together|create|build)|shall I (draft|put together|create)|should I (draft|put together)",
    re.IGNORECASE,
)


def _run_scenario(test: EvaluationTest) -> tuple[str, IntentExtraction | None, AssistantResponse | None, TripContext]:
    """Run all user turns in a scenario. Returns (transcript, last_extraction, last_response, final_trip_context)."""
    ctx = ContextManager()
    extractor = IntentExtractor()
    generator = ResponseGenerator()

    transcript_lines: list[str] = []
    last_extraction: IntentExtraction | None = None
    last_response: AssistantResponse | None = None

    for turn in test.conversation:
        role = turn.get("role", "user")
        content = turn.get("content", "")

        if role != "user":
            continue  # scenarios only have user turns as input

        transcript_lines.append(f"User: {content}")

        extraction = extractor.extract(
            user_message=content,
            history=ctx.history,
            trip_context=ctx.trip_context,
        )
        last_extraction = extraction
        ctx.update(extraction, content)

        weather, attractions = ctx.router.route(extraction, ctx.trip_context)

        response = generator.generate(
            user_message=content,
            intent_extraction=extraction,
            trip_context=ctx.trip_context,
            history=ctx.history,
            weather=weather,
            attractions=attractions,
        )
        last_response = response

        transcript_lines.append(f"Assistant: {response.response_text}")
        ctx.add_assistant_turn(response.response_text)
        ctx.set_last_topic(extraction.intent)

    return "\n\n".join(transcript_lines), last_extraction, last_response, ctx.trip_context


def _deterministic_checks(
    test: EvaluationTest,
    extraction: IntentExtraction | None,
    response: AssistantResponse | None,
    trip_context: TripContext | None = None,
) -> DeterministicChecks:
    valid_json = extraction is not None and response is not None
    intent_ok = extraction is not None and extraction.intent in _ALLOWED_INTENTS
    length_ok = response is not None and len(response.response_text) < 2500

    tool_used_when_required: bool | None = None
    if test.requires_tool != "none" and response is not None:
        tool_used_when_required = response.used_external_data

    no_live_facts = True
    if test.must_not_claim_live_facts and response is not None:
        text = response.response_text
        has_time = bool(_TIME_PATTERN.search(text))
        has_price = bool(_PRICE_PATTERN.search(text))
        no_live_facts = not (has_time or has_price)

    # New check: when should_clarify=True, the response must NOT contain an itinerary
    fabricated_plan_when_should_clarify: bool | None = None
    if test.should_clarify and response is not None:
        has_plan = bool(_ITINERARY_MARKER_PATTERN.search(response.response_text))
        fabricated_plan_when_should_clarify = not has_plan  # True=pass, False=fail

    # New check: when duration_days is set, Day N count must match
    day_count_matches_duration: bool | None = None
    if trip_context is not None and trip_context.duration_days is not None and response is not None:
        day_headers = _DAY_HEADER_PATTERN.findall(response.response_text)
        day_count_matches_duration = len(day_headers) == trip_context.duration_days

    confirmation_prompt_emitted: bool | None = None
    if test.test_id == "incremental_02" and response is not None:
        confirmation_prompt_emitted = bool(
            _CONFIRMATION_PROMPT_PATTERN.search(response.response_text)
        )

    return DeterministicChecks(
        valid_json_output=valid_json,
        intent_in_allowed_set=intent_ok,
        response_length_ok=length_ok,
        tool_used_when_required=tool_used_when_required,
        no_claimed_live_facts_without_tool=no_live_facts,
        fabricated_plan_when_should_clarify=fabricated_plan_when_should_clarify,
        day_count_matches_duration=day_count_matches_duration,
        confirmation_prompt_emitted=confirmation_prompt_emitted,
    )


def _llm_judge(
    test: EvaluationTest,
    transcript: str,
    checks: DeterministicChecks,
) -> EvaluationResult:
    backend = get_backend()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + EVALUATOR_SYSTEM_ADDENDUM},
        {"role": "user", "content": format_evaluator_user_prompt(
            transcript=transcript,
            deterministic_checks=checks.model_dump(),
            expected_intent=test.expected_intent,
            requires_tool=test.requires_tool,
            should_clarify=test.should_clarify,
            must_not_claim_live_facts=test.must_not_claim_live_facts,
        )},
    ]

    for attempt in range(2):
        try:
            raw = backend.chat(messages, JudgeOutput)
            judge = JudgeOutput.model_validate_json(raw)

            # Hard overrides — enforced in Python regardless of LLM score.
            # Prevents a generous judge from masking deterministic violations.
            if test.must_not_claim_live_facts and not checks.no_claimed_live_facts_without_tool:
                judge = judge.model_copy(update={"hallucination_resistance": 1})
            if checks.tool_used_when_required is False:
                judge = judge.model_copy(update={"tool_usage_appropriateness": 1})

            return EvaluationResult(
                test_id=test.test_id,
                intent_accuracy=judge.intent_accuracy,
                context_handling=judge.context_handling,
                response_quality=judge.response_quality,
                hallucination_resistance=judge.hallucination_resistance,
                tool_usage_appropriateness=judge.tool_usage_appropriateness,
                overall_score=3.0,  # overwritten by model_validator
                deterministic_checks=checks,
                improvement_suggestion=judge.improvement_suggestion,
                judge_reasoning=judge.judge_reasoning,
                judge_succeeded=True,
            )
        except (ValidationError, Exception):
            if attempt == 0:
                continue
            if os.getenv("DEBUG"):
                traceback.print_exc()

    # Fallback result on judge failure
    return EvaluationResult(
        test_id=test.test_id,
        intent_accuracy=3,
        context_handling=3,
        response_quality=3,
        hallucination_resistance=3,
        tool_usage_appropriateness=3,
        overall_score=3.0,
        deterministic_checks=checks,
        improvement_suggestion="Judge failed to produce structured output — review this transcript manually.",
        judge_reasoning="LLM judge call failed after two attempts.",
        judge_succeeded=False,
    )


def run_single(test: EvaluationTest) -> EvaluationResult:
    transcript, extraction, response, trip_context = _run_scenario(test)
    checks = _deterministic_checks(test, extraction, response, trip_context)
    return _llm_judge(test, transcript, checks)


def run_evaluation() -> list[EvaluationResult]:
    from scenarios import EVALUATION_SCENARIOS

    print("\n=== Travel Assistant Evaluation ===\n")
    results: list[EvaluationResult] = []

    for test in EVALUATION_SCENARIOS:
        print(f"  [{test.test_id}] {test.category}...", end=" ", flush=True)
        try:
            result = run_single(test)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        results.append(result)
        checks = result.deterministic_checks
        flags = [
            f"json={'OK' if checks.valid_json_output else 'FAIL'}",
            f"intent={'OK' if checks.intent_in_allowed_set else 'FAIL'}",
            f"length={'OK' if checks.response_length_ok else 'FAIL'}",
            f"no_halluc={'OK' if checks.no_claimed_live_facts_without_tool else 'FAIL'}",
        ]
        if checks.tool_used_when_required is not None:
            flags.append(f"tool={'OK' if checks.tool_used_when_required else 'FAIL'}")
        if checks.fabricated_plan_when_should_clarify is not None:
            flags.append(f"no_fab_plan={'OK' if checks.fabricated_plan_when_should_clarify else 'FAIL'}")
        if checks.day_count_matches_duration is not None:
            flags.append(f"day_count={'OK' if checks.day_count_matches_duration else 'FAIL'}")
        if checks.confirmation_prompt_emitted is not None:
            flags.append(f"confirm_prompt={'OK' if checks.confirmation_prompt_emitted else 'FAIL'}")

        print(f"score={result.overall_score:.1f}/5 | {' '.join(flags)}")
        print(f"    → {result.improvement_suggestion}")

    if results:
        avg = sum(r.overall_score for r in results) / len(results)
        print(f"\n=== Average score: {avg:.2f}/5.0 across {len(results)} scenarios ===\n")

    # Persist output
    Path("evaluator_output").mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"evaluator_output/eval_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)
    print(f"[Full results saved to {out_path}]\n")

    return results
