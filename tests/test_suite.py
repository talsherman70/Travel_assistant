"""
tests/test_suite.py

Layer 1 (T1–T16): Deterministic integration tests.
  All LLM calls are intercepted by _FakeBackend / _SequenceFakeBackend / _FakeFlakyBackend.
  No network I/O. Covers context mutation, tool routing, and failure-mode handling.

Layer 3 (T21–T25): Evaluator / LLM-judge tests.
  T21–T23 use _deterministic_checks() only (zero LLM cost).
  T24 uses _llm_judge() and is skipped when no API key is configured.
  T25 is fully deterministic (mocked backend).

Layer 4: Schema, override, and pattern unit tests (fully deterministic).
"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from context import ContextManager, IntentExtractor, ResponseGenerator
from evaluator import _deterministic_checks, _llm_judge
from models import (
    AssistantResponse,
    AttractionsResult,
    EvaluationTest,
    IntentExtraction,
    TripContext,
    WeatherDay,
    WeatherResult,
)


# ── JSON fixture builders ─────────────────────────────────────────────────────

def _intent_json(
    intent: str = "destination_recommendation",
    confidence: float = 0.9,
    destination: str | None = None,
    start_date: str | None = None,
    duration_days: int | None = None,
    travelers: int | None = None,
    interests: list[str] | None = None,
    needs_weather: bool = False,
    needs_attractions: bool = False,
) -> str:
    ctx = {
        "destination": destination,
        "start_date": start_date,
        "end_date": None,
        "duration_days": duration_days,
        "travelers": travelers,
        "budget_level": None,
        "interests": interests or [],
        "constraints": [],
        "pace": None,
        "last_topic": None,
    }
    return json.dumps({
        "intent": intent,
        "confidence": confidence,
        "context_updates": ctx,
        "needs_weather": needs_weather,
        "needs_attractions": needs_attractions,
        "clarification_questions": [],
    })


def _clarification_intent_json(questions: list[str] | None = None) -> str:
    ctx = {
        "destination": None, "start_date": None,
        "end_date": None, "duration_days": None, "travelers": None,
        "budget_level": None, "interests": [], "constraints": [],
        "pace": None, "last_topic": None,
    }
    return json.dumps({
        "intent": "clarification_needed",
        "confidence": 0.4,
        "context_updates": ctx,
        "needs_weather": False,
        "needs_attractions": False,
        "clarification_questions": questions or ["Could you be more specific about your destination?"],
    })


def _response_json(
    text: str = "Here are some recommendations for your trip.",
    used_external_data: bool = False,
) -> str:
    return json.dumps({
        "response_text": text,
        "internal_summary_update": "Provided travel advice.",
        "used_external_data": used_external_data,
    })


# ── Fake backends ─────────────────────────────────────────────────────────────

class _FakeBackend:
    """Intercepts LLM calls. Returns intent_json when schema is IntentExtraction,
    response_json for all other schemas (AssistantResponse, JudgeOutput)."""

    def __init__(self, intent_json: str, response_json: str) -> None:
        self._intent_json = intent_json
        self._response_json = response_json
        self.call_count = 0

    def chat(self, messages: list[dict], schema_model: type) -> str:
        self.call_count += 1
        if schema_model is IntentExtraction:
            return self._intent_json
        return self._response_json


class _SequenceFakeBackend:
    """Returns responses in insertion order regardless of schema type.
    After exhausting the list, repeats the last entry."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._idx = 0
        self.call_count = 0

    def chat(self, messages: list[dict], schema_model: type) -> str:
        self.call_count += 1
        val = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return val


class _FakeFlakyBackend:
    """First IntentExtraction call returns invalid JSON; subsequent calls succeed."""

    def __init__(self, intent_json: str, response_json: str) -> None:
        self._intent_json = intent_json
        self._response_json = response_json
        self._intent_calls = 0
        self.call_count = 0

    def chat(self, messages: list[dict], schema_model: type) -> str:
        self.call_count += 1
        if schema_model is IntentExtraction:
            self._intent_calls += 1
            if self._intent_calls == 1:
                return "{{NOT VALID JSON"
            return self._intent_json
        return self._response_json


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_components(
    fake: _FakeBackend | _SequenceFakeBackend | _FakeFlakyBackend,
) -> tuple[ContextManager, IntentExtractor, ResponseGenerator]:
    """Return (ctx, extractor, generator) all wired to the fake backend."""
    ctx = ContextManager()
    with patch("context.get_backend", return_value=fake):
        extractor = IntentExtractor()
        generator = ResponseGenerator()
    return ctx, extractor, generator


def _run_turn(
    user_msg: str,
    ctx: ContextManager,
    extractor: IntentExtractor,
    generator: ResponseGenerator,
) -> tuple[IntentExtraction, AssistantResponse, WeatherResult | None, AttractionsResult | None]:
    """One conversation turn — mirrors the loop in evaluator._run_scenario."""
    extraction = extractor.extract(user_msg, ctx.history, ctx.trip_context)
    ctx.update(extraction, user_msg)
    weather, attractions = ctx.router.route(extraction, ctx.trip_context)
    response = generator.generate(
        user_message=user_msg,
        intent_extraction=extraction,
        trip_context=ctx.trip_context,
        history=ctx.history,
        weather=weather,
        attractions=attractions,
    )
    ctx.add_assistant_turn(response.response_text)
    ctx.set_last_topic(extraction.intent)
    return extraction, response, weather, attractions


def _requires_api() -> bool:
    """True when DEEPSEEK_API_KEY is not configured (used to skip real-LLM judge tests)."""
    return not bool(os.getenv("DEEPSEEK_API_KEY", "").strip())


def _make_eval_test(**kwargs) -> EvaluationTest:
    defaults: dict = dict(
        test_id="stub",
        category="hallucination_resistance",
        conversation=[{"role": "user", "content": "stub"}],
        expected_intent=None,
        requires_tool="none",
        should_clarify=False,
        must_not_claim_live_facts=False,
    )
    defaults.update(kwargs)
    return EvaluationTest(**defaults)


# ── Static transcripts for Layer 3 evaluator tests ───────────────────────────

_VERBOSE_TRANSCRIPT = (
    "User: Should I bring an umbrella to London?\n\n"
    "Assistant: That's such a great question! I'm so glad you asked. London weather can be "
    "quite unpredictable, and as a seasoned travel advisor I always recommend being prepared. "
    "Let me walk you through the comprehensive weather patterns of the United Kingdom, starting "
    "with its geographical position in the North Atlantic and how that affects precipitation "
    "patterns throughout the year. The Gulf Stream plays a crucial role in moderating "
    "temperatures. In general, yes, you should bring an umbrella. Also consider a light jacket, "
    "waterproof shoes, and perhaps a travel poncho. Let me know if you need a full packing "
    "list for London!"
)

_HALLUCINATION_TRANSCRIPT = (
    "User: What are some attractions in Barcelona?\n\n"
    "Assistant: The Sagrada Família opens at 9:00 AM and closes at 6 PM. "
    "Entry costs €26 per adult. La Boqueria market is free to enter but "
    "expect to spend around $15 on tastings."
)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1: Deterministic integration tests (T1–T16)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSimpleContextTurnSetsDestination(unittest.TestCase):
    """T1: A single user message naming a destination sets TripContext.destination."""

    def test_destination_stored_in_context(self) -> None:
        fake = _FakeBackend(
            intent_json=_intent_json("destination_recommendation", destination="Rome"),
            response_json=_response_json("Rome is lovely in spring."),
        )
        ctx, extractor, generator = _make_components(fake)
        with patch("context.get_backend", return_value=fake):
            _run_turn("I want to go to Rome", ctx, extractor, generator)
        self.assertEqual(ctx.trip_context.destination, "Rome")


class TestFollowUpPackingUsesContext(unittest.TestCase):
    """T2: After destination is set, a packing question preserves that destination."""

    def test_destination_persists_across_turns(self) -> None:
        fake = _SequenceFakeBackend([
            _intent_json("destination_recommendation", destination="Barcelona"),
            _response_json("Barcelona is great."),
            _intent_json("packing_advice"),          # no new destination this turn
            _response_json("Pack light clothes."),
        ])
        ctx, extractor, generator = _make_components(fake)
        with patch("context.get_backend", return_value=fake):
            _run_turn("I'm going to Barcelona", ctx, extractor, generator)
            _run_turn("What should I pack?", ctx, extractor, generator)
        self.assertEqual(ctx.trip_context.destination, "Barcelona")


class TestWeatherToolFailureDegradesgracefully(unittest.TestCase):
    """T3: When the weather tool returns an error, the response is generated without crashing."""

    def test_graceful_degradation_on_weather_error(self) -> None:
        error_weather = WeatherResult(
            destination="Rome",
            retrieved_at="2026-06-24T10:00",
            error="Weather API timed out",
        )
        fake = _FakeBackend(
            intent_json=_intent_json("weather_advice", destination="Rome", needs_weather=True),
            response_json=_response_json("I couldn't get live weather data for Rome.", used_external_data=False),
        )
        ctx, extractor, generator = _make_components(fake)
        with patch("context.get_backend", return_value=fake), \
             patch("tools.get_weather", return_value=error_weather):
            _, response, weather, _ = _run_turn(
                "What's the weather in Rome next week?", ctx, extractor, generator
            )
        self.assertIsNotNone(response)
        self.assertGreater(len(response.response_text), 0)
        self.assertIsNotNone(weather)
        self.assertIsNotNone(weather.error)


class TestAttractionsToolFailureDegradesgracefully(unittest.TestCase):
    """T4: When the attractions tool returns an error, the response is generated without crashing."""

    def test_graceful_degradation_on_attractions_error(self) -> None:
        error_attractions = AttractionsResult(
            destination="Barcelona",
            retrieved_at="2026-06-24T10:00",
            error="Overpass API timed out",
        )
        fake = _FakeBackend(
            intent_json=_intent_json("local_attractions", destination="Barcelona", needs_attractions=True),
            response_json=_response_json("I couldn't fetch live attractions right now.", used_external_data=False),
        )
        ctx, extractor, generator = _make_components(fake)
        with patch("context.get_backend", return_value=fake), \
             patch("tools.get_attractions", return_value=error_attractions):
            _, response, _, attractions = _run_turn(
                "What should I do in Barcelona?", ctx, extractor, generator
            )
        self.assertIsNotNone(response)
        self.assertGreater(len(response.response_text), 0)
        self.assertIsNotNone(attractions)
        self.assertIsNotNone(attractions.error)


class TestToolTurnPassesResultsToGenerator(unittest.TestCase):
    """T5: When needs_weather=True and a destination is set, get_weather is called exactly once."""

    def test_weather_tool_called_when_flagged(self) -> None:
        mock_weather = WeatherResult(
            destination="Tokyo",
            retrieved_at="2026-06-24T10:00",
            forecast=[WeatherDay(date="2026-06-25", temp_high_c=28.0, temp_low_c=20.0, condition="partly cloudy")],
        )
        fake = _FakeBackend(
            intent_json=_intent_json("weather_advice", destination="Tokyo", needs_weather=True),
            response_json=_response_json("Tokyo: highs around 28°C, partly cloudy.", used_external_data=True),
        )
        ctx, extractor, generator = _make_components(fake)
        with patch("context.get_backend", return_value=fake), \
             patch("tools.get_weather", return_value=mock_weather) as mock_get_weather:
            _, response, _, _ = _run_turn(
                "What's the weather in Tokyo next week?", ctx, extractor, generator
            )
        mock_get_weather.assert_called_once()
        self.assertTrue(response.used_external_data)


class TestNonToolTurnDoesNotCallTools(unittest.TestCase):
    """T6: When needs_weather=False and needs_attractions=False, no tool functions are invoked."""

    def test_tools_not_called_for_general_query(self) -> None:
        fake = _FakeBackend(
            intent_json=_intent_json("packing_advice", destination="Paris"),
            response_json=_response_json("Pack layers for Paris in autumn."),
        )
        ctx, extractor, generator = _make_components(fake)
        with patch("context.get_backend", return_value=fake), \
             patch("tools.get_weather") as mock_weather, \
             patch("tools.get_attractions") as mock_attractions:
            _run_turn("What should I pack for Paris?", ctx, extractor, generator)
        mock_weather.assert_not_called()
        mock_attractions.assert_not_called()


class TestAmbiguousRequestTriggersClarification(unittest.TestCase):
    """T7: A vague message produces a clarification_needed intent with at least one question."""

    def test_clarification_intent_on_vague_input(self) -> None:
        questions = ["Which destination are you considering?", "How long is your trip?"]
        fake = _FakeBackend(
            intent_json=_clarification_intent_json(questions),
            response_json=_response_json("Could you tell me more? " + " ".join(questions)),
        )
        ctx, extractor, generator = _make_components(fake)
        with patch("context.get_backend", return_value=fake):
            extraction, _, _, _ = _run_turn("I want to travel somewhere nice.", ctx, extractor, generator)
        self.assertEqual(extraction.intent, "clarification_needed")
        self.assertGreater(len(extraction.clarification_questions), 0)


class TestContextChangeInvalidatesToolCache(unittest.TestCase):
    """T8: Changing destination clears the cache so the second destination re-fetches weather."""

    def test_cache_cleared_on_destination_change(self) -> None:
        rome_weather = WeatherResult(destination="Rome", retrieved_at="2026-06-24T10:00")
        paris_weather = WeatherResult(destination="Paris", retrieved_at="2026-06-24T10:01")
        resp = _response_json("Weather looks good.", used_external_data=True)

        fake = _SequenceFakeBackend([
            _intent_json("weather_advice", destination="Rome", needs_weather=True), resp,
            _intent_json("weather_advice", destination="Paris", needs_weather=True), resp,
        ])
        ctx, extractor, generator = _make_components(fake)

        call_log: list[str] = []

        def _side_effect(destination: str, start_date: str | None = None) -> WeatherResult:
            call_log.append(destination)
            return rome_weather if destination == "Rome" else paris_weather

        with patch("context.get_backend", return_value=fake), \
             patch("tools.get_weather", side_effect=_side_effect):
            _run_turn("What's the weather in Rome next week?", ctx, extractor, generator)
            _run_turn("Actually, what about Paris next week?", ctx, extractor, generator)

        self.assertEqual(len(call_log), 2)
        self.assertIn("Rome", call_log)
        self.assertIn("Paris", call_log)


class TestContextChangePreservesValidFields(unittest.TestCase):
    """T9: A partial context update (new travelers) does not overwrite existing destination/duration."""

    def test_existing_fields_survive_partial_update(self) -> None:
        fake = _SequenceFakeBackend([
            _intent_json("destination_recommendation", destination="Stockholm", duration_days=5),
            _response_json("Stockholm for 5 days sounds great."),
            _intent_json("context_update", travelers=2),   # no destination sent this turn
            _response_json("Got it — 2 travelers to Stockholm."),
        ])
        ctx, extractor, generator = _make_components(fake)
        with patch("context.get_backend", return_value=fake):
            _run_turn("I'm going to Stockholm for 5 days", ctx, extractor, generator)
            _run_turn("We'll be 2 travelers", ctx, extractor, generator)
        self.assertEqual(ctx.trip_context.destination, "Stockholm")
        self.assertEqual(ctx.trip_context.duration_days, 5)
        self.assertEqual(ctx.trip_context.travelers, 2)


class TestHallucinationResistanceForLiveFacts(unittest.TestCase):
    """T10: Deterministic check flags specific time/price claims made without tool data."""

    def _make_test(self, test_id: str) -> EvaluationTest:
        return _make_eval_test(
            test_id=test_id,
            category="hallucination_resistance",
            must_not_claim_live_facts=True,
        )

    def _extraction(self) -> IntentExtraction:
        return IntentExtraction(
            intent="local_attractions",
            confidence=0.9,
            context_updates=TripContext(),
            needs_weather=False,
            needs_attractions=False,
            clarification_questions=[],
        )

    def test_time_pattern_fails_check(self) -> None:
        response = AssistantResponse(
            response_text="The Colosseum opens at 9:00 AM and closes at 7:00 PM.",
            used_external_data=False,
        )
        checks = _deterministic_checks(self._make_test("t10a"), self._extraction(), response)
        self.assertFalse(checks.no_claimed_live_facts_without_tool)

    def test_price_pattern_fails_check(self) -> None:
        response = AssistantResponse(
            response_text="The Louvre costs €17 per adult.",
            used_external_data=False,
        )
        checks = _deterministic_checks(self._make_test("t10b"), self._extraction(), response)
        self.assertFalse(checks.no_claimed_live_facts_without_tool)

    def test_clean_response_passes_check(self) -> None:
        response = AssistantResponse(
            response_text="Rome has the Colosseum, Pantheon, and Vatican. Check official sites for current hours.",
            used_external_data=False,
        )
        checks = _deterministic_checks(self._make_test("t10c"), self._extraction(), response)
        self.assertTrue(checks.no_claimed_live_facts_without_tool)


class TestSingleWordFollowupUsesContext(unittest.TestCase):
    """T12: A one-word follow-up ('weather?') retains the previously set destination."""

    def test_destination_present_for_followup(self) -> None:
        mock_weather = WeatherResult(destination="Tokyo", retrieved_at="2026-06-24T10:00")
        fake = _SequenceFakeBackend([
            _intent_json("destination_recommendation", destination="Tokyo"),
            _response_json("Tokyo is wonderful."),
            _intent_json("weather_advice", needs_weather=True),   # no destination in this turn
            _response_json("Tokyo: mild and partly cloudy.", used_external_data=True),
        ])
        ctx, extractor, generator = _make_components(fake)
        with patch("context.get_backend", return_value=fake), \
             patch("tools.get_weather", return_value=mock_weather):
            _run_turn("I'm going to Tokyo", ctx, extractor, generator)
            _run_turn("weather?", ctx, extractor, generator)
        self.assertEqual(ctx.trip_context.destination, "Tokyo")


class TestSlidingHistoryWindowCapsAtWindowSize(unittest.TestCase):
    """T13: History never exceeds _WINDOW_SIZE entries regardless of turn count."""

    def test_history_length_bounded(self) -> None:
        from context import _WINDOW_SIZE

        # Each _run_turn adds 2 entries (user + assistant); add enough to overflow the window
        num_turns = _WINDOW_SIZE // 2 + 4

        responses: list[str] = []
        for i in range(num_turns):
            responses.append(_intent_json("general_travel_qa"))
            responses.append(_response_json(f"Response {i}."))

        fake = _SequenceFakeBackend(responses)
        ctx, extractor, generator = _make_components(fake)
        with patch("context.get_backend", return_value=fake):
            for i in range(num_turns):
                _run_turn(f"Question {i}", ctx, extractor, generator)

        self.assertLessEqual(len(ctx.history), _WINDOW_SIZE)


class TestValidationRetryThenFallback(unittest.TestCase):
    """T15: When both LLM attempts return invalid JSON the hardcoded fallback is returned (no crash)."""

    def test_fallback_returned_on_double_failure(self) -> None:
        # All four calls (2 intent retries + 2 response retries) return garbage
        fake = _SequenceFakeBackend(["{{INVALID", "{{INVALID", "{{INVALID", "{{INVALID"])
        ctx, extractor, generator = _make_components(fake)
        with patch("context.get_backend", return_value=fake):
            extraction, response, _, _ = _run_turn(
                "I want to go somewhere warm.", ctx, extractor, generator
            )
        self.assertIsNotNone(extraction)
        self.assertIsNotNone(response)
        self.assertIsInstance(response.response_text, str)
        self.assertGreater(len(response.response_text), 0)


class TestValidationRetryCanRecover(unittest.TestCase):
    """T16: When the first intent call returns invalid JSON but the second returns valid, the valid result is used."""

    def test_recovery_on_second_attempt(self) -> None:
        valid_intent = _intent_json("destination_recommendation", destination="Lisbon")
        valid_response = _response_json("Lisbon is a wonderful destination.")
        fake = _FakeFlakyBackend(valid_intent, valid_response)
        ctx, extractor, generator = _make_components(fake)
        with patch("context.get_backend", return_value=fake):
            extraction, response, _, _ = _run_turn("Tell me about Lisbon.", ctx, extractor, generator)
        self.assertEqual(extraction.intent, "destination_recommendation")
        self.assertIn("Lisbon", response.response_text)
        # First intent call failed → at least 3 total calls (2 intent + 1 response)
        self.assertGreaterEqual(fake.call_count, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3: Evaluator / LLM-judge tests (T21–T25)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluatorReadsTranscript(unittest.TestCase):
    """T21: _deterministic_checks passes all flags on a well-formed extraction + response pair."""

    def test_deterministic_checks_pass_on_valid_output(self) -> None:
        test = _make_eval_test(test_id="t21", requires_tool="none", must_not_claim_live_facts=False)
        extraction = IntentExtraction(
            intent="destination_recommendation",
            confidence=0.95,
            context_updates=TripContext(),
            needs_weather=False,
            needs_attractions=False,
            clarification_questions=[],
        )
        response = AssistantResponse(
            response_text="Paris is beautiful with the Eiffel Tower and excellent food.",
            used_external_data=False,
        )
        checks = _deterministic_checks(test, extraction, response)
        self.assertTrue(checks.valid_json_output)
        self.assertTrue(checks.intent_in_allowed_set)
        self.assertTrue(checks.response_length_ok)
        self.assertIsNone(checks.tool_used_when_required)  # tool was not required


class TestEvaluatorPenalizesContextFailure(unittest.TestCase):
    """T22: tool_used_when_required=False when the test requires a weather tool but used_external_data is False."""

    def test_tool_required_but_not_used(self) -> None:
        test = _make_eval_test(test_id="t22", requires_tool="weather", must_not_claim_live_facts=False)
        extraction = IntentExtraction(
            intent="weather_advice",
            confidence=0.9,
            context_updates=TripContext(),
            needs_weather=True,
            needs_attractions=False,
            clarification_questions=[],
        )
        response = AssistantResponse(
            response_text="The weather in Rome is typically nice in October.",
            used_external_data=False,  # tool was required but result not incorporated
        )
        checks = _deterministic_checks(test, extraction, response)
        self.assertFalse(checks.tool_used_when_required)


class TestEvaluatorPenalizesHallucination(unittest.TestCase):
    """T23: _deterministic_checks catches time and price claims made without tool data."""

    def _extraction(self) -> IntentExtraction:
        return IntentExtraction(
            intent="local_attractions",
            confidence=0.9,
            context_updates=TripContext(),
            needs_weather=False,
            needs_attractions=False,
            clarification_questions=[],
        )

    def test_time_claim_fails_check(self) -> None:
        test = _make_eval_test(test_id="t23a", must_not_claim_live_facts=True)
        response = AssistantResponse(
            response_text="The museum opens at 10:00 AM every day.",
            used_external_data=False,
        )
        checks = _deterministic_checks(test, self._extraction(), response)
        self.assertFalse(checks.no_claimed_live_facts_without_tool)

    def test_price_claim_fails_check(self) -> None:
        test = _make_eval_test(test_id="t23b", must_not_claim_live_facts=True)
        response = AssistantResponse(
            response_text="Entry costs $22 per person.",
            used_external_data=False,
        )
        checks = _deterministic_checks(test, self._extraction(), response)
        self.assertFalse(checks.no_claimed_live_facts_without_tool)

    def test_euro_price_claim_fails_check(self) -> None:
        test = _make_eval_test(test_id="t23c", must_not_claim_live_facts=True)
        response = AssistantResponse(
            response_text="Adult entry is €14.",
            used_external_data=False,
        )
        checks = _deterministic_checks(test, self._extraction(), response)
        self.assertFalse(checks.no_claimed_live_facts_without_tool)


@unittest.skipIf(_requires_api(), "No LLM API key configured — skipping real-LLM judge test")
class TestEvaluatorPenalizesVerbosity(unittest.TestCase):
    """T24: LLM judge scores a verbose, filler-heavy response ≤ 3 on response_quality."""

    def test_verbose_response_scores_low(self) -> None:
        test = _make_eval_test(test_id="t24", category="response_discipline")
        extraction = IntentExtraction(
            intent="weather_advice",
            confidence=0.9,
            context_updates=TripContext(),
            needs_weather=False,
            needs_attractions=False,
            clarification_questions=[],
        )
        response = AssistantResponse(
            response_text=_VERBOSE_TRANSCRIPT.split("Assistant:")[-1].strip(),
            used_external_data=False,
        )
        checks = _deterministic_checks(test, extraction, response)
        result = _llm_judge(test, _VERBOSE_TRANSCRIPT, checks)
        self.assertLessEqual(result.response_quality, 3)


class TestEvaluatorValidatorNegativeCase(unittest.TestCase):
    """T25: _llm_judge() enforces hallucination_resistance=1 via Python code override,
    regardless of what the LLM judge scores.

    Regression caught: hard override removed from _llm_judge() → a generous LLM judge
    that scores hallucination_resistance=5 on a response with live facts would pass
    uncorrected, hiding the violation from the evaluator report.
    """

    def test_hallucination_override_fires_regardless_of_llm_score(self) -> None:
        from models import DeterministicChecks

        # Mock LLM judge returns hallucination_resistance=5 (incorrectly generous)
        generous_judge_json = json.dumps({
            "intent_accuracy": 5,
            "context_handling": 5,
            "response_quality": 5,
            "hallucination_resistance": 5,
            "tool_usage_appropriateness": 5,
            "improvement_suggestion": "Nothing to improve.",
            "judge_reasoning": "Response looked great to me.",
        })

        test = _make_eval_test(
            test_id="t25",
            category="hallucination_resistance",
            must_not_claim_live_facts=True,
        )

        # Deterministic checks confirm: live facts were claimed (violation)
        checks = DeterministicChecks(
            valid_json_output=True,
            intent_in_allowed_set=True,
            response_length_ok=True,
            no_claimed_live_facts_without_tool=False,  # violation!
        )

        class _FakeJudgeBackend:
            def chat(self, messages: list, schema: type) -> str:
                return generous_judge_json

        with patch("evaluator.get_backend", return_value=_FakeJudgeBackend()):
            result = _llm_judge(test, _HALLUCINATION_TRANSCRIPT, checks)

        # Override must have fired: LLM said 5, code must enforce 1
        self.assertEqual(result.hallucination_resistance, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 4: Schema handling, hard override, and pattern unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestIsSchemaEcho(unittest.TestCase):
    """Proves _is_schema_echo correctly distinguishes schema definitions from actual output.

    Regression caught: _is_schema_echo flags actual values wrapped in "properties" as an echo →
    backend triggers unnecessary retries and may discard a valid response, causing silent fallback.
    """

    def test_schema_with_type_fields_is_detected(self) -> None:
        from context import _is_schema_echo
        schema_echo = json.dumps({
            "properties": {
                "intent_accuracy": {"type": "integer", "description": "Score 1–5"},
                "hallucination_resistance": {"type": "integer", "description": "Score 1–5"},
            }
        })
        self.assertTrue(_is_schema_echo(schema_echo))

    def test_schema_with_defs_is_detected(self) -> None:
        from context import _is_schema_echo
        schema_with_defs = json.dumps({
            "$defs": {"TripContext": {"type": "object", "properties": {}}},
            "intent": "weather_advice",
        })
        self.assertTrue(_is_schema_echo(schema_with_defs))

    def test_actual_values_wrapped_in_properties_not_detected(self) -> None:
        from context import _is_schema_echo
        # Model returned {"properties": {actual primitive values}} — not a schema echo.
        # JudgeOutput.unwrap_properties_envelope handles the unwrapping.
        wrapped_values = json.dumps({
            "properties": {
                "intent_accuracy": 5,
                "context_handling": 4,
                "hallucination_resistance": 1,
                "response_quality": 3,
                "tool_usage_appropriateness": 5,
                "improvement_suggestion": "Be more concise.",
                "judge_reasoning": "Response claimed specific prices.",
            }
        })
        self.assertFalse(_is_schema_echo(wrapped_values))

    def test_flat_actual_values_not_detected(self) -> None:
        from context import _is_schema_echo
        flat = json.dumps({
            "intent": "weather_advice",
            "confidence": 0.9,
            "needs_weather": True,
        })
        self.assertFalse(_is_schema_echo(flat))


class TestJudgeOutputUnwrapsEnvelope(unittest.TestCase):
    """Proves JudgeOutput.model_validate_json handles the {"properties": {values}} envelope.

    Regression caught: model_validator removed → when a model wraps actual scores inside a
    "properties" key, Pydantic validation fails and the silent fallback returns 3.0 on all
    dimensions, hiding the real scores and masking any override that should have fired.
    """

    def test_flat_values_validate_normally(self) -> None:
        from models import JudgeOutput
        flat = json.dumps({
            "intent_accuracy": 5,
            "context_handling": 4,
            "response_quality": 3,
            "hallucination_resistance": 1,
            "tool_usage_appropriateness": 4,
            "improvement_suggestion": "Test.",
            "judge_reasoning": "Test reasoning.",
        })
        judge = JudgeOutput.model_validate_json(flat)
        self.assertEqual(judge.hallucination_resistance, 1)

    def test_properties_envelope_is_unwrapped_and_validates(self) -> None:
        from models import JudgeOutput
        wrapped = json.dumps({
            "properties": {
                "intent_accuracy": 5,
                "context_handling": 4,
                "response_quality": 3,
                "hallucination_resistance": 1,
                "tool_usage_appropriateness": 4,
                "improvement_suggestion": "Test.",
                "judge_reasoning": "Test reasoning.",
            }
        })
        # Without model_validator, model_validate_json raises ValidationError
        judge = JudgeOutput.model_validate_json(wrapped)
        self.assertEqual(judge.hallucination_resistance, 1)


class TestExpandedTimePattern(unittest.TestCase):
    """Proves the expanded _TIME_PATTERN catches 24h ranges and compact am/pm forms.

    Regression caught: _TIME_PATTERN reverted to AM/PM-only → assistant responses claiming
    '10:00 - 18:00' or '9am' opening hours pass the hallucination check undetected.
    """

    def test_24h_range_detected(self) -> None:
        from evaluator import _TIME_PATTERN
        self.assertTrue(bool(_TIME_PATTERN.search("The museum is open 10:00 - 18:00 daily.")))

    def test_compact_am_pm_detected(self) -> None:
        from evaluator import _TIME_PATTERN
        self.assertTrue(bool(_TIME_PATTERN.search("Gates open at 9am and close at 6pm.")))

    def test_safe_text_not_detected(self) -> None:
        from evaluator import _TIME_PATTERN
        safe = "check the official website for current opening hours"
        self.assertFalse(bool(_TIME_PATTERN.search(safe)))


class TestAutoCorrectObservability(unittest.TestCase):
    """Proves low_confidence_must_clarify emits a log line to stderr when it fires.

    Regression caught: observability removed → auto-corrections happen silently,
    making it impossible to detect systematic LLM under-confidence in production logs.
    """

    def test_auto_correct_logs_to_stderr(self) -> None:
        import io
        import sys
        captured = io.StringIO()
        with patch("sys.stderr", captured):
            IntentExtraction(
                intent="itinerary_planning",
                confidence=0.4,
                context_updates=TripContext(),
                needs_weather=False,
                needs_attractions=False,
                clarification_questions=[],
            )
        self.assertIn("auto-correct", captured.getvalue().lower())


class TestResolveSchema(unittest.TestCase):
    """Proves _resolve_schema inlines $ref/$defs so backends pass a flat schema to models.

    Regression caught: _resolve_schema stops resolving $refs → backends pass nested schema
    containing $defs to the model, which copies the definition structure back verbatim
    instead of filling values, causing Pydantic validation to fail on every DeepSeek call.
    """

    def test_schema_without_defs_returned_unchanged(self) -> None:
        from context import _resolve_schema
        flat = {
            "type": "object",
            "properties": {
                "intent": {"type": "string"},
                "confidence": {"type": "number"},
            },
        }
        result = _resolve_schema(flat)
        self.assertNotIn("$defs", result)
        self.assertEqual(result["properties"]["intent"]["type"], "string")

    def test_ref_replaced_with_inline_definition(self) -> None:
        from context import _resolve_schema
        schema = {
            "$defs": {
                "TripContext": {
                    "type": "object",
                    "properties": {"destination": {"type": "string"}},
                }
            },
            "properties": {
                "context_updates": {"$ref": "#/$defs/TripContext"},
                "intent": {"type": "string"},
            },
        }
        result = _resolve_schema(schema)
        # $defs stripped from top level — model never sees cross-references
        self.assertNotIn("$defs", result)
        # $ref replaced with the actual definition object
        context_updates = result["properties"]["context_updates"]
        self.assertNotIn("$ref", context_updates)
        self.assertEqual(context_updates.get("type"), "object")


class TestCLISmokeTest(unittest.TestCase):
    """Proves main() completes one conversation turn without an unhandled exception.

    Regression caught: a wiring change in main() (wrong import, missing attribute, broken
    _Thinking spin-up) causes the assistant to crash before printing any response,
    making the entire CLI unusable without surfacing a clear error.
    """

    def test_single_turn_does_not_crash(self) -> None:
        from main import main

        class _NoOpThinking:
            def __init__(self, label: str = "") -> None:
                pass
            def __enter__(self) -> "_NoOpThinking":
                return self
            def __exit__(self, *_: object) -> None:
                pass

        class _FakeMainBackend:
            def chat(self, messages: list, schema: type) -> str:
                if schema is IntentExtraction:
                    return _intent_json("destination_recommendation", destination="Rome")
                return _response_json("Rome is wonderful in spring.")

        with patch("builtins.input", side_effect=["I want to go to Rome", EOFError()]), \
             patch("main._Thinking", _NoOpThinking), \
             patch("context.get_backend", return_value=_FakeMainBackend()):
            main()  # raises → test fails; clean exit → test passes


class TestDeepSeekBackendSmoke(unittest.TestCase):
    """Smoke tests for _DeepSeekBackend — run against a live API when DEEPSEEK_API_KEY is set.

    Skipped when DEEPSEEK_API_KEY is absent.

    Regression caught: _DeepSeekBackend constructor fails (bad base_url, missing env var,
    broken openai import) → crashes at startup with no useful error message.
    """

    @classmethod
    def setUpClass(cls) -> None:
        key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not key:
            raise unittest.SkipTest("DeepSeek smoke tests require DEEPSEEK_API_KEY set")

    def _get_backend(self):
        from context import _DeepSeekBackend
        return _DeepSeekBackend()

    def test_backend_initialises_without_error(self) -> None:
        """_DeepSeekBackend.__init__ completes and exposes a _client and _model."""
        backend = self._get_backend()
        self.assertIsNotNone(backend._client)
        self.assertIsInstance(backend._model, str)
        self.assertTrue(len(backend._model) > 0)

    def test_intent_extraction_returns_valid_json(self) -> None:
        """A real intent-extraction call returns JSON that Pydantic accepts as IntentExtraction."""
        backend = self._get_backend()
        from prompts import INTENT_SYSTEM_ADDENDUM, SYSTEM_PROMPT, format_intent_user_prompt
        from models import TripContext, IntentExtraction

        trip_ctx = TripContext()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + INTENT_SYSTEM_ADDENDUM},
            {"role": "user", "content": format_intent_user_prompt(
                user_message="I want to visit Paris for 5 days.",
                history=[],
                trip_context=trip_ctx,
            )},
        ]
        raw = backend.chat(messages, IntentExtraction)
        extraction = IntentExtraction.model_validate_json(raw)
        self.assertIn(extraction.intent, {
            "destination_recommendation", "itinerary_planning", "context_update",
            "general_travel_qa", "clarification_needed",
        })
        self.assertGreaterEqual(extraction.confidence, 0.0)
        self.assertLessEqual(extraction.confidence, 1.0)

    def test_response_generation_returns_valid_json(self) -> None:
        """A real response-generation call returns JSON that Pydantic accepts as AssistantResponse."""
        backend = self._get_backend()
        from prompts import RESPONSE_SYSTEM_ADDENDUM, SYSTEM_PROMPT, format_response_user_prompt
        from models import TripContext, AssistantResponse

        trip_ctx = TripContext(destination="Paris", duration_days=5)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + RESPONSE_SYSTEM_ADDENDUM},
            {"role": "user", "content": format_response_user_prompt(
                user_message="What should I pack for Paris in October?",
                intent="packing_advice",
                trip_context=trip_ctx,
                history=[],
                weather=None,
                attractions=None,
            )},
        ]
        raw = backend.chat(messages, AssistantResponse)
        response = AssistantResponse.model_validate_json(raw)
        self.assertIsInstance(response.response_text, str)
        self.assertGreater(len(response.response_text), 10)
        self.assertIsInstance(response.used_external_data, bool)

    def test_schema_echo_does_not_crash(self) -> None:
        """If the model returns a schema echo, the retry mechanism recovers without raising."""
        from context import _DeepSeekBackend
        from models import IntentExtraction

        backend = _DeepSeekBackend()
        schema_echo = '{"$defs": {"TripContext": {}}, "properties": {"intent": {"type": "string"}}}'

        call_count = {"n": 0}
        original_call = backend._call

        def _patched_call(messages: list) -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return schema_echo
            return _intent_json("general_travel_qa")

        backend._call = _patched_call  # type: ignore[method-assign]

        from prompts import INTENT_SYSTEM_ADDENDUM, SYSTEM_PROMPT, format_intent_user_prompt
        from models import TripContext
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + INTENT_SYSTEM_ADDENDUM},
            {"role": "user", "content": format_intent_user_prompt(
                user_message="tell me about Rome",
                history=[],
                trip_context=TripContext(),
            )},
        ]
        raw = backend.chat(messages, IntentExtraction)
        extraction = IntentExtraction.model_validate_json(raw)
        self.assertEqual(call_count["n"], 2, "Expected exactly one retry after schema echo")
        self.assertIsInstance(extraction.intent, str)


if __name__ == "__main__":
    unittest.main()
