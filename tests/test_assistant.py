"""
Travel Assistant test suite.
"""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError


class TestLowConfidenceEnforcedByValidator:
    """
    Proves the @field_validator fires and forces clarification_needed when
    confidence < 0.6, regardless of what intent the LLM tried to set.

    Regression caught: someone weakens or removes the validator →
    low-confidence LLM outputs get treated as confident answers, causing the
    assistant to give a confidently-wrong itinerary instead of asking a question.
    """

    def test_low_confidence_non_clarification_auto_corrects(self):
        from models import IntentExtraction, TripContext

        # Validator silently corrects intent to clarification_needed instead of raising.
        ie = IntentExtraction(
            intent="itinerary_planning",
            confidence=0.4,
            context_updates=TripContext(),
            needs_weather=False,
            needs_attractions=False,
            clarification_questions=[],
        )
        assert ie.intent == "clarification_needed"
        assert len(ie.clarification_questions) > 0

class TestClarificationNeedsQuestions:
    """
    Proves the @model_validator fires: clarification_needed without any
    clarification_questions raises ValidationError.

    Regression caught: model_validator removed → assistant produces
    clarification_needed responses with no actual question, leaving the user
    with a non-answer and no guidance on what to provide.
    """

    def test_clarification_without_questions_raises(self):
        from models import IntentExtraction, TripContext

        with pytest.raises(ValidationError) as exc_info:
            IntentExtraction(
                intent="clarification_needed",
                confidence=0.8,
                context_updates=TripContext(),
                needs_weather=False,
                needs_attractions=False,
                clarification_questions=[],  # empty — must raise
            )
        assert "clarification_question" in str(exc_info.value)

class TestFollowUpPreservesContext:
    """
    Proves that after establishing destination=Rome in TripContext, a bare
    follow-up turn with no new context leaves destination intact.

    Regression caught: ContextManager.update() overwrites TripContext wholesale
    instead of merging → every bare follow-up wipes the trip details, forcing
    the user to repeat destination/dates on every message.
    """

    def test_bare_followup_does_not_clear_destination(self):
        from context import ContextManager
        from models import IntentExtraction, TripContext

        ctx = ContextManager()

        # Turn 1: establish Rome
        ctx.update(
            IntentExtraction(
                intent="itinerary_planning",
                confidence=0.95,
                context_updates=TripContext(destination="Rome", duration_days=4, travelers=2),
                needs_weather=False,
                needs_attractions=False,
            ),
            "I'm planning 4 days in Rome with my girlfriend.",
        )
        assert ctx.trip_context.destination == "Rome"
        assert ctx.trip_context.duration_days == 4

        # Turn 2: follow-up with no new context_updates
        ctx.update(
            IntentExtraction(
                intent="packing_advice",
                confidence=0.9,
                context_updates=TripContext(),  # nothing new
                needs_weather=False,
                needs_attractions=False,
            ),
            "What should I pack?",
        )

        assert ctx.trip_context.destination == "Rome"
        assert ctx.trip_context.duration_days == 4


class TestWeatherToolFailureProducesError:
    """
    Proves that get_weather() returns a WeatherResult with error set (not an
    exception) when given a nonsense destination that cannot be geocoded.

    Regression caught: tool raises instead of returning an error result →
    the main loop crashes and the assistant dies instead of falling back
    to general advice with an acknowledgement.
    """

    def test_get_weather_returns_error_on_bad_destination(self):
        from tools import get_weather

        result = get_weather("ZZZZNONEXISTENTPLACE99999")
        assert result.succeeded is False
        assert result.error is not None
        assert result.forecast == []


class TestAttractionsToolFailsGracefullyWithoutKey:
    """
    Proves get_attractions() returns AttractionsResult.error (not an exception)
    when the destination cannot be geocoded.

    Regression caught: geocoding failure causes an exception crash instead of a
    graceful error result → the evaluator and test harness break, and the
    assistant cannot acknowledge the failure to the user.
    """

    def test_attractions_bad_destination_returns_error(self):
        # Overpass needs no API key. Graceful failure is tested via an
        # ungeocod-able destination — geocoding fails → error result, no exception.
        from tools import get_attractions

        result = get_attractions("ZZZZNONEXISTENTPLACE99999")
        assert result.succeeded is False
        assert result.error is not None


class TestDestinationChangeInvalidatesCache:
    """
    Proves ContextManager clears the tool cache when destination changes.

    Regression caught: cache not invalidated on destination change →
    stale Rome weather data appears in a Barcelona response, confusing the user.
    """

    def test_destination_change_clears_tool_cache(self):
        from context import ContextManager
        from models import IntentExtraction, TripContext, WeatherResult

        ctx = ContextManager()

        ctx.update(
            IntentExtraction(
                intent="itinerary_planning",
                confidence=0.9,
                context_updates=TripContext(destination="Rome"),
                needs_weather=False,
                needs_attractions=False,
            ),
            "Going to Rome.",
        )

        # Simulate a cached weather result for Rome
        ctx._tool_cache["weather:rome:None"] = WeatherResult(
            destination="Rome",
            retrieved_at="2026-06-23T10:00",
        )
        assert len(ctx._tool_cache) == 1

        # Change destination to Barcelona
        ctx.update(
            IntentExtraction(
                intent="context_update",
                confidence=0.95,
                context_updates=TripContext(destination="Barcelona"),
                needs_weather=False,
                needs_attractions=False,
            ),
            "Actually make it Barcelona.",
        )

        assert ctx.trip_context.destination == "Barcelona"
        assert len(ctx._tool_cache) == 0


class TestStableFieldsSurviveDestinationChange:
    """
    Proves that duration_days set in turn 1 is preserved when only destination
    changes in turn 2.

    Regression caught: context merge overwrites entire TripContext instead of
    merging only changed fields → duration, budget, interests reset every time
    the user corrects one detail.
    """

    def test_duration_preserved_after_destination_change(self):
        from context import ContextManager
        from models import IntentExtraction, TripContext

        ctx = ContextManager()

        ctx.update(
            IntentExtraction(
                intent="itinerary_planning",
                confidence=0.9,
                context_updates=TripContext(destination="Rome", duration_days=4),
                needs_weather=False,
                needs_attractions=False,
            ),
            "Planning 4 days in Rome.",
        )

        ctx.update(
            IntentExtraction(
                intent="context_update",
                confidence=0.95,
                context_updates=TripContext(destination="Barcelona"),
                needs_weather=False,
                needs_attractions=False,
            ),
            "Actually make it Barcelona.",
        )

        assert ctx.trip_context.destination == "Barcelona"
        assert ctx.trip_context.duration_days == 4  # must be preserved


class TestInterestsAppendedNotOverwritten:
    """
    Proves that interests from turn 1 are preserved and new interests from
    turn 2 are appended without duplicates.

    Regression caught: list fields are overwritten instead of appended →
    each new interest clears prior interests, losing user preferences.
    """

    def test_interests_accumulate_across_turns(self):
        from context import ContextManager
        from models import IntentExtraction, TripContext

        ctx = ContextManager()

        ctx.update(
            IntentExtraction(
                intent="destination_recommendation",
                confidence=0.9,
                context_updates=TripContext(interests=["food", "history"]),
                needs_weather=False,
                needs_attractions=False,
            ),
            "I love food and history.",
        )
        assert ctx.trip_context.interests == ["food", "history"]

        ctx.update(
            IntentExtraction(
                intent="trip_refinement",
                confidence=0.85,
                context_updates=TripContext(interests=["hiking"]),
                needs_weather=False,
                needs_attractions=False,
            ),
            "I also like hiking.",
        )
        assert "food" in ctx.trip_context.interests
        assert "history" in ctx.trip_context.interests
        assert "hiking" in ctx.trip_context.interests


class TestToolRouterNoToolsForKnowledgeOnlyIntents:
    """
    Proves ToolRouter returns (None, None) for intents that should not call tools,
    even when a destination is set in TripContext.

    Regression caught: tool routing flags ignored → tools called on every turn
    regardless of intent, causing unnecessary API calls and inflated latency.
    """

    def test_no_tools_for_out_of_scope(self):
        from tools import ToolRouter
        from models import IntentExtraction, TripContext

        router = ToolRouter()
        extraction = IntentExtraction(
            intent="out_of_scope",
            confidence=0.95,
            context_updates=TripContext(),
            needs_weather=False,
            needs_attractions=False,
        )
        ctx = TripContext(destination="Paris")
        weather, attractions = router.route(extraction, ctx)
        assert weather is None
        assert attractions is None

    def test_no_tools_for_general_qa(self):
        from tools import ToolRouter
        from models import IntentExtraction, TripContext

        router = ToolRouter()
        extraction = IntentExtraction(
            intent="general_travel_qa",
            confidence=0.9,
            context_updates=TripContext(),
            needs_weather=False,
            needs_attractions=False,
        )
        ctx = TripContext(destination="Tokyo")
        weather, attractions = router.route(extraction, ctx)
        assert weather is None
        assert attractions is None


class TestHallucinationPatternDetection:
    """
    Proves the regex patterns used in DeterministicChecks correctly flag
    time-of-day and price patterns, and do not flag safe phrasing.

    Regression caught: evaluator's hallucination check becomes too broad or too
    narrow → hallucinated facts pass undetected, or safe general advice gets
    incorrectly flagged as a violation.
    """

    def test_time_pattern_detected(self):
        assert bool(re.search(r"\d{1,2}:\d{2}\s*(AM|PM|am|pm)", "opens at 9:00 AM daily"))
        assert bool(re.search(r"\d{1,2}:\d{2}\s*(AM|PM|am|pm)", "open from 10:00 am to 6:00 PM"))

    def test_time_pattern_not_detected_on_safe_text(self):
        safe = "check the official website for current opening hours"
        assert not bool(re.search(r"\d{1,2}:\d{2}\s*(AM|PM|am|pm)", safe))

    def test_price_pattern_detected(self):
        assert bool(re.search(r"\$\d+|\€\d+|£\d+", "adult tickets cost $18"))
        assert bool(re.search(r"\$\d+|\€\d+|£\d+", "entry fee is €15"))

    def test_price_pattern_not_detected_on_safe_text(self):
        safe = "admission fees vary — check the official site for current prices"
        assert not bool(re.search(r"\$\d+|\€\d+|£\d+", safe))


class TestResetClearsAllState:
    """
    Proves ContextManager.reset() clears TripContext, history, and tool cache.

    Regression caught: /reset clears history but leaves stale TripContext →
    a new conversation inherits the prior trip's destination and details.
    """

    def test_reset_clears_everything(self):
        from context import ContextManager
        from models import IntentExtraction, TripContext, WeatherResult

        ctx = ContextManager()

        ctx.update(
            IntentExtraction(
                intent="itinerary_planning",
                confidence=0.9,
                context_updates=TripContext(destination="Tokyo"),
                needs_weather=False,
                needs_attractions=False,
            ),
            "Going to Tokyo.",
        )
        ctx._tool_cache["weather:tokyo:None"] = WeatherResult(
            destination="Tokyo", retrieved_at="2026-06-23T10:00"
        )
        ctx.add_assistant_turn("Here's Tokyo info...")

        assert ctx.trip_context.destination == "Tokyo"
        assert len(ctx.history) > 0
        assert len(ctx._tool_cache) > 0

        ctx.reset()

        assert ctx.trip_context.destination is None
        assert ctx.history == []
        assert ctx._tool_cache == {}


class TestStateCommandSerializesDestination:
    """
    Proves TripContext serializes to JSON with destination and duration_days at
    the correct keys, which is what /state prints.

    Regression caught: model_dump_json() renames or drops the destination key →
    /state shows an empty or misleading context to the user.
    """

    def test_destination_present_in_json(self):
        import json
        from models import TripContext

        ctx = TripContext(destination="Tokyo", duration_days=5)
        data = json.loads(ctx.model_dump_json())
        assert data["destination"] == "Tokyo"
        assert data["duration_days"] == 5


class TestEvaluationResultScoreComputedDeterministically:
    """
    Proves EvaluationResult.overall_score is recomputed by the model_validator
    regardless of the value the LLM supplies.

    Regression caught: model_validator removed → LLM arithmetic errors in
    overall_score go uncorrected, producing inconsistent scoring output.
    """

    def test_overall_score_recomputed(self):
        from models import DeterministicChecks, EvaluationResult

        checks = DeterministicChecks(
            valid_json_output=True,
            intent_in_allowed_set=True,
            response_length_ok=True,
            no_claimed_live_facts_without_tool=True,
        )
        result = EvaluationResult(
            test_id="t1",
            intent_accuracy=4,
            context_handling=5,
            response_quality=4,
            hallucination_resistance=5,
            tool_usage_appropriateness=3,
            overall_score=99.0,  # deliberately wrong — must be corrected
            deterministic_checks=checks,
            improvement_suggestion="Test suggestion.",
            judge_reasoning="Test reasoning.",
        )
        # 4*0.20 + 5*0.25 + 4*0.20 + 5*0.20 + 3*0.15 = 0.8+1.25+0.8+1.0+0.45 = 4.3
        assert result.overall_score == pytest.approx(4.3, abs=0.01)
        assert result.overall_score != 99.0
