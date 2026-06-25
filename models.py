"""
Pydantic data contracts for the Travel Assistant.

Every typed boundary (LLM input/output, persistent state, tool result, evaluation)
is defined here. Field descriptions feed directly into the LLM prompt via
schema-constrained decoding — they are part of the prompt, not just documentation.
"""

from __future__ import annotations

import sys
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


def _unwrap_llm_envelope(data: object, known_fields: frozenset[str]) -> object:
    """Unwrap two wrapping patterns LLMs produce instead of flat JSON values.

    Pattern 1: {"properties": {actual_values}} — present when the model echoes
               the schema envelope but fills the inner dict with real values.
    Pattern 2: {field: {"value": X, "type": "…"}} — per-field metadata objects
               produced by some models (e.g. gpt-4o-mini) that wrap each value
               in a tiny schema descriptor.
    """
    if not isinstance(data, dict):
        return data

    # Pattern 1: "properties" envelope wrapping actual values
    if "properties" in data and isinstance(data.get("properties"), dict):
        props = data["properties"]
        if known_fields & set(props.keys()):
            data = props

    # Pattern 2: per-field {"value": X, ...} wrappers
    if isinstance(data, dict) and any(
        isinstance(v, dict) and "value" in v for v in data.values()
    ):
        data = {
            k: v["value"] if isinstance(v, dict) and "value" in v else v
            for k, v in data.items()
        }

    return data


# ── Persistent trip state ─────────────────────────────────────────────────────

class TripContext(BaseModel):
    """Stable facts about the user's trip that persist across conversation turns.
    Serialized as JSON into every LLM prompt so the model always sees current state."""

    model_config = ConfigDict(extra="forbid")

    destination: str | None = Field(
        default=None,
        description=(
            "City or region the user is travelling to. "
            "Set only when the user explicitly names a destination this turn. "
            "Do not infer from vague phrases like 'somewhere warm' — leave None. "
            "When the user changes destination mid-conversation, overwrite this field."
        ),
    )
    start_date: str | None = Field(
        default=None,
        description=(
            "Trip start date in ISO 8601 format (YYYY-MM-DD). "
            "If the user says 'next Friday', convert to the absolute date. "
            "Do not fill if the user only mentions a season or month."
        ),
    )
    end_date: str | None = Field(
        default=None,
        description="Trip end date in ISO 8601 format (YYYY-MM-DD). Fill only if stated.",
    )
    duration_days: int | None = Field(
        default=None,
        ge=1,
        le=60,
        description=(
            "Total trip length in days. "
            "If start_date and end_date are both known, compute this. "
            "If the user says '3 days', fill directly."
        ),
    )
    travelers: int | None = Field(
        default=None,
        description=(
            "Number of people travelling. Set to an integer 1–20. "
            "Map: 'couple' → 2, 'family of four' → 4, 'just me' → 1. "
            "Leave null if unspecified."
        ),
    )
    budget_level: Literal["shoestring", "moderate", "premium"] | None = Field(
        default=None,
        description=(
            "Coarse budget bucket. Map free-form input: "
            "'cheap'/'tight'/'backpacker' → shoestring; "
            "'comfortable'/'mid-range'/'normal' → moderate; "
            "'no limit'/'luxury'/'splurge' → premium. "
            "Leave None if user has not indicated budget."
        ),
    )
    interests: list[str] = Field(
        default_factory=list,
        description=(
            "Free-form interests the user mentioned (e.g. 'food', 'history', 'hiking'). "
            "Append new interests from this turn; do not remove prior ones unless asked."
        ),
    )
    constraints: list[str] = Field(
        default_factory=list,
        description=(
            "Hard constraints and soft preferences the user stated. Append from this turn; never remove. "
            "Capture ALL of the following when mentioned: "
            "activity exclusions ('no museums', 'no nightlife', 'no beaches'); "
            "mobility/accessibility ('wheelchair accessible', 'limited walking', 'no stairs'); "
            "dietary ('vegetarian', 'vegan', 'halal', 'kosher', 'nut allergy'); "
            "crowd preferences ('avoid tourist traps', 'off the beaten path'); "
            "safety preferences ('safety priority'); "
            "transport preferences ('no driving', 'has rental car', 'public transport only'); "
            "accommodation preferences ('city center preferred', 'near nature'); "
            "group composition details ('traveling with toddlers', 'elderly traveler in group'). "
            "Use short, consistent phrases."
        ),
    )
    climate_preference: Literal["cold", "mild", "warm", "any"] | None = Field(
        default=None,
        description=(
            "The user's preferred climate. Map: "
            "'cold'/'cool'/'chilly'/'not too hot' → cold; "
            "'mild'/'temperate'/'not extreme' → mild; "
            "'warm'/'hot'/'sunny'/'beach weather' → warm; "
            "no preference stated → any. "
            "Set when the user expresses a climate preference, even indirectly "
            "(e.g. 'I want to escape the heat' → cold; 'I love the sun' → warm). "
            "Do not overwrite once set unless the user explicitly changes it."
        ),
    )
    group_type: Literal["solo", "couple", "family", "friends", "business"] | None = Field(
        default=None,
        description=(
            "The social context of the trip. Map: "
            "'just me'/'alone'/'solo' → solo; "
            "'my partner'/'wife'/'husband'/'girlfriend'/'boyfriend'/'romantic' → couple; "
            "'kids'/'children'/'family' → family; "
            "'friends'/'group of friends'/'guys trip'/'girls trip' → friends; "
            "'work trip'/'conference'/'business' → business. "
            "Leave null if unspecified."
        ),
    )
    pace: Literal["relaxed", "moderate", "packed"] | None = Field(
        default=None,
        description=(
            "Preferred trip pace. Map: 'slow'/'chill'/'relaxed' → relaxed; "
            "'normal'/'balanced' → moderate; 'jam-packed'/'see everything' → packed."
        ),
    )
    last_topic: str | None = Field(
        default=None,
        description=(
            "Short phrase for what the last assistant response covered "
            "(e.g. 'itinerary', 'packing', 'weather', 'recommendation'). "
            "Used to disambiguate vague follow-ups like 'make it cheaper'."
        ),
    )


# ── Conversation turn ─────────────────────────────────────────────────────────

class ConversationTurn(BaseModel):
    """One turn in the conversation. Serialized into the prompt history window."""

    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant"] = Field(
        description="Who produced this turn."
    )
    content: str = Field(
        description="The text of this turn. For assistant turns: response_text only."
    )


# ── Intent extraction output ──────────────────────────────────────────────────

class IntentExtraction(BaseModel):
    """Structured output from the intent-extraction LLM call.
    Schema is sent via format= for constrained decoding."""

    @model_validator(mode="before")
    @classmethod
    def unwrap_llm_envelope(cls, data: object) -> object:
        return _unwrap_llm_envelope(data, frozenset({"intent", "confidence", "needs_weather", "needs_attractions"}))

    intent: Literal[
        "destination_recommendation",
        "itinerary_planning",
        "packing_advice",
        "local_attractions",
        "weather_advice",
        "trip_refinement",
        "context_update",
        "clarification_needed",
        "general_travel_qa",
        "out_of_scope",
    ] = Field(
        description=(
            "What the user is asking for in this turn. "
            "Choose the single most applicable intent. "
            "If confidence < 0.6, this MUST be 'clarification_needed'."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Your confidence (0.0–1.0) that this is the correct intent. "
            "Be honest: vague or ambiguous messages should have confidence < 0.6, "
            "which forces clarification_needed."
        ),
    )
    context_updates: TripContext = Field(
        description=(
            "Trip facts the user explicitly stated in THIS TURN ONLY. "
            "Do not copy prior TripContext fields — the orchestrator merges state. "
            "Leave all fields at defaults unless the user said something new."
        )
    )
    needs_weather: bool = Field(
        default=False,
        description=(
            "True ONLY if answering requires a live weather forecast. "
            "Qualifies: 'will it rain next week in Rome', 'should I bring a jacket this Friday'. "
            "Does NOT qualify: 'is October good for Rome' (that is general knowledge)."
        )
    )
    needs_attractions: bool = Field(
        default=False,
        description=(
            "True ONLY if the user wants specific places for a known destination. "
            "Examples: 'what should I do in Barcelona', 'top attractions in Rome'. "
            "False for general travel Q&A or unknown destination."
        )
    )
    clarification_questions: Annotated[list[str], Field(max_length=2)] = Field(
        default_factory=list,
        description=(
            "If intent is clarification_needed, list 1–2 specific questions. "
            "Empty for all other intents. Do not ask for info already in TripContext."
        ),
    )

    @model_validator(mode="after")
    def low_confidence_must_clarify(self) -> "IntentExtraction":
        """Runs after all fields are set, so confidence is always available.
        Auto-corrects instead of raising: if confidence < 0.6 and intent isn't
        clarification_needed, silently fix it so the pipeline never crashes on this."""
        if self.confidence < 0.6 and self.intent != "clarification_needed":
            sys.stderr.write(
                f"[auto-correct] intent={self.intent!r} confidence={self.confidence:.2f}"
                " → clarification_needed\n"
            )
            object.__setattr__(self, "intent", "clarification_needed")
            if not self.clarification_questions:
                object.__setattr__(self, "clarification_questions", ["Could you tell me more about what you're looking for?"])
        return self

    @model_validator(mode="after")
    def clarification_needs_questions(self) -> "IntentExtraction":
        if self.intent == "clarification_needed" and not self.clarification_questions:
            raise ValueError(
                "intent='clarification_needed' requires at least one clarification_question"
            )
        return self


# ── Tool results ──────────────────────────────────────────────────────────────

class WeatherDay(BaseModel):
    date: str = Field(description="ISO 8601 date for this forecast day.")
    temp_high_c: float | None = Field(default=None, description="High temperature in Celsius.")
    temp_low_c: float | None = Field(default=None, description="Low temperature in Celsius.")
    condition: str | None = Field(
        default=None,
        description="Human-readable condition summary, e.g. 'partly cloudy', 'heavy rain'.",
    )
    precipitation_mm: float | None = Field(
        default=None, description="Total precipitation in mm."
    )


class WeatherResult(BaseModel):
    """Result from the weather tool. error field signals failure."""

    destination: str
    retrieved_at: str = Field(description="ISO 8601 timestamp of retrieval.")
    forecast: list[WeatherDay] = Field(default_factory=list)
    error: str | None = Field(
        default=None,
        description=(
            "If non-None, the tool failed with this message. "
            "The response generator must acknowledge the failure briefly."
        ),
    )

    @property
    def succeeded(self) -> bool:
        return self.error is None


class AttractionItem(BaseModel):
    name: str
    kinds: list[str] = Field(
        default_factory=list,
        description="OpenTripMap category tags, e.g. ['museums', 'historic'].",
    )
    rating: float | None = Field(default=None, ge=0.0, le=10.0)
    description: str | None = Field(default=None)
    wikidata: str | None = Field(default=None)


class AttractionsResult(BaseModel):
    """Result from the attractions tool. error field signals failure."""

    destination: str
    retrieved_at: str
    attractions: list[AttractionItem] = Field(default_factory=list)
    error: str | None = Field(
        default=None,
        description="If non-None, tool failed. Response generator must acknowledge briefly.",
    )

    @property
    def succeeded(self) -> bool:
        return self.error is None


# ── LLM response output ───────────────────────────────────────────────────────

class AssistantResponse(BaseModel):
    """Output of the response-generation LLM call.
    Only response_text is shown to the user."""

    @model_validator(mode="before")
    @classmethod
    def unwrap_llm_envelope(cls, data: object) -> object:
        return _unwrap_llm_envelope(data, frozenset({"response_text", "used_external_data", "internal_summary_update"}))

    response_text: str = Field(
        description=(
            "Natural-language response to show the user. "
            "No JSON, no internal field names. "
            "Lead with the answer. No filler like 'I hope this helps.' "
            "At most 1–2 clarifying questions at the end if needed."
        )
    )
    internal_summary_update: str | None = Field(
        default=None,
        description=(
            "One sentence summarizing what changed this turn. "
            "For internal logging only — never shown to the user."
        ),
    )
    used_external_data: bool = Field(
        default=False,
        description=(
            "True if response_text incorporates live data from a tool result "
            "(weather forecast values or specific named attractions from the API)."
        ),
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

class EvaluationTest(BaseModel):
    """A single evaluation scenario."""

    test_id: str
    category: Literal[
        "rephrasing_consistency",
        "followup_context",
        "weather_tool_usage",
        "attractions_tool_usage",
        "llm_knowledge_only",
        "ambiguous_request",
        "hallucination_resistance",
        "context_change",
        "response_discipline",
        "incremental_info_gathering",
    ]
    conversation: list[dict] = Field(
        description="List of {role, content} dicts forming the test input conversation."
    )
    expected_intent: str | None = Field(default=None)
    requires_tool: Literal["weather", "attractions", "none", "any"] = Field(
        description="Which tool must be called for this test to pass the tool-usage check."
    )
    should_clarify: bool = Field(
        description="True if the expected behavior is to ask a clarifying question."
    )
    must_not_claim_live_facts: bool = Field(
        description="True if the scenario has no tool data and the response must not state specific live facts."
    )


class DeterministicChecks(BaseModel):
    """Pre-computed checks run before the LLM judge. Fast, zero-LLM-cost."""

    valid_json_output: bool
    intent_in_allowed_set: bool
    response_length_ok: bool = Field(
        description="True if response_text < 2500 characters."
    )
    tool_used_when_required: bool | None = Field(
        default=None,
        description="None if tool was not required. True/False if tool was required.",
    )
    no_claimed_live_facts_without_tool: bool
    fabricated_plan_when_should_clarify: bool | None = Field(
        default=None,
        description=(
            "None when should_clarify is False. "
            "True = no itinerary markers found when should_clarify=True (pass). "
            "False = itinerary markers found when should_clarify=True (fail — plan was fabricated)."
        ),
    )
    day_count_matches_duration: bool | None = Field(
        default=None,
        description=(
            "None when trip_context.duration_days is not set. "
            "True when the count of 'Day N' headers in the response equals duration_days. "
            "False when the counts differ."
        ),
    )
    confirmation_prompt_emitted: bool | None = Field(
        default=None,
        description=(
            "None when not applicable. "
            "True when the response contains a draft-offer phrase "
            "('want me to draft', 'shall I draft', etc.) — used for incremental_02."
        ),
    )


class JudgeOutput(BaseModel):
    """LLM judge output schema. test_id and deterministic_checks are added by the evaluator."""

    @model_validator(mode="before")
    @classmethod
    def unwrap_llm_envelope(cls, data: object) -> object:
        return _unwrap_llm_envelope(data, frozenset({"intent_accuracy", "hallucination_resistance", "context_handling"}))

    intent_accuracy: int = Field(
        ge=1,
        le=5,
        description="Did the assistant correctly identify what the user wanted? 5=perfect.",
    )
    context_handling: int = Field(
        ge=1,
        le=5,
        description="Did follow-ups correctly use prior context? 5=perfect.",
    )
    response_quality: int = Field(
        ge=1,
        le=5,
        description="Is the response concise, practical, and leading with the answer? 5=excellent.",
    )
    hallucination_resistance: int = Field(
        ge=1,
        le=5,
        description=(
            "Did the assistant avoid inventing specific live facts "
            "(hours, prices, current weather) without tool data? 5=no hallucination at all."
        ),
    )
    tool_usage_appropriateness: int = Field(
        ge=1,
        le=5,
        description=(
            "Were tools called when needed and NOT called when not needed? 5=perfect."
        ),
    )
    improvement_suggestion: str = Field(
        description=(
            "One concrete, specific suggestion: name the prompt section or behavior to change "
            "and explain why. Not generic advice like 'be more helpful'."
        )
    )
    judge_reasoning: str = Field(
        description="2–4 sentences explaining the scores. Shown in evaluator output."
    )


class EvaluationResult(BaseModel):
    """Complete evaluation result combining judge output and deterministic checks."""

    test_id: str
    intent_accuracy: int = Field(ge=1, le=5)
    context_handling: int = Field(ge=1, le=5)
    response_quality: int = Field(ge=1, le=5)
    hallucination_resistance: int = Field(ge=1, le=5)
    tool_usage_appropriateness: int = Field(ge=1, le=5)
    overall_score: float = Field(description="Weighted average; always recomputed by model_validator.")
    deterministic_checks: DeterministicChecks
    improvement_suggestion: str
    judge_reasoning: str
    judge_succeeded: bool = Field(
        default=True,
        description="False when the LLM judge failed to produce structured output and fallback scores were used.",
    )

    @model_validator(mode="after")
    def recompute_overall_score(self) -> "EvaluationResult":
        """Compute overall_score deterministically so LLM arithmetic errors don't propagate."""
        self.overall_score = round(
            self.intent_accuracy * 0.20
            + self.context_handling * 0.25
            + self.response_quality * 0.20
            + self.hallucination_resistance * 0.20
            + self.tool_usage_appropriateness * 0.15,
            2,
        )
        return self
