"""
LLM backend (DeepSeek), conversation state management, and LLM call wrappers.

Uses the DeepSeek API via DEEPSEEK_API_KEY (pay-per-use, no daily limit).

Four exported items used by the rest of the codebase:
  get_backend()      — returns a _DeepSeekBackend instance
  ContextManager     — owns TripContext, sliding history window, and tool cache
  IntentExtractor    — wraps the intent-extraction LLM call
  ResponseGenerator  — wraps the response-generation and CoT itinerary LLM calls
"""

from __future__ import annotations

import os
import time
import traceback
from pydantic import BaseModel, ValidationError

from models import (
    AssistantResponse,
    AttractionsResult,
    ConversationTurn,
    IntentExtraction,
    TripContext,
    WeatherResult,
)
from prompts import (
    COT_ITINERARY_ADDENDUM,
    INTENT_SYSTEM_ADDENDUM,
    RESPONSE_SYSTEM_ADDENDUM,
    SYSTEM_PROMPT,
    format_intent_user_prompt,
    format_itinerary_user_prompt,
    format_response_user_prompt,
)
from tools import ToolRouter

_WINDOW_SIZE = 10

_TOPIC_MAP: dict[str, str] = {
    "itinerary_planning": "itinerary",
    "packing_advice": "packing",
    "weather_advice": "weather",
    "local_attractions": "attractions",
    "destination_recommendation": "recommendation",
}


# ── LLM backend abstraction ───────────────────────────────────────────────────

class _LLMBackend:
    """Base class for LLM backends. Subclasses implement chat()."""

    def chat(self, messages: list[dict], schema_model: type[BaseModel]) -> str:
        """Send messages to the LLM with the given Pydantic schema as output format.
        Returns the raw JSON string from the model."""
        raise NotImplementedError



def _is_schema_echo(raw: str) -> bool:
    """Return True if the model returned the JSON schema itself instead of filled values.
    Catches two patterns:
    1. '$defs' at top level — always a schema echo.
    2. 'properties' at top level where the inner values are schema objects (dicts with
       'type'/'description'/'$ref') rather than actual output values.
       When properties contains primitive values (int, str, etc.), the model_validator
       on JudgeOutput will unwrap the envelope — no retry needed.
    3. Description-echo: top-level 'description' key alongside absent 'intent'/'response_text'.
    """
    import json as _json
    try:
        obj = _json.loads(raw)
        if not isinstance(obj, dict):
            return False
        if "$defs" in obj:
            return True
        if "properties" in obj and isinstance(obj.get("properties"), dict):
            props = obj["properties"]
            if not props:
                return True  # empty properties block
            # Schema echo: property values are schema definition objects
            if any(
                isinstance(v, dict) and ("type" in v or "description" in v or "$ref" in v)
                for v in props.values()
            ):
                return True
            # Actual values wrapped in "properties" — model_validator handles unwrapping
            return False
        # Pattern 3: model echoed the schema's own 'description' field as a top-level key
        if "description" in obj and ("intent" not in obj and "response_text" not in obj):
            return True
        return False
    except Exception:
        return False


def _resolve_schema(schema: dict) -> dict:
    """Inline all $ref/$defs in a JSON schema so no cross-references remain.
    Small models often copy $defs back verbatim rather than filling values."""
    import copy
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def _resolve(obj: object) -> object:
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].split("/")[-1]
                resolved = _resolve(copy.deepcopy(defs.get(ref_name, {})))
                extra = {k: v for k, v in obj.items() if k != "$ref"}
                if isinstance(resolved, dict):
                    resolved.update(extra)
                return resolved
            return {k: _resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve(i) for i in obj]
        return obj

    return _resolve(schema)


class _DeepSeekBackend(_LLMBackend):
    """DeepSeek backend. Uses the OpenAI-compatible DeepSeek API.
    Supports json_object response format. No daily token limits — pay per use.

    DEEPSEEK_MODEL may be a comma-separated list; falls back to the next model
    on rate-limit errors.
    """

    def __init__(self) -> None:
        import openai
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY must be set. "
                "Get a key at platform.deepseek.com → API keys."
            )
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        raw = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self._models = [m.strip() for m in raw.split(",") if m.strip()]
        self._model_idx = 0

    @property
    def _model(self) -> str:
        return self._models[self._model_idx]

    def _next_model(self) -> bool:
        if self._model_idx + 1 < len(self._models):
            self._model_idx += 1
            print(f"\n[deepseek] Rate limit hit — switching to fallback model: {self._model}")
            return True
        return False

    def _call(self, messages: list[dict]) -> str:
        import openai as _openai
        while True:
            for _attempt in range(3):
                try:
                    resp = self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        response_format={"type": "json_object"},
                    )
                    return resp.choices[0].message.content
                except _openai.RateLimitError as e:
                    if not self._next_model():
                        raise
                    break
                except _openai.APIError:
                    if _attempt < 2:
                        time.sleep(5)
                    else:
                        raise
            else:
                break
        raise RuntimeError("DeepSeek: all models exhausted")

    def chat(self, messages: list[dict], schema_model: type[BaseModel]) -> str:
        import json

        schema_note = (
            "\n\nOUTPUT INSTRUCTIONS: Return a single JSON object with actual filled-in values. "
            "Do NOT include '$defs', '$ref', or any schema metadata. "
            "Fill every required field.\n\n"
            f"Required JSON structure:\n{json.dumps(_resolve_schema(schema_model.model_json_schema()), indent=2)}"
        )

        patched = []
        has_system = any(m["role"] == "system" for m in messages)
        if not has_system:
            patched.append({"role": "system", "content": schema_note.strip()})
        for msg in messages:
            if msg["role"] == "system":
                patched.append({"role": "system", "content": msg["content"] + schema_note})
            else:
                patched.append(msg)

        content = self._call(patched)

        if _is_schema_echo(content):
            retry_msgs = patched + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "That response is the schema definition, not filled values. "
                        "Return a JSON object where every field contains a real value. "
                        "For example: intent should be a string like \"weather_advice\", "
                        "confidence should be a number like 0.9, needs_weather should be "
                        "true or false — NOT schema metadata like {\"type\": \"string\"}."
                    ),
                },
            ]
            content = self._call(retry_msgs)

        return content


def get_backend() -> _LLMBackend:
    """Returns a DeepSeek backend instance."""
    return _DeepSeekBackend()


# ── Context manager ───────────────────────────────────────────────────────────

class ContextManager:
    """
    Owns the mutable conversation state: TripContext, history window, tool cache.

    Merge strategy for TripContext updates:
    - Uses model_fields_set to detect which fields were explicitly set this turn.
    - Lists (interests, constraints) are appended, not overwritten.
    - If destination or dates change, tool cache is cleared.
    """

    def __init__(self) -> None:
        self.trip_context = TripContext()
        self.history: list[ConversationTurn] = []
        self._tool_cache: dict = {}
        self.router = ToolRouter(cache=self._tool_cache)

    def reset(self) -> None:
        self.trip_context = TripContext()
        self.history = []
        self._tool_cache.clear()

    def update(self, extraction: IntentExtraction, user_message: str) -> None:
        """Record the user turn and merge context_updates into TripContext."""
        self._add_turn(ConversationTurn(role="user", content=user_message))

        updates = extraction.context_updates
        if not updates.model_fields_set:
            return

        current = self.trip_context.model_dump()

        new_dest = updates.destination if "destination" in updates.model_fields_set else None
        new_start = updates.start_date if "start_date" in updates.model_fields_set else None
        new_end = updates.end_date if "end_date" in updates.model_fields_set else None

        dest_changed = new_dest is not None and new_dest != current.get("destination")
        date_changed = (
            (new_start is not None and new_start != current.get("start_date"))
            or (new_end is not None and new_end != current.get("end_date"))
        )
        if dest_changed or date_changed:
            self._tool_cache.clear()

        for field in updates.model_fields_set:
            val = getattr(updates, field)
            if isinstance(val, list):
                existing: list = current.get(field, [])
                current[field] = existing + [v for v in val if v not in existing]
            elif val is not None:
                # Never overwrite an existing value with null — the model often sets
                # fields it doesn't intend to change to null in context_updates.
                current[field] = val

        self.trip_context = TripContext(**current)

    def add_assistant_turn(self, response_text: str) -> None:
        self._add_turn(ConversationTurn(role="assistant", content=response_text))

    def set_last_topic(self, intent: str) -> None:
        topic = _TOPIC_MAP.get(intent)
        if topic and topic != self.trip_context.last_topic:
            self.trip_context = self.trip_context.model_copy(update={"last_topic": topic})

    def _add_turn(self, turn: ConversationTurn) -> None:
        self.history.append(turn)
        if len(self.history) > _WINDOW_SIZE:
            self.history = self.history[-_WINDOW_SIZE:]


# ── Intent extractor ──────────────────────────────────────────────────────────

class IntentExtractor:
    """Wraps the intent-extraction LLM call. Retries once on parse failure."""

    def __init__(self) -> None:
        self._backend = get_backend()
        self._system = SYSTEM_PROMPT + "\n\n" + INTENT_SYSTEM_ADDENDUM

    def extract(
        self,
        user_message: str,
        history: list[ConversationTurn],
        trip_context: TripContext,
    ) -> IntentExtraction:
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": format_intent_user_prompt(user_message, history, trip_context)},
        ]

        for attempt in range(2):
            try:
                raw = self._backend.chat(messages, IntentExtraction)
                if not raw or not raw.strip():
                    raise ValueError("empty response from backend")
                return IntentExtraction.model_validate_json(raw)
            except (ValidationError, Exception):
                if attempt == 0:
                    continue
                if os.getenv("DEBUG"):
                    traceback.print_exc()

        return IntentExtraction(
            intent="clarification_needed",
            confidence=0.3,
            context_updates=TripContext(),
            needs_weather=False,
            needs_attractions=False,
            clarification_questions=["Could you tell me more about what you're looking for?"],
        )


# ── Response generator ────────────────────────────────────────────────────────

class ResponseGenerator:
    """Wraps response-generation and CoT itinerary LLM calls."""

    def __init__(self) -> None:
        self._backend = get_backend()

    def generate(
        self,
        user_message: str,
        intent_extraction: IntentExtraction,
        trip_context: TripContext,
        history: list[ConversationTurn],
        weather: WeatherResult | None,
        attractions: AttractionsResult | None,
    ) -> AssistantResponse:
        intent = intent_extraction.intent

        if intent == "itinerary_planning":
            addendum = COT_ITINERARY_ADDENDUM
            user_prompt = format_itinerary_user_prompt(
                user_message, trip_context, history, weather, attractions
            )
        else:
            addendum = RESPONSE_SYSTEM_ADDENDUM
            user_prompt = format_response_user_prompt(
                user_message, intent, trip_context, history, weather, attractions
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + addendum},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(2):
            try:
                raw = self._backend.chat(messages, AssistantResponse)
                if not raw or not raw.strip():
                    raise ValueError("empty response from backend")
                return AssistantResponse.model_validate_json(raw)
            except (ValidationError, Exception):
                if attempt == 0:
                    continue
                if os.getenv("DEBUG"):
                    traceback.print_exc()

        return AssistantResponse(
            response_text=(
                "I wasn't able to generate a full response just now — "
                "could you try rephrasing or give me a bit more detail about what you're looking for?"
            ),
            used_external_data=False,
        )
