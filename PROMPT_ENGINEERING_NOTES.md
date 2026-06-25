# Prompt Engineering Notes

Brief notes on why the prompts are structured the way they are.

---

## Architecture: two-call pipeline

Every conversation turn makes two LLM calls:

1. **Intent extraction** → returns structured `IntentExtraction` (Pydantic-validated JSON)
2. **Response generation** → returns `AssistantResponse` (natural text + metadata)

**Why two calls instead of one?**
In a single-call design, the model simultaneously classifies intent and generates a response. This causes the *reasoning* failure mode: the model skips the intent classification step and jumps directly to a generic answer. Separating the calls forces deliberate classification before generation and allows the response prompt to use the already-verified intent, destination, and tool results as explicit context.

---

## Prompt 1 — System / Persona

**Key design decisions:**

**"Lead with the answer"** — response-shape directive targeting the style failure mode. Without it, the model front-loads every response with 2–3 sentences of preamble ("That's a great question! Let me help you...") before the actual content. This directive moves the substantive answer to position 1.

**Uncertainty policy** — grounding rules targeting the factual-accuracy failure mode. LLMs hallucinate opening hours and prices from training data with high confidence. The explicit prohibition ("never state specific opening hours, real-time prices, or live availability unless retrieved from a tool") plus the required framing ("Rome in October tends to be..." vs "the current weather in Rome is...") reduces this hallucination category significantly.

**Prompt injection defense** — placed at the end of the system prompt (low-salience position). It doesn't need to be the first thing the model reads; it just needs to be present to catch the most common injection patterns.

---

## Prompt 2 — Intent extraction

**Key design decisions:**

**Separation of `trip_refinement` vs `context_update`** — Without this distinction, vague inputs like "make it Barcelona" were classified inconsistently. "trip_refinement" modifies the previous answer (style, pace, budget). "context_update" corrects a stated fact (destination, dates). Naming both explicitly in the prompt reduces confusion between them.

**"Extract ONLY what the user said in THIS turn"** — prevents the multi-turn drift failure mode where the model copies the entire prior TripContext into context_updates, causing spurious cache invalidations and state resets. The field description on `context_updates` reinforces this: "Trip facts the user explicitly stated in THIS TURN ONLY. Do not copy prior TripContext fields."

**Tool flag calibration** — `needs_weather` is explicitly scoped to "specific upcoming weather with a date anchor." Without this, the model sets `needs_weather=True` for general seasonal questions ("is October good for Rome?"), causing unnecessary API calls and slower responses. The example of what does NOT qualify is as important as what does.

**`@model_validator` for confidence-intent consistency** — The JSON schema can require `confidence: float` and `intent: str`, but it cannot enforce "if confidence < 0.6 then intent must be clarification_needed." A `@model_validator` auto-corrects after parsing — if the LLM emits low confidence with a non-clarification intent, the validator rewrites intent to `clarification_needed` and emits a question, then logs the correction to stderr (see `TestAutoCorrectObservability`). Downstream code always receives a consistent object; corrections are observable for monitoring but never raise.

---

## Prompt 3 — Response generation

**Key design decisions:**

**Blending rules** — Without explicit instructions, the model blends tool results and general knowledge without distinguishing them. A response to "what's the weather in Rome next week" with actual forecast data would say "the weather is 28°C" — indistinguishable from a hallucinated value. The blending rules force natural attribution: "The forecast shows highs around 28°C" (tool data) vs "Rome in October generally sees temperatures around 15–22°C" (general knowledge). The `used_external_data` field makes this distinction machine-readable for the evaluator.

**Tool failure acknowledgement** — If the weather tool failed, the response must briefly acknowledge it before continuing with general advice. Without this instruction, the model silently ignores the error and either gives no weather context or hallucinates one.

---

## Prompt 4 — Chain-of-Thought itinerary planning

**Why Chain-of-Thought here?**

Itinerary planning is the highest-complexity task in this system: it requires simultaneously considering geography (cluster activities to minimize travel), day structure (energy distribution across multi-day trips), user preferences (interests, pace, budget), and practical constraints (venue closing days, transit time). Without scaffolding, the model produces itineraries that ignore geographic logic (e.g. visiting venues on opposite sides of a city on the same afternoon) or forget the user's stated interests by Day 3.

CoT was chosen over ReAct because:
- **Not a tool-routing task**: both tools are already called before this prompt fires. ReAct is designed for interleaved tool calls; there's nothing to interleave here.
- **Pure multi-step reasoning**: the five steps map directly to the subtasks: constraints → geography → day structure → preferences → sanity check.

**"Do not expose reasoning steps in response_text"** — Without this, the model outputs its Step 1–5 reasoning as part of the user-visible response. The CoT is for internal use only; the user sees the final itinerary.

**Step 5 sanity check** — Addresses the hallucination sub-type where models confidently state museum opening days. "Flag uncertain facts with 'check ahead'" is a compromise: it allows the model to include its general knowledge while flagging that it cannot guarantee accuracy.

---

## Prompt 5 — Evaluator / LLM judge

**Key design decisions:**

**Scoring calibration anchor** — LLM judges default to inflated scores (3.5–4.5 range for mediocre responses). The explicit statement "a response that states a specific live fact without tool data MUST score 1 on hallucination_resistance" provides a concrete anchor that pulls scores toward realistic values.

**`improvement_suggestion` format instruction** — Without "name the prompt section or behavior," judge suggestions are generic ("be more helpful", "improve context handling"). With the instruction, suggestions become actionable: "The response generation system prompt should add an explicit instruction to..."

**`JudgeOutput` is a separate schema from `EvaluationResult`** — The judge only needs to produce 7 fields. `test_id` and `deterministic_checks` are added by the evaluator after the judge call. This keeps the LLM schema small and focused, reducing the chance of the model producing invalid JSON for the full `EvaluationResult` structure.

**`overall_score` recomputed by `@model_validator`** — The judge is asked to compute overall_score (so it "thinks" about the aggregate), but the evaluator overwrites it with the deterministic weighted formula. LLM arithmetic is unreliable; deterministic computation is not. The weights (intent_accuracy×0.20, context_handling×0.25, response_quality×0.20, hallucination_resistance×0.20, tool_usage_appropriateness×0.15) give context_handling the highest weight because multi-turn context is the hardest and most important behavior to maintain.

---

## Pydantic as the prompt contract

Every LLM output goes through a JSON schema instruction embedded in the system prompt. This has two effects:

1. **Field descriptions as prompt**: `Field(description=...)` text appears in the JSON schema the model sees. "Trip facts the user explicitly stated in THIS TURN ONLY" is not repeated in the system prompt — it lives in the schema and the model reads it there.
2. **Validators as semantic enforcement**: Cross-field rules that JSON Schema cannot express (confidence ↔ intent, clarification_needed ↔ questions) are enforced by `@model_validator` after parsing. The model cannot satisfy them at generation time; they're a hard code-level gate.
