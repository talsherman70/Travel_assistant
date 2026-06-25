"""
Fixed evaluation scenarios for the Travel Assistant evaluator.

22 scenarios covering 10 categories.
These are the inputs used in the evaluator's fixed test set.
"""

from models import EvaluationTest

EVALUATION_SCENARIOS: list[EvaluationTest] = [

    # ── 1. Rephrasing consistency (same intent, different wording) ─────────────
    EvaluationTest(
        test_id="rephrase_01",
        category="rephrasing_consistency",
        conversation=[
            {"role": "user", "content": "Where should I go in Europe in October?"},
        ],
        expected_intent="destination_recommendation",
        requires_tool="none",
        should_clarify=False,
        must_not_claim_live_facts=False,
    ),
    EvaluationTest(
        test_id="rephrase_02",
        category="rephrasing_consistency",
        conversation=[
            {"role": "user", "content": "What European destination do you recommend for an October trip?"},
        ],
        expected_intent="destination_recommendation",
        requires_tool="none",
        should_clarify=False,
        must_not_claim_live_facts=False,
    ),

    # ── 2. Follow-up context (prior turn info must carry forward) ──────────────
    EvaluationTest(
        test_id="followup_01",
        category="followup_context",
        conversation=[
            {"role": "user", "content": "I'm planning a 4-day trip to Rome in July with my girlfriend."},
            {"role": "user", "content": "What should I pack?"},
        ],
        expected_intent="packing_advice",
        requires_tool="none",
        should_clarify=False,
        must_not_claim_live_facts=False,
    ),
    EvaluationTest(
        test_id="followup_02",
        category="followup_context",
        conversation=[
            {"role": "user", "content": "We're going to Tokyo for 5 days, moderate budget, couple, interested in food and anime."},
            {"role": "user", "content": "Plan our itinerary."},
        ],
        expected_intent="itinerary_planning",
        requires_tool="any",
        should_clarify=False,
        must_not_claim_live_facts=False,
    ),

    # ── 3. Weather tool usage (should trigger live forecast) ───────────────────
    EvaluationTest(
        test_id="weather_01",
        category="weather_tool_usage",
        conversation=[
            {"role": "user", "content": "Will I need a jacket in London next week?"},
        ],
        expected_intent="weather_advice",
        requires_tool="weather",
        should_clarify=False,
        must_not_claim_live_facts=False,
    ),
    EvaluationTest(
        test_id="weather_02",
        category="weather_tool_usage",
        conversation=[
            {"role": "user", "content": "I'm going to Paris this weekend. What's the weather going to be like?"},
        ],
        expected_intent="weather_advice",
        requires_tool="weather",
        should_clarify=False,
        must_not_claim_live_facts=False,
    ),

    # ── 4. Attractions tool usage (should trigger places API) ─────────────────
    EvaluationTest(
        test_id="attractions_01",
        category="attractions_tool_usage",
        conversation=[
            {"role": "user", "content": "What should I do in Barcelona?"},
        ],
        expected_intent="local_attractions",
        requires_tool="attractions",
        should_clarify=False,
        must_not_claim_live_facts=False,
    ),
    EvaluationTest(
        test_id="attractions_02",
        category="attractions_tool_usage",
        conversation=[
            {"role": "user", "content": "Show me the top attractions in Amsterdam."},
        ],
        expected_intent="local_attractions",
        requires_tool="attractions",
        should_clarify=False,
        must_not_claim_live_facts=False,
    ),

    # ── 5. LLM knowledge only (should NOT call tools) ─────────────────────────
    EvaluationTest(
        test_id="knowledge_01",
        category="llm_knowledge_only",
        conversation=[
            {"role": "user", "content": "Is Rome good for a first-time Italy trip?"},
        ],
        expected_intent="general_travel_qa",
        requires_tool="none",
        should_clarify=False,
        must_not_claim_live_facts=True,
    ),
    EvaluationTest(
        test_id="knowledge_02",
        category="llm_knowledge_only",
        conversation=[
            {"role": "user", "content": "What's the best season to visit Japan?"},
        ],
        expected_intent="general_travel_qa",
        requires_tool="none",
        should_clarify=False,
        must_not_claim_live_facts=True,
    ),

    # ── 6. Ambiguous request (should clarify, not invent) ─────────────────────
    EvaluationTest(
        test_id="ambiguous_01",
        category="ambiguous_request",
        conversation=[
            {"role": "user", "content": "Plan me a trip."},
        ],
        expected_intent="clarification_needed",
        requires_tool="none",
        should_clarify=True,
        must_not_claim_live_facts=False,
    ),
    EvaluationTest(
        test_id="ambiguous_02",
        category="ambiguous_request",
        conversation=[
            {"role": "user", "content": "I want to go somewhere nice."},
        ],
        expected_intent="clarification_needed",
        requires_tool="none",
        should_clarify=True,
        must_not_claim_live_facts=False,
    ),

    # ── 7. Hallucination resistance (no live facts without tool) ──────────────
    EvaluationTest(
        test_id="hallucination_01",
        category="hallucination_resistance",
        conversation=[
            {"role": "user", "content": "What time does the Colosseum open and how much does it cost?"},
        ],
        expected_intent="general_travel_qa",
        requires_tool="none",
        should_clarify=False,
        must_not_claim_live_facts=True,
    ),
    EvaluationTest(
        test_id="hallucination_02",
        category="hallucination_resistance",
        conversation=[
            {"role": "user", "content": "What are the current entry fees for the Louvre?"},
        ],
        expected_intent="general_travel_qa",
        requires_tool="none",
        should_clarify=False,
        must_not_claim_live_facts=True,
    ),

    # ── 8. Context change (destination update must propagate) ─────────────────
    EvaluationTest(
        test_id="context_change_01",
        category="context_change",
        conversation=[
            {"role": "user", "content": "I'm planning 3 days in Rome next month."},
            {"role": "user", "content": "Actually, make it Barcelona instead of Rome."},
        ],
        expected_intent="context_update",
        requires_tool="none",
        should_clarify=False,
        must_not_claim_live_facts=False,
    ),
    EvaluationTest(
        test_id="context_change_02",
        category="context_change",
        conversation=[
            {"role": "user", "content": "We're thinking 5 days in Tokyo, just the two of us."},
            {"role": "user", "content": "Change it to 7 days actually."},
        ],
        expected_intent="context_update",
        requires_tool="none",
        should_clarify=False,
        must_not_claim_live_facts=False,
    ),

    # ── 9. Response discipline ─────────────────────────────────────────────────

    # No unsolicited suggestions: a conversational acknowledgement should get a
    # brief reply, NOT a list of recommendations or activity ideas.
    EvaluationTest(
        test_id="response_discipline_01",
        category="response_discipline",
        conversation=[
            {"role": "user", "content": "I'm planning a 4-day trip to Rome in July with my partner."},
            {"role": "user", "content": "Sounds good, thanks!"},
        ],
        expected_intent="general_travel_qa",
        requires_tool="none",
        should_clarify=False,
        must_not_claim_live_facts=False,
    ),

    # Confirmation before applying a refinement: "make it more romantic" must
    # produce a confirmation question summarising the planned change, NOT the
    # updated itinerary itself. The change should only be applied after the user
    # confirms.
    EvaluationTest(
        test_id="response_discipline_02",
        category="response_discipline",
        conversation=[
            {"role": "user", "content": "I'm planning a 4-day trip to London in July with my girlfriend. Plan the itinerary."},
            {"role": "user", "content": "Make it more romantic and less touristy."},
        ],
        expected_intent="trip_refinement",
        requires_tool="none",
        should_clarify=True,
        must_not_claim_live_facts=False,
    ),

    # Refinement applied after user confirms: after the assistant asked for
    # confirmation and the user says "yes", the full updated itinerary must be
    # produced.
    EvaluationTest(
        test_id="response_discipline_03",
        category="response_discipline",
        conversation=[
            {"role": "user", "content": "I'm planning a 4-day trip to London in July with my girlfriend. Plan the itinerary."},
            {"role": "user", "content": "Make it more romantic and less touristy."},
            {"role": "user", "content": "Yes, go ahead."},
        ],
        expected_intent="itinerary_planning",
        requires_tool="any",
        should_clarify=False,
        must_not_claim_live_facts=False,
    ),

    # ── 10. Incremental info gathering ─────────────────────────────────────────
    # Tests the realistic pattern where the user dribbles info across turns.
    # All three scenarios were absent from the original test set, which is why
    # the 37% itinerary fabrication rate on bare destinations went undetected.

    # Bare destination only — assistant must ask for trip length, must NOT fabricate a plan.
    EvaluationTest(
        test_id="incremental_01",
        category="incremental_info_gathering",
        conversation=[
            {"role": "user", "content": "I would like to travel to London in early November."},
        ],
        expected_intent="destination_recommendation",
        requires_tool="none",
        should_clarify=True,
        must_not_claim_live_facts=False,
    ),

    # User answers the follow-up question — assistant must go straight to the plan
    # without another confirmation step.
    EvaluationTest(
        test_id="incremental_02",
        category="incremental_info_gathering",
        conversation=[
            {"role": "user", "content": "I would like to travel to London in early November."},
            {"role": "user", "content": "7 days, I love food and history."},
        ],
        expected_intent="itinerary_planning",
        requires_tool="none",  # assistant should offer to draft, not generate a plan yet
        should_clarify=False,
        must_not_claim_live_facts=False,
    ),

    # Full three-turn flow — final plan must contain exactly 7 days (day_count_matches_duration check).
    EvaluationTest(
        test_id="incremental_03",
        category="incremental_info_gathering",
        conversation=[
            {"role": "user", "content": "I would like to travel to London in early November."},
            {"role": "user", "content": "7 days, I love food and history."},
            {"role": "user", "content": "Yes, go ahead and make the plan."},
        ],
        expected_intent="itinerary_planning",
        requires_tool="any",
        should_clarify=False,
        must_not_claim_live_facts=False,
    ),
]
