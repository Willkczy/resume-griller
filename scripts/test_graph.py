"""
Interactive test script for the Interview Graph.

=== How to Use ===

Run from the project root:
    PYTHONPATH=. python scripts/test_graph.py

This script lets you step through the graph manually and see state at each step.
It uses mock services (no real LLM calls) so you can test the graph topology
without API keys or a running backend.

=== What You'll Learn ===

1. How StateGraph flows: START → route_action → nodes → END
2. How state updates accumulate across nodes
3. How conditional edges make routing decisions
4. How checkpointing preserves state between invocations
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

# ─── Setup ───

def create_mock_services():
    """
    Create mock services that simulate LLM and RAG without real API calls.

    This is useful for testing graph topology and state flow.
    Real integration tests would use actual services.
    """
    from backend.app.graph.services import GraphServices
    from backend.app.core.grilling_engine import (
        GrillingEngine, AnswerEvaluation, DetailedScores, GapAnalysis, GapType
    )

    # Mock LLM — returns canned responses
    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=(
        "1. Tell me about the architecture of your main project?\n"
        "2. What debugging tools did you use for production issues?\n"
        "3. How did you optimize the database queries?\n"
    ))

    # Mock retriever — returns fake resume chunks
    mock_retriever = MagicMock()
    mock_retriever.build_prompt = MagicMock(return_value="Resume context: Senior SWE at Acme Corp...")
    mock_retriever.retrieve = MagicMock(return_value=[
        {"content": "Senior Software Engineer at Acme Corp, 2020-2023"},
        {"content": "Built microservices architecture serving 1M users"},
    ])

    # Mock grilling engine — returns a realistic evaluation
    mock_grilling = AsyncMock(spec=GrillingEngine)

    # Default evaluation: answer has gaps (triggers follow-up)
    default_eval = AnswerEvaluation(
        is_sufficient=False,
        score=0.65,
        detailed_scores=DetailedScores(
            relevancy=0.8, clarity=0.7, informativeness=0.6,
            specificity=0.5, quantification=0.3, depth=0.6, completeness=0.5,
        ),
        gap_analysis=GapAnalysis(
            detected_gaps=[GapType.NO_METRICS, GapType.UNCLEAR_PERSONAL_ROLE],
            gap_details={
                GapType.NO_METRICS: "No specific numbers or metrics mentioned",
                GapType.UNCLEAR_PERSONAL_ROLE: "Used 'we' without clarifying personal contribution",
            },
            severity=0.6,
            priority_gap=GapType.NO_METRICS,
        ),
        missing_elements=["metrics", "personal role"],
        strengths=["relevant experience", "clear communication"],
        suggested_follow_up="What specific metrics improved as a result of your work?",
        reasoning="Answer lacks quantifiable results and unclear personal contribution.",
    )

    mock_grilling.evaluate_answer = AsyncMock(return_value=default_eval)
    mock_grilling.should_grill = MagicMock(return_value=True)
    mock_grilling.generate_follow_up = AsyncMock(
        return_value="You mentioned improving performance. What specific metrics did you measure, and what were the before/after numbers?"
    )
    mock_grilling.check_resume_consistency = AsyncMock(return_value=(True, []))

    return GraphServices(
        retriever=mock_retriever,
        llm=mock_llm,
        grilling_engine=mock_grilling,
    )


def print_state(state: dict, label: str = ""):
    """Pretty-print relevant state fields."""
    print(f"\n{'='*60}")
    if label:
        print(f"  {label}")
        print(f"{'='*60}")

    fields_to_show = [
        "status", "action", "current_question_index", "current_follow_up_count",
        "response_type", "response_content",
    ]
    for key in fields_to_show:
        if key in state and state[key] is not None:
            val = state[key]
            if isinstance(val, str) and len(val) > 80:
                val = val[:77] + "..."
            print(f"  {key}: {val}")

    if state.get("questions"):
        print(f"  questions: {len(state['questions'])} generated")

    convo = state.get("conversation", [])
    if convo:
        print(f"  conversation: {len(convo)} messages")
        # Show last 2 messages
        for msg in convo[-2:]:
            role = msg.get("role", "?")
            content = msg.get("content", "")[:60]
            fu = " [follow-up]" if msg.get("is_follow_up") else ""
            print(f"    [{role}]{fu}: {content}")

    print()


# ─── Main Test ───

async def main():
    """
    Walk through a complete interview flow step by step.

    This demonstrates:
    1. "start" action  → generates questions, asks first one
    2. "answer" action → evaluates answer, generates follow-up (first answer always grilled)
    3. "answer" action → evaluates follow-up answer, advances to next question
    4. "skip" action   → skips a question
    5. "end" action    → ends interview early
    """
    from langgraph.checkpoint.memory import MemorySaver
    from backend.app.graph.builder import build_interview_graph
    from backend.app.graph.state import create_initial_state

    print("\n" + "="*60)
    print("  INTERVIEW GRAPH — INTERACTIVE TEST")
    print("="*60)

    # Build graph with in-memory checkpointer (no SQLite needed for tests)
    graph = build_interview_graph().compile(checkpointer=MemorySaver())

    # Create mock services
    services = create_mock_services()

    # Config that gets passed to every invocation
    # thread_id = our session ID (checkpoint key)
    session_id = "test_session_001"
    config = {
        "configurable": {
            "thread_id": session_id,
            "services": services,
        }
    }

    # Create initial state
    initial = create_initial_state(
        session_id=session_id,
        resume_id="resume_test_123",
        mode="tech",
        model_type="api",
        num_questions=3,
        max_follow_ups=2,
    )

    # ─── Step 1: Start Interview ───
    print("\n>>> Step 1: START INTERVIEW (action='start')")
    print("    This will generate questions and return the first one.")

    result = await graph.ainvoke(
        {**initial, "action": "start"},
        config=config,
    )
    print_state(result, "After START")

    # ─── Step 2: Answer First Question ───
    print(">>> Step 2: ANSWER QUESTION (action='answer')")
    print("    Submitting an answer. First answer always gets a follow-up.")

    result = await graph.ainvoke(
        {
            "action": "answer",
            "current_answer": "We built a microservices architecture at Acme Corp. It improved performance significantly.",
        },
        config=config,
    )
    print_state(result, "After ANSWER (expect follow-up)")

    # ─── Step 3: Answer Follow-up ───
    print(">>> Step 3: ANSWER FOLLOW-UP (action='answer')")
    print("    Answering the follow-up question. Mock says no more grilling needed.")

    # Make the grilling engine return a "sufficient" evaluation this time.
    # Note: route_after_evaluate reads current_evaluation from state directly
    # (it doesn't call should_grill), so we need to change evaluate_answer's return.
    from backend.app.core.grilling_engine import (
        AnswerEvaluation, DetailedScores, GapAnalysis,
    )
    sufficient_eval = AnswerEvaluation(
        is_sufficient=True,
        score=0.90,
        detailed_scores=DetailedScores(
            relevancy=0.9, clarity=0.9, informativeness=0.9,
            specificity=0.9, quantification=0.8, depth=0.9, completeness=0.85,
        ),
        gap_analysis=GapAnalysis(
            detected_gaps=[], gap_details={}, severity=0.1, priority_gap=None,
        ),
        missing_elements=[],
        strengths=["excellent specifics", "quantified results"],
        suggested_follow_up=None,
        reasoning="Answer is thorough with metrics and clear personal contribution.",
    )
    services.grilling_engine.evaluate_answer = AsyncMock(return_value=sufficient_eval)

    result = await graph.ainvoke(
        {
            "action": "answer",
            "current_answer": "Specifically, I reduced API latency from 450ms to 120ms by implementing Redis caching. I personally designed the cache invalidation strategy.",
        },
        config=config,
    )
    print_state(result, "After FOLLOW-UP ANSWER (expect next question)")

    # ─── Step 4: Skip Question ───
    print(">>> Step 4: SKIP QUESTION (action='skip')")
    print("    Skipping the current question.")

    result = await graph.ainvoke(
        {"action": "skip"},
        config=config,
    )
    print_state(result, "After SKIP")

    # ─── Step 5: End Interview Early ───
    print(">>> Step 5: END INTERVIEW (action='end')")
    print("    Ending the interview early.")

    result = await graph.ainvoke(
        {"action": "end"},
        config=config,
    )
    print_state(result, "After END")

    # ─── Show final summary ───
    summary = result.get("response_data", {}).get("summary", {})
    if summary:
        print("─── Interview Summary ───")
        print(json.dumps(summary, indent=2))

    print("\n" + "="*60)
    print("  TEST COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
