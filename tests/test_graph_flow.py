from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.memory import MemorySaver

from backend.app.core.grilling_engine import (
    AnswerEvaluation,
    DetailedScores,
    GapAnalysis,
    GapType,
)
from backend.app.graph.builder import build_interview_graph
from backend.app.graph.edges import (
    route_action,
    route_after_advance,
    route_after_evaluate,
)
from backend.app.graph.services import GraphServices
from backend.app.graph.state import (
    calculate_duration_seconds,
    calculate_questions_asked,
    create_initial_state,
    normalize_public_status,
)


def evaluation(*, sufficient: bool, with_gap: bool) -> AnswerEvaluation:
    gaps = [GapType.NO_METRICS] if with_gap else []
    return AnswerEvaluation(
        is_sufficient=sufficient,
        score=0.9 if sufficient else 0.5,
        detailed_scores=DetailedScores(
            relevancy=0.8,
            clarity=0.8,
            informativeness=0.8,
            specificity=0.8,
            quantification=0.8,
            depth=0.8,
            completeness=0.8,
        ),
        gap_analysis=GapAnalysis(
            detected_gaps=gaps,
            gap_details={GapType.NO_METRICS: "Missing metrics"} if gaps else {},
            severity=0.5 if gaps else 0.0,
            priority_gap=GapType.NO_METRICS if gaps else None,
        ),
        missing_elements=["metrics"] if gaps else [],
        strengths=["clear"],
        suggested_follow_up="What measurable result did you achieve?" if gaps else None,
        reasoning="test evaluation",
    )


@pytest.fixture
def services() -> GraphServices:
    llm = AsyncMock()
    llm.generate.return_value = (
        "1. How did you design the payment service?\n"
        "2. How did you debug the production outage?\n"
        "3. What measurable result did your migration achieve?"
    )
    grilling = AsyncMock()
    grilling.evaluate_answer.return_value = evaluation(sufficient=False, with_gap=True)
    grilling.generate_follow_up.return_value = "What measurable result did you achieve?"
    grilling.check_resume_consistency.return_value = (True, [])
    return GraphServices(llm=llm, grilling_engine=grilling)


def graph_config(services: GraphServices, thread_id: str = "test-session") -> dict:
    return {"configurable": {"thread_id": thread_id, "services": services}}


@pytest.mark.asyncio
async def test_complete_graph_flow_with_follow_up_skip_and_end(services):
    graph = build_interview_graph().compile(checkpointer=MemorySaver())
    initial = create_initial_state(
        session_id="test-session",
        resume_id="resume-1",
        full_resume_text="# CANDIDATE: Test",
        mode="tech",
        num_questions=3,
        max_follow_ups=1,
    )
    config = graph_config(services)

    result = await graph.ainvoke({**initial, "action": "start"}, config=config)
    assert result["response_type"] == "question"
    assert result["current_question_index"] == 0
    assert calculate_questions_asked(result) == 1

    result = await graph.ainvoke(
        {"action": "answer", "current_answer": "We improved it."}, config=config
    )
    assert result["response_type"] == "follow_up"
    assert result["current_follow_up_count"] == 1

    services.grilling_engine.evaluate_answer.return_value = evaluation(
        sufficient=True, with_gap=False
    )
    result = await graph.ainvoke(
        {
            "action": "answer",
            "current_answer": "I reduced latency from 400ms to 100ms.",
        },
        config=config,
    )
    assert result["response_type"] == "question"
    assert result["current_question_index"] == 1

    result = await graph.ainvoke({"action": "skip"}, config=config)
    assert result["response_type"] == "question"
    assert result["current_question_index"] == 2

    result = await graph.ainvoke({"action": "end"}, config=config)
    assert result["status"] == "cancelled"
    assert result["response_data"]["summary"]["status"] == "cancelled"
    assert result["response_data"]["summary"]["questions_asked"] == 3


@pytest.mark.asyncio
async def test_last_question_completes_without_follow_up(services):
    graph = build_interview_graph().compile(checkpointer=MemorySaver())
    initial = create_initial_state(
        session_id="one-question",
        resume_id="resume-1",
        full_resume_text="resume",
        num_questions=1,
        max_follow_ups=0,
    )
    config = graph_config(services, "one-question")
    await graph.ainvoke({**initial, "action": "start"}, config=config)
    result = await graph.ainvoke(
        {"action": "answer", "current_answer": "A complete quantified answer."},
        config=config,
    )

    assert result["status"] == "completed"
    summary = result["response_data"]["summary"]
    assert summary["questions_asked"] == summary["total_questions"] == 1
    assert summary["answers_given"] == 1


@pytest.mark.asyncio
async def test_invalid_action_returns_error(services):
    graph = build_interview_graph().compile(checkpointer=MemorySaver())
    initial = create_initial_state("bad-action", "resume-1")
    result = await graph.ainvoke(
        {**initial, "action": None, "error": "Unsupported action"},
        config=graph_config(services, "bad-action"),
    )
    assert result["response_type"] == "error"
    assert result["response_content"] == "Unsupported action"


def test_routes_and_public_state_helpers():
    assert route_action({"action": "skip"}) == "skip"
    assert route_action({}) == "error"
    assert normalize_public_status("asking") == "in_progress"
    assert normalize_public_status("evaluating") == "in_progress"
    assert normalize_public_status("unknown") == "pending"
    assert calculate_duration_seconds({}) == 0
    assert (
        route_after_advance({"current_question_index": 2, "questions": ["q"]}) == "done"
    )
    assert (
        route_after_evaluate(
            {
                "current_evaluation": {"is_sufficient": True, "gap_analysis": {}},
                "current_follow_up_count": 0,
                "max_follow_ups": 0,
                "current_question_index": 0,
                "questions": ["q1", "q2"],
            }
        )
        == "advance"
    )
