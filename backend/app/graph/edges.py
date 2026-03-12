"""
Graph Edges — routing functions that decide which node runs next.

=== LangGraph Concept: Conditional Edges ===

In a StateGraph, edges connect nodes. There are two kinds:

1. Normal edge:      graph.add_edge("node_a", "node_b")
   Always goes from A to B. No decision needed.

2. Conditional edge:  graph.add_conditional_edges("node_a", my_router_fn, {...})
   Calls my_router_fn(state) which returns a STRING key.
   That key is looked up in a mapping to decide the next node.

   Example:
     graph.add_conditional_edges(
         "evaluate_answer",             # after this node...
         route_after_evaluate,          # call this function...
         {                              # and map its return value:
             "grill": "generate_follow_up",
             "advance": "advance_question",
             "done": "complete_interview",
         }
     )

The router functions below are pure decision logic — they read state and return a string.
They never modify state or call external services.
"""

from __future__ import annotations

from backend.app.graph.state import InterviewState


def route_action(state: InterviewState) -> str:
    """
    Entry-point router: what should the graph do this invocation?

    Called at START of every graph run. The route handler sets state["action"]
    before invoking the graph, so this just reads it.

    Returns one of: "start", "answer", "skip", "end", "error"
    """
    action = state.get("action")
    if action in ("start", "answer", "skip", "end"):
        return action
    return "error"


def route_after_evaluate(state: InterviewState) -> str:
    """
    After evaluating an answer: should we grill, advance, or finish?

    This is the core decision point of the interview:
    - "grill"   → ask a follow-up question (gap detected, haven't hit max)
    - "advance" → move to next question (answer sufficient or max follow-ups hit)
    - "done"    → interview complete (was the last question AND no more grilling)

    The logic mirrors GrillingEngine.should_grill() but reads from state.
    We re-implement the decision here rather than calling should_grill()
    because the evaluation result is already in state as a dict, and
    this keeps edge functions pure (no service calls).
    """
    evaluation = state.get("current_evaluation", {})
    follow_up_count = state["current_follow_up_count"]
    max_follow_ups = state["max_follow_ups"]

    # Check if we should ask a follow-up
    # Mirrors GrillingEngine.should_grill():
    #   - Max follow-ups reached → no
    #   - First answer (count=0) → always yes (forced first follow-up)
    #   - Gaps detected → yes
    #   - Answer sufficient → no
    if follow_up_count < max_follow_ups:
        is_sufficient = evaluation.get("is_sufficient", True)
        gaps = evaluation.get("gap_analysis", {}).get("detected_gaps", [])

        # Forced first follow-up: always dig deeper on the first answer
        if follow_up_count == 0:
            return "grill"

        # If gaps detected, keep grilling
        if gaps and not is_sufficient:
            return "grill"

    # No more grilling — check if this was the last question
    next_idx = state["current_question_index"] + 1
    if next_idx >= len(state["questions"]):
        return "done"

    return "advance"


def route_after_advance(state: InterviewState) -> str:
    """
    After advancing to the next question: more questions or done?

    Simple check: is the new index within bounds?
    """
    if state["current_question_index"] >= len(state["questions"]):
        return "done"
    return "more"
