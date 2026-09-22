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

from backend.app.core.follow_up_policy import should_follow_up
from backend.app.graph.state import InterviewState
from backend.app.graph.transitions import command_error


def route_action(state: InterviewState) -> str:
    """
    Entry-point router: what should the graph do this invocation?

    Called at START of every graph run. The route handler sets state["action"]
    before invoking the graph, so this just reads it.

    Returns one of: "start", "answer", "skip", "end", "error"
    """
    action = state.get("action")
    if command_error(state, action, state.get("current_answer")):
        return "error"
    if action == "start" and state.get("status") == "asking":
        return "resume"
    return action


def route_after_evaluate(state: InterviewState) -> str:
    """Use the shared policy, then advance or finish the final main question."""
    evaluation = state.get("current_evaluation") or {}
    if evaluation.get("outcome") != "valid":
        return "unavailable"
    if should_follow_up(
        state["current_evaluation"],
        state["current_follow_up_count"],
        state["max_follow_ups"],
    ):
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


def route_after_generation(state: InterviewState) -> str:
    """Never present default questions or complete an empty failed session."""
    return "ready" if state.get("status") == "asking" else "unavailable"
