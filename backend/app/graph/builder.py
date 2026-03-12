"""
Graph Builder — assembles the StateGraph from nodes and edges.

=== LangGraph Concept: StateGraph ===

A StateGraph is the core abstraction in LangGraph. You build it in 3 steps:

1. Define nodes:   graph.add_node("name", function)
2. Define edges:   graph.add_edge("from", "to")             — unconditional
                   graph.add_conditional_edges("from", fn, mapping)  — conditional
3. Set entry:      graph.set_conditional_entry_point(fn, mapping)

Then compile:      compiled = graph.compile(checkpointer=...)

The compiled graph is what you .ainvoke() from route handlers.

=== Our Interview Graph Topology ===

    START
      │
      ▼
  route_action ──────────────────────────────┐
      │          │           │               │
      ▼          ▼           ▼               ▼
   "start"    "answer"     "skip"          "end"
      │          │           │               │
      ▼          ▼           ▼               ▼
  generate    evaluate    handle_skip    handle_end
  _questions  _answer        │               │
      │          │           ▼               ▼
      ▼          ▼       advance_q      complete_
  ask_question  route_      │           interview
      │        after_    route_after_       │
      ▼       evaluate    advance           ▼
     END         │           │             END
              ┌──┼──┐    ┌──┴──┐
              ▼  ▼  ▼    ▼     ▼
          grill adv done more  done
            │    │    │    │     │
            ▼    │    │    ▼     │
         gen_    │    │  ask_    │
        follow   │    │  question│
          _up    │    │    │     │
            │    ▼    │    ▼     ▼
            ▼  advance│   END  complete_
           END  _q    │         interview
                 │    │           │
                 ▼    ▼           ▼
              (same) complete_   END
                     interview
                        │
                        ▼
                       END

   "error" → handle_error → END
"""

from langgraph.graph import StateGraph, END

from backend.app.graph.state import InterviewState
from backend.app.graph import nodes, edges


def build_interview_graph() -> StateGraph:
    """
    Build the interview StateGraph (uncompiled).

    Call .compile(checkpointer=...) on the result to get a runnable graph.
    Separated from compilation so tests can compile with different checkpointers.
    """

    # Step 1: Create graph with our state schema
    graph = StateGraph(InterviewState)

    # Step 2: Register all nodes
    # Each node is an async function: (state, config) -> partial state update
    graph.add_node("generate_questions", nodes.generate_questions)
    graph.add_node("ask_question", nodes.ask_question)
    graph.add_node("evaluate_answer", nodes.evaluate_answer)
    graph.add_node("generate_follow_up", nodes.generate_follow_up)
    graph.add_node("advance_question", nodes.advance_question)
    graph.add_node("complete_interview", nodes.complete_interview)
    graph.add_node("handle_skip", nodes.handle_skip)
    graph.add_node("handle_end", nodes.handle_end)
    graph.add_node("handle_error", nodes.handle_error)

    # Step 3: Set entry point — first thing that runs on every invocation
    # This is a CONDITIONAL entry: route_action reads state["action"]
    # and routes to the appropriate starting node.
    graph.set_conditional_entry_point(
        edges.route_action,
        {
            "start": "generate_questions",
            "answer": "evaluate_answer",
            "skip": "handle_skip",
            "end": "handle_end",
            "error": "handle_error",
        },
    )

    # Step 4: Define edges (the wiring between nodes)

    # --- "start" flow ---
    # generate_questions → ask_question → END
    graph.add_edge("generate_questions", "ask_question")
    graph.add_edge("ask_question", END)

    # --- "answer" flow ---
    # evaluate_answer → route_after_evaluate → {grill, advance, done}
    graph.add_conditional_edges(
        "evaluate_answer",
        edges.route_after_evaluate,
        {
            "grill": "generate_follow_up",
            "advance": "advance_question",
            "done": "complete_interview",
        },
    )
    # generate_follow_up → END (return follow-up to client)
    graph.add_edge("generate_follow_up", END)
    # advance_question → route_after_advance → {more questions, or done}
    graph.add_conditional_edges(
        "advance_question",
        edges.route_after_advance,
        {
            "more": "ask_question",
            "done": "complete_interview",
        },
    )

    # --- "skip" flow ---
    # handle_skip → advance_question → (same routing as above)
    graph.add_edge("handle_skip", "advance_question")

    # --- "end" flow ---
    # handle_end → complete_interview → END
    graph.add_edge("handle_end", "complete_interview")
    graph.add_edge("complete_interview", END)

    # --- "error" flow ---
    graph.add_edge("handle_error", END)

    return graph
