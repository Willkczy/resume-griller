"""
Interview Graph — LangGraph-based interview orchestration.

This module replaces the duplicated interview logic that was previously
spread across interview_agent.py, session.py, and websocket.py.

Usage from route handlers:

    from backend.app.graph import get_compiled_graph, GraphServices, create_initial_state

    # 1. Create services based on model type
    services = GraphServices.create(model_type, retriever, prepared_context)

    # 2. Get the compiled graph
    graph = await get_compiled_graph()

    # 3. Invoke with action
    result = await graph.ainvoke(
        {"action": "start"},
        config={"configurable": {"thread_id": session_id, "services": services}},
    )

    # 4. Read output from result
    response_type = result["response_type"]    # "question", "follow_up", "complete", "error"
    response_content = result["response_content"]
    response_data = result["response_data"]
"""

from backend.app.graph.state import InterviewState, create_initial_state
from backend.app.graph.services import GraphServices
from backend.app.graph.checkpointer import get_compiled_graph

__all__ = [
    "InterviewState",
    "create_initial_state",
    "GraphServices",
    "get_compiled_graph",
]
