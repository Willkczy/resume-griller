"""
Interview Graph — LangGraph-based interview orchestration.

This module replaces the duplicated interview logic that was previously
spread across interview_agent.py, session.py, and websocket.py.

Route handlers call graph.commands.execute_command with the compiled graph,
session_id, action and optional answer. That boundary validates the latest checkpoint
and injects GraphServices via config. It does not own a separate session store.
Only graph nodes advance interview state. See decisions/0002-interview-commands-and-turns.md.
"""

from backend.app.graph.checkpointer import get_compiled_graph
from backend.app.graph.services import GraphServices
from backend.app.graph.state import InterviewState, create_initial_state

__all__ = [
    "InterviewState",
    "create_initial_state",
    "GraphServices",
    "get_compiled_graph",
]
