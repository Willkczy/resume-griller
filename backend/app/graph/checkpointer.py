"""
Checkpointer — persists graph state to SQLite.

=== LangGraph Concept: Checkpointing ===

Without a checkpointer, graph state is lost after each .ainvoke() call.
With one, LangGraph saves the full state after every node execution.

This enables:
1. Resume across invocations: Each API call runs the graph once, then exits.
   The next call restores state from checkpoint and continues.
2. Survive restarts: State is in SQLite, not memory. Restarting the server
   doesn't lose interview progress.
3. Time travel: You can replay or inspect any past state.

How it works:
- When you compile with a checkpointer: graph.compile(checkpointer=saver)
- You pass a thread_id in config: {"configurable": {"thread_id": session_id}}
- LangGraph uses thread_id as the key to save/load state

We use session_id as thread_id — each interview session has its own checkpoint.

=== AsyncSqliteSaver ===

LangGraph provides several checkpointer backends:
- MemorySaver     — in-memory (lost on restart, good for tests)
- AsyncSqliteSaver — SQLite file (persists, good for single-server)
- PostgresSaver    — PostgreSQL (for production, multi-server)

We use AsyncSqliteSaver since this app runs on a single server.
The DB file is stored at data/interview_checkpoints.db.
"""

from contextlib import asynccontextmanager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend.app.graph.builder import build_interview_graph


# Path for the SQLite checkpoint database.
# Stored alongside other data files. Each session_id gets its own "thread".
CHECKPOINT_DB_PATH = "data/interview_checkpoints.db"

# Singleton compiled graph — built once, reused for all requests.
_compiled_graph = None

# We hold a reference to the context manager so it stays alive
# (and doesn't close the DB connection) for the lifetime of the process.
_checkpointer_cm = None
_checkpointer: AsyncSqliteSaver | None = None


async def get_checkpointer() -> AsyncSqliteSaver:
    """
    Get or create the SQLite checkpointer.

    AsyncSqliteSaver.from_conn_string() is an async context manager that
    creates the aiosqlite connection internally. We enter the context manager
    once and hold it open for the process lifetime.
    """
    global _checkpointer, _checkpointer_cm
    if _checkpointer is None:
        _checkpointer_cm = AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB_PATH)
        _checkpointer = await _checkpointer_cm.__aenter__()
    return _checkpointer


async def get_compiled_graph():
    """
    Get the compiled interview graph with checkpointing.

    This is the main entry point for route handlers:
        graph = await get_compiled_graph()
        result = await graph.ainvoke(
            {"action": "start", ...},
            config={
                "configurable": {
                    "thread_id": session_id,     # checkpoint key
                    "services": services,         # injected dependencies
                }
            }
        )
    """
    global _compiled_graph
    if _compiled_graph is None:
        checkpointer = await get_checkpointer()
        graph_builder = build_interview_graph()
        _compiled_graph = graph_builder.compile(checkpointer=checkpointer)
    return _compiled_graph
