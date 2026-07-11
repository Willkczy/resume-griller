"""
Graph Services — dependency injection for graph nodes.

=== LangGraph Concept: Config & Dependency Injection ===

Graph nodes need access to external services (LLM, RAG retriever, grilling engine).
But nodes are plain functions — they can't take constructor args like a class would.

LangGraph solves this with the `config` parameter:
  result = await graph.ainvoke(state, config={"configurable": {"services": my_services}})

Inside a node, you access it via:
  services = config["configurable"]["services"]

This GraphServices class bundles all the services a node might need.
It's created once per request in the route handler and passed through config.

Why not just use global singletons?
- Testability: in tests, you can inject mock services
- Model type flexibility: "api" vs "custom" needs different LLM instances
- Clean separation: nodes don't import from services/ or rag/ directly
"""

from dataclasses import dataclass

from backend.app.core.grilling_engine import GrillingEngine
from backend.app.services.llm_service import (
    BaseLLMService,
    HybridModelService,
    LLMServiceFactory,
)


@dataclass
class GraphServices:
    """
    Bundle of services that graph nodes can use.

    Created by the route handler based on model_type, then passed
    via LangGraph config so every node can access it.

    Note: retriever is NOT included — the full resume text is stored
    in InterviewState.full_resume_text, so nodes don't need the retriever.
    """

    llm: BaseLLMService
    grilling_engine: GrillingEngine
    # Only set for model_type="custom" — provides preprocessing via Groq
    hybrid_service: HybridModelService | None = None

    @classmethod
    def for_api_mode(cls) -> "GraphServices":
        """
        Create services for API mode (Groq/Gemini/OpenAI/Claude).

        Uses the default LLM provider from .env config.
        GrillingEngine uses full prompts (not compact).
        """
        llm = LLMServiceFactory.get_service()
        return cls(
            llm=llm,
            grilling_engine=GrillingEngine(llm),
        )

    @classmethod
    def for_custom_mode(
        cls,
        prepared_context: dict | None = None,
    ) -> "GraphServices":
        """
        Create services for Custom/Hybrid mode.

        - hybrid_service.interviewer (Custom Model) is used for evaluation
        - GrillingEngine gets model_type="custom" for compact prompts
        - prepared_context from Groq preprocessing is passed to GrillingEngine
        """
        hybrid = LLMServiceFactory.get_hybrid_service()
        return cls(
            llm=hybrid.interviewer,
            grilling_engine=GrillingEngine(
                llm_service=hybrid.interviewer,
                model_type="custom",
                prepared_context=prepared_context or {},
            ),
            hybrid_service=hybrid,
        )

    @classmethod
    def create(
        cls,
        model_type: str,
        prepared_context: dict | None = None,
    ) -> "GraphServices":
        """
        Factory method — picks the right mode based on model_type.

        This is the main entry point used by route handlers:
          services = GraphServices.create(model_type, prepared_context)
        """
        if model_type == "custom":
            return cls.for_custom_mode(prepared_context)
        return cls.for_api_mode()
