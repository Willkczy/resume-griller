"""
Graph Nodes — the functions that do actual work in the interview graph.

=== LangGraph Concept: Nodes ===

A node is a plain Python function (sync or async) with this signature:
    async def my_node(state: InterviewState, config: RunnableConfig) -> dict:

Rules:
1. Receives the FULL current state + config (which has our services)
2. Returns a PARTIAL dict of state fields to update
3. Should be focused — do ONE thing (evaluate, generate, advance, etc.)
4. Delegates heavy lifting to existing services (GrillingEngine, LLM, RAG)
   Nodes are thin wrappers, not reimplementations.

The `config` parameter gives access to injected services:
    services = config["configurable"]["services"]  # -> GraphServices instance

Each node in this file maps to a step in the interview flow:

  generate_questions  → Uses RAG + LLM to create interview questions from resume
  ask_question        → Formats current question as output, retrieves RAG context
  evaluate_answer     → Runs GrillingEngine.evaluate_answer() + consistency check
  generate_follow_up  → Runs GrillingEngine.generate_follow_up()
  advance_question    → Moves to next question, resets follow-up counter
  complete_interview  → Generates summary, marks interview done
  handle_skip         → Records skip, delegates to advance_question logic
  handle_end          → Marks interview cancelled
  handle_error        → Captures error state
"""

from __future__ import annotations

import re
import time
from typing import Any

from langchain_core.runnables import RunnableConfig

from backend.app.graph.state import InterviewState
from backend.app.graph.services import GraphServices
from backend.app.core.grilling_engine import GapType


# === Helper: extract services from config ===

def _get_services(config: RunnableConfig) -> GraphServices:
    """
    Pull our GraphServices out of the LangGraph config.

    Every route handler passes services via:
      graph.ainvoke(state, config={"configurable": {"services": svc}})

    This helper avoids repeating the dict lookup in every node.
    """
    return config["configurable"]["services"]


# === Helper: build a conversation message dict ===

def _msg(role: str, content: str, is_follow_up: bool = False, **metadata) -> dict:
    """Create a message dict for the conversation list."""
    from datetime import datetime
    return {
        "role": role,
        "content": content,
        "is_follow_up": is_follow_up,
        "timestamp": datetime.utcnow().isoformat(),
        "metadata": metadata,
    }


# ─────────────────────────────────────────────
# Node: generate_questions
# ─────────────────────────────────────────────
# Source: InterviewAgent._generate_questions() + _parse_questions()
# When:   action="start"
# Does:   RAG prompt + LLM → parse → store questions in state

async def generate_questions(
    state: InterviewState, config: RunnableConfig
) -> dict:
    """Generate interview questions from resume using RAG + LLM."""
    services = _get_services(config)

    resume_id = state["resume_id"]
    mode = state["mode"]
    model_type = state["model_type"]
    num_questions = state["num_questions"]
    focus_areas = state.get("focus_areas", [])

    # For custom/hybrid mode, questions may already be pre-generated
    # during preprocessing (Groq prepared them). Check prepared_context.
    prepared = state.get("prepared_context")
    if model_type == "custom" and prepared and prepared.get("questions"):
        questions = prepared["questions"]
        return {
            "questions": questions[:num_questions],
            "status": "asking",
        }

    # API mode: generate questions via RAG + LLM
    try:
        focus_area = focus_areas[0] if focus_areas else None

        # Step 1: Build RAG prompt (retrieves relevant resume chunks)
        prompt = services.retriever.build_prompt(
            resume_id=resume_id,
            focus_area=focus_area,
            question_type=mode,
            n_questions=num_questions,
        )

        # Step 2: Create mode-specific system prompt
        system_prompt = _build_question_generation_prompt(mode, num_questions)

        # Step 3: Call LLM to generate questions
        # Add diversity note to prevent identical questions across sessions
        diversity_note = f"\n\n[Generation ID: {int(time.time() * 1000) % 10000}]"

        response = await services.llm.generate(
            prompt=prompt + diversity_note,
            system_prompt=system_prompt,
            temperature=0.85,
            max_tokens=2000,
        )

        # Step 4: Parse the LLM response into individual questions
        questions = _parse_questions(response, num_questions)

        # Step 5: Fill with defaults if not enough questions generated
        if len(questions) < num_questions:
            defaults = _get_default_questions(mode)
            while len(questions) < num_questions and defaults:
                questions.append(defaults.pop(0))

        return {
            "questions": questions[:num_questions],
            "status": "asking",
        }

    except Exception as e:
        # Fallback to default questions on any error
        return {
            "questions": _get_default_questions(mode)[:num_questions],
            "status": "asking",
            "error": f"Question generation fell back to defaults: {e}",
        }


# ─────────────────────────────────────────────
# Node: ask_question
# ─────────────────────────────────────────────
# Source: session message-adding logic
# When:   After generate_questions, or after advance_question
# Does:   Formats current question as output, retrieves RAG context

async def ask_question(
    state: InterviewState, config: RunnableConfig
) -> dict:
    """Prepare the current question for the candidate."""
    services = _get_services(config)

    idx = state["current_question_index"]
    questions = state["questions"]

    if idx >= len(questions):
        # No more questions — shouldn't happen, but handle gracefully
        return {
            "response_type": "complete",
            "response_content": "Interview complete! Thank you for your responses.",
            "response_data": {},
            "status": "completed",
        }

    question = questions[idx]

    # Retrieve relevant resume context for this question (for future evaluation)
    resume_context = ""
    try:
        chunks = services.retriever.retrieve(
            resume_id=state["resume_id"],
            focus_area=question,
            n_chunks=3,
        )
        resume_context = "\n\n".join(
            c.get("content", "") for c in chunks if c.get("content")
        )
    except Exception:
        pass  # Non-critical — evaluation can work without context

    # Build the interviewer message
    msg = _msg("interviewer", question, question_number=idx + 1)

    return {
        "resume_context": resume_context,
        "status": "asking",
        # Clear per-interaction fields from previous cycle
        "current_answer": None,
        "current_evaluation": None,
        # Append question to conversation
        "conversation": [msg],
        # Output for the route handler
        "response_type": "question",
        "response_content": question,
        "response_data": {
            "question_number": idx + 1,
            "total_questions": len(questions),
        },
    }


# ─────────────────────────────────────────────
# Node: evaluate_answer
# ─────────────────────────────────────────────
# Source: InterviewAgent.process_answer() core logic
# When:   action="answer"
# Does:   GrillingEngine.evaluate_answer() + optional consistency check

async def evaluate_answer(
    state: InterviewState, config: RunnableConfig
) -> dict:
    """Evaluate the candidate's answer using the GrillingEngine."""
    services = _get_services(config)

    answer = state["current_answer"]
    question = state["questions"][state["current_question_index"]]
    resume_context = state.get("resume_context", "")
    follow_up_count = state["current_follow_up_count"]
    mode = state["mode"]

    # Record the candidate's answer in conversation
    answer_msg = _msg("candidate", answer)

    # Build conversation history (last 10 messages for context)
    existing_convo = state.get("conversation", [])
    recent = existing_convo[-10:] if len(existing_convo) > 10 else existing_convo
    conversation_history = [
        {
            "role": m["role"],
            "content": m["content"],
            "is_follow_up": m.get("is_follow_up", False),
        }
        for m in recent
    ]

    # Call GrillingEngine to evaluate the answer
    # This is the core grilling logic — detects gaps across 18 types,
    # scores across 7 dimensions, and suggests follow-ups.
    evaluation = await services.grilling_engine.evaluate_answer(
        question=question,
        answer=answer,
        resume_context=resume_context,
        question_type=mode,
        conversation_history=conversation_history,
        follow_up_count=follow_up_count,
        question_index=state["current_question_index"],
    )

    # Resume consistency check (API mode, first answer to a question only)
    # Catches candidates claiming work that doesn't match their resume
    if (
        resume_context
        and follow_up_count == 0
        and state["model_type"] != "custom"
    ):
        try:
            is_consistent, inconsistencies = (
                await services.grilling_engine.check_resume_consistency(
                    answer=answer,
                    resume_context=resume_context,
                    question=question,
                )
            )
            if not is_consistent and inconsistencies:
                evaluation.gap_analysis.detected_gaps.append(
                    GapType.RESUME_INCONSISTENT
                )
                evaluation.gap_analysis.gap_details[
                    GapType.RESUME_INCONSISTENT
                ] = inconsistencies[0]
                evaluation.is_sufficient = False
                evaluation.suggested_follow_up = (
                    f"I'd like to clarify something. {inconsistencies[0]} "
                    "Could you help me understand this better?"
                )
        except Exception:
            pass  # Non-critical — don't fail the evaluation

    return {
        "current_evaluation": evaluation.to_dict(),
        "status": "evaluating",
        "conversation": [answer_msg],
    }


# ─────────────────────────────────────────────
# Node: generate_follow_up
# ─────────────────────────────────────────────
# Source: GrillingEngine.generate_follow_up() call in InterviewAgent
# When:   route_after_evaluate returns "grill"
# Does:   Generates a targeted follow-up question based on detected gaps

async def generate_follow_up(
    state: InterviewState, config: RunnableConfig
) -> dict:
    """Generate a follow-up question targeting the priority gap."""
    services = _get_services(config)

    question = state["questions"][state["current_question_index"]]
    answer = state["current_answer"]
    mode = state["mode"]

    # Reconstruct AnswerEvaluation from the dict stored in state
    # GrillingEngine.generate_follow_up needs the evaluation object
    from backend.app.core.grilling_engine import AnswerEvaluation
    evaluation = AnswerEvaluation.from_dict(state["current_evaluation"])

    # Build conversation history
    existing_convo = state.get("conversation", [])
    recent = existing_convo[-10:] if len(existing_convo) > 10 else existing_convo
    conversation_history = [
        {
            "role": m["role"],
            "content": m["content"],
            "is_follow_up": m.get("is_follow_up", False),
        }
        for m in recent
    ]

    # Generate follow-up question via GrillingEngine
    follow_up = await services.grilling_engine.generate_follow_up(
        question=question,
        answer=answer,
        evaluation=evaluation,
        conversation_history=conversation_history,
        question_type=mode,
    )

    new_count = state["current_follow_up_count"] + 1

    # Build follow-up message with gap metadata
    priority_gap = (
        evaluation.gap_analysis.priority_gap.value
        if evaluation.gap_analysis.priority_gap
        else None
    )
    detected_gaps = [g.value for g in evaluation.gap_analysis.detected_gaps]

    fu_msg = _msg(
        "interviewer",
        follow_up,
        is_follow_up=True,
        follow_up_number=new_count,
        priority_gap=priority_gap,
        detected_gaps=detected_gaps,
    )

    return {
        "current_follow_up_count": new_count,
        "status": "asking",
        "conversation": [fu_msg],
        # Output for route handler
        "response_type": "follow_up",
        "response_content": follow_up,
        "response_data": {
            "question_number": state["current_question_index"] + 1,
            "total_questions": len(state["questions"]),
            "evaluation": state["current_evaluation"],
            "follow_up_count": new_count,
            "priority_gap": priority_gap,
        },
    }


# ─────────────────────────────────────────────
# Node: advance_question
# ─────────────────────────────────────────────
# Source: session.next_question() logic
# When:   After evaluate (no more grilling) or after skip
# Does:   Increments question index, resets follow-up count

async def advance_question(
    state: InterviewState, config: RunnableConfig
) -> dict:
    """Move to the next question."""
    return {
        "current_question_index": state["current_question_index"] + 1,
        "current_follow_up_count": 0,
        "current_answer": None,
        "current_evaluation": None,
    }


# ─────────────────────────────────────────────
# Node: complete_interview
# ─────────────────────────────────────────────
# Source: InterviewAgent.get_interview_summary()
# When:   All questions done, or user ends early
# Does:   Generates summary, sets completed status

async def complete_interview(
    state: InterviewState, config: RunnableConfig
) -> dict:
    """Mark interview as complete and generate summary."""
    conversation = state.get("conversation", [])

    # Count answers and follow-ups from conversation
    candidate_msgs = [
        m for m in conversation
        if m["role"] == "candidate" and m["content"] != "[Skipped]"
    ]
    follow_ups = [m for m in conversation if m.get("is_follow_up")]

    # Collect gap statistics
    gap_freq: dict[str, int] = {}
    for m in conversation:
        for gap in m.get("metadata", {}).get("detected_gaps", []):
            gap_freq[gap] = gap_freq.get(gap, 0) + 1

    questions_asked = state["current_question_index"]

    summary = {
        "session_id": state["session_id"],
        "resume_id": state["resume_id"],
        "mode": state["mode"],
        "model_type": state["model_type"],
        "status": "completed",
        "questions_asked": questions_asked,
        "total_questions": len(state["questions"]),
        "answers_given": len(candidate_msgs),
        "follow_ups_asked": len(follow_ups),
        "conversation_length": len(conversation),
        "gap_statistics": gap_freq,
        "grilling_intensity": len(follow_ups) / max(questions_asked, 1),
    }

    # Determine completion message based on how we got here
    is_cancelled = state.get("status") == "cancelled"
    content = (
        "Interview ended. Thank you for your time."
        if is_cancelled
        else "Excellent! That concludes our interview. Thank you for your thoughtful responses."
    )

    sys_msg = _msg("system", "Interview completed." if not is_cancelled else "Interview ended by user.")

    return {
        "status": "cancelled" if is_cancelled else "completed",
        "conversation": [sys_msg],
        "response_type": "complete",
        "response_content": content,
        "response_data": {"summary": summary},
    }


# ─────────────────────────────────────────────
# Node: handle_skip
# ─────────────────────────────────────────────
# Source: InterviewAgent.skip_question()
# When:   action="skip"
# Does:   Records "[Skipped]" answer, then advance_question runs next

async def handle_skip(
    state: InterviewState, config: RunnableConfig
) -> dict:
    """Record that the candidate skipped this question."""
    skip_msg = _msg("candidate", "[Skipped]", skipped=True)
    return {
        "conversation": [skip_msg],
    }


# ─────────────────────────────────────────────
# Node: handle_end
# ─────────────────────────────────────────────
# Source: InterviewAgent.end_interview()
# When:   action="end"
# Does:   Marks cancelled, complete_interview runs next for summary

async def handle_end(
    state: InterviewState, config: RunnableConfig
) -> dict:
    """Mark the interview as cancelled by user."""
    return {
        "status": "cancelled",
    }


# ─────────────────────────────────────────────
# Node: handle_error
# ─────────────────────────────────────────────

async def handle_error(
    state: InterviewState, config: RunnableConfig
) -> dict:
    """Capture an error and set error response."""
    error_msg = state.get("error", "An unknown error occurred.")
    return {
        "response_type": "error",
        "response_content": error_msg,
        "response_data": {"error": error_msg},
    }


# ═══════════════════════════════════════════════
# Private helpers (moved from InterviewAgent)
# ═══════════════════════════════════════════════

def _build_question_generation_prompt(mode: str, num_questions: int) -> str:
    """Build the system prompt for question generation based on interview mode."""
    if mode == "tech":
        mode_description = "TECHNICAL"
        mode_rules = (
            "STRICT RULES FOR TECHNICAL QUESTIONS:\n"
            "DO ask about: architecture, algorithms, implementation details, "
            "technical trade-offs, debugging, optimization\n"
            "DO reference: specific technologies, frameworks, projects from the resume\n"
            "DO NOT ask: behavioral questions, STAR situations, team dynamics, soft skills"
        )
    elif mode == "hr":
        mode_description = "BEHAVIORAL/HR"
        mode_rules = (
            "STRICT RULES FOR BEHAVIORAL QUESTIONS:\n"
            'DO ask about: experiences, situations, teamwork, leadership, conflict, challenges\n'
            'DO use: "Tell me about a time...", "Describe a situation...", '
            '"Give me an example..."\n'
            "DO NOT ask: technical implementation details, code, algorithms, architecture"
        )
    else:
        mode_description = "MIXED (Technical + Behavioral)"
        tech_count = num_questions // 2
        behav_count = num_questions - tech_count
        mode_rules = (
            f"STRICT RULES FOR MIXED INTERVIEW:\n"
            f"Generate exactly {tech_count} TECHNICAL questions and "
            f"{behav_count} BEHAVIORAL questions.\n"
            "TECHNICAL: architecture, implementation, debugging, trade-offs\n"
            "BEHAVIORAL: experiences, teamwork, challenges, leadership\n"
            "Clearly separate the two types."
        )

    return (
        f"You are an expert {mode_description} interviewer conducting a rigorous mock interview.\n\n"
        f"{mode_rules}\n\n"
        f"Generate exactly {num_questions} questions that will DEEPLY PROBE the candidate's experience.\n\n"
        "CRITICAL REQUIREMENTS:\n"
        "1. Questions MUST be SPECIFIC to this candidate's resume\n"
        "2. Reference actual projects, companies, or experiences mentioned\n"
        "3. Avoid generic questions anyone could answer\n"
        "4. Each question should require detailed, specific answers\n"
        "5. Questions should be challenging but answerable from their experience\n\n"
        f"FORMAT:\n"
        f"Return ONLY the questions, one per line, numbered 1 to {num_questions}.\n"
        "No explanations, no other text, no markdown formatting."
    )


def _parse_questions(response: str, expected_count: int) -> list[str]:
    """Parse questions from LLM response text."""
    questions = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove numbering (1., 1), 1:, etc.)
        cleaned = re.sub(r"^[\d]+[.)\-:]\s*", "", line)
        cleaned = re.sub(r"^\*+\s*", "", cleaned)
        cleaned = re.sub(r"^Question\s*\d*[.:]\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()

        if cleaned and len(cleaned) > 20:
            if cleaned.endswith("?") or len(cleaned) > 40:
                if not cleaned.endswith("?"):
                    cleaned += "?"
                questions.append(cleaned)

    return questions[:expected_count]


def _get_default_questions(mode: str) -> list[str]:
    """Fallback questions when generation fails."""
    if mode == "tech":
        return [
            "Walk me through the architecture of the most complex system you've designed. What were the key technical decisions and their trade-offs?",
            "Tell me about a time you had to optimize performance in a production system. What was the bottleneck and how did you identify and fix it?",
            "Describe your approach to debugging a critical production issue. Give me a specific example with details.",
            "What's the most challenging technical problem you've solved? Walk me through your problem-solving process.",
            "Tell me about a time when your first technical approach didn't work. How did you pivot and what did you learn?",
        ]
    elif mode == "hr":
        return [
            "Tell me about a time when you had to work with a difficult team member. How did you handle the situation and what was the outcome?",
            "Describe a situation where you had to deliver a project under a tight deadline. What was your approach and how did you prioritize?",
            "Give me an example of when you had to persuade someone to see things differently. What strategies did you use?",
            "Tell me about a project that didn't go as planned. What went wrong, what did you learn, and how did you handle it?",
            "Describe a time when you received critical feedback. How did you respond and what changes did you make?",
        ]
    else:
        return [
            "Walk me through your most impactful project. What was your specific technical role and what were the measurable outcomes?",
            "Tell me about a technical decision you made that required buy-in from non-technical stakeholders. How did you approach it?",
            "Describe a time when you had to debug a critical issue under pressure. What was your technical approach and how did you manage the stress?",
            "Give me an example of when you had to learn a new technology quickly for a project. What was your learning strategy?",
            "Tell me about a time you improved a system or process. What technical changes did you make and what was the impact on the team?",
        ]
