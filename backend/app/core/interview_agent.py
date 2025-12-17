"""
Interview Agent for Interview Coach.
Orchestrates the interview flow, question generation, and grilling logic.

Enhanced to work with the new Mistake-Guided Grilling Engine.
"""

from typing import Optional, List, Dict
from dataclasses import dataclass
from enum import Enum

from backend.app.db.session_store import (
    InterviewSession,
    SessionStatus,
    MessageRole,
    get_session_store,
)
from backend.app.core.grilling_engine import GrillingEngine, AnswerEvaluation, GapType
from backend.app.services.llm_service import BaseLLMService
from rag.retriever import InterviewRetriever


class ResponseType(str, Enum):
    """Type of interviewer response."""
    QUESTION = "question"           # New question
    FOLLOW_UP = "follow_up"         # Follow-up/grilling question
    FEEDBACK = "feedback"           # Feedback on answer
    COMPLETE = "complete"           # Interview complete
    ERROR = "error"                 # Error occurred


@dataclass
class InterviewerResponse:
    """Response from the interview agent."""
    type: ResponseType
    content: str
    question_number: Optional[int] = None
    total_questions: Optional[int] = None
    evaluation: Optional[Dict] = None
    metadata: Dict = None

    def to_dict(self) -> Dict:
        return {
            "type": self.type.value,
            "content": self.content,
            "question_number": self.question_number,
            "total_questions": self.total_questions,
            "evaluation": self.evaluation,
            "metadata": self.metadata or {},
        }


class InterviewAgent:
    """
    Orchestrates the interview process with enhanced grilling capabilities.
    
    Responsibilities:
    - Generate questions based on resume
    - Manage interview flow
    - Coordinate with GrillingEngine for gap detection and follow-ups
    - Track conversation state and context
    - Check resume consistency
    """
    
    DEFAULT_NUM_QUESTIONS = 5
    
    def __init__(
        self,
        llm_service: BaseLLMService,
        retriever: InterviewRetriever,
    ):
        self.llm = llm_service
        self.retriever = retriever
        self.grilling_engine = GrillingEngine(llm_service)
        self.session_store = get_session_store()
    
    async def start_interview(
        self,
        session: InterviewSession,
        num_questions: int = None,
    ) -> InterviewerResponse:
        """
        Start an interview session.
        
        Generates questions and returns the first one.
        """
        num_questions = num_questions or self.DEFAULT_NUM_QUESTIONS
        
        # Generate questions based on resume and mode
        questions = await self._generate_questions(
            resume_id=session.resume_id,
            mode=session.mode,
            focus_areas=session.focus_areas,
            num_questions=num_questions,
        )
        
        if not questions:
            return InterviewerResponse(
                type=ResponseType.ERROR,
                content="Failed to generate interview questions. Please try again.",
            )
        
        # Update session
        session.questions = questions
        session.status = SessionStatus.IN_PROGRESS
        session.current_question_index = 0
        
        # Add first question to conversation
        first_question = questions[0]
        session.add_message(
            role=MessageRole.INTERVIEWER,
            content=first_question,
            metadata={"question_number": 1},
        )
        
        self.session_store.update(session)
        
        return InterviewerResponse(
            type=ResponseType.QUESTION,
            content=first_question,
            question_number=1,
            total_questions=len(questions),
        )
    
    async def process_answer(
        self,
        session: InterviewSession,
        answer: str,
    ) -> InterviewerResponse:
        """
        Process a candidate's answer with enhanced gap detection.
        
        This is the core "grilling" logic:
        1. Build conversation context
        2. Evaluate the answer with gap detection
        3. Check resume consistency (optional)
        4. If gaps detected and not max follow-ups, ask follow-up
        5. If sufficient (or max follow-ups reached), move to next question
        """
        if session.status != SessionStatus.IN_PROGRESS:
            return InterviewerResponse(
                type=ResponseType.ERROR,
                content="Interview is not in progress.",
            )
        
        current_question = session.current_question
        if not current_question:
            return InterviewerResponse(
                type=ResponseType.COMPLETE,
                content="Interview complete! Thank you for your responses.",
            )
        
        # Add answer to conversation
        session.add_message(
            role=MessageRole.CANDIDATE,
            content=answer,
        )
        
        # Get resume context for evaluation
        resume_context = await self._get_resume_context(
            session.resume_id,
            current_question,
        )
        
        # Build conversation history for context-aware evaluation
        conversation_history = self._build_conversation_history(session)
        
        # Evaluate the answer with the enhanced grilling engine
        evaluation = await self.grilling_engine.evaluate_answer(
            question=current_question,
            answer=answer,
            resume_context=resume_context,
            question_type=session.mode,
            conversation_history=conversation_history,
            follow_up_count=session.current_follow_up_count,
        )
        
        # Optional: Check resume consistency for significant claims
        if resume_context and session.current_follow_up_count == 0:
            is_consistent, inconsistencies = await self.grilling_engine.check_resume_consistency(
                answer=answer,
                resume_context=resume_context,
                question=current_question,
            )
            
            # If inconsistent, add to gap analysis and force follow-up
            if not is_consistent and inconsistencies:
                evaluation.gap_analysis.detected_gaps.append(GapType.RESUME_INCONSISTENT)
                evaluation.gap_analysis.gap_details[GapType.RESUME_INCONSISTENT] = inconsistencies[0]
                evaluation.is_sufficient = False
                
                # Generate consistency-focused follow-up
                evaluation.suggested_follow_up = f"I'd like to clarify something. {inconsistencies[0]} Could you help me understand this better?"
        
        # Decide: follow-up or next question?
        should_grill = self.grilling_engine.should_grill(
            evaluation=evaluation,
            follow_up_count=session.current_follow_up_count,
            max_follow_ups=session.max_follow_ups,
        )
        
        if should_grill:
            # Generate and ask follow-up question
            follow_up = await self.grilling_engine.generate_follow_up(
                question=current_question,
                answer=answer,
                evaluation=evaluation,
                conversation_history=conversation_history,
                question_type=session.mode,
            )
            
            session.increment_follow_up()
            session.add_message(
                role=MessageRole.INTERVIEWER,
                content=follow_up,
                is_follow_up=True,
                metadata={
                    "follow_up_number": session.current_follow_up_count,
                    "priority_gap": evaluation.gap_analysis.priority_gap.value if evaluation.gap_analysis.priority_gap else None,
                    "detected_gaps": [g.value for g in evaluation.gap_analysis.detected_gaps],
                },
            )
            
            self.session_store.update(session)
            
            return InterviewerResponse(
                type=ResponseType.FOLLOW_UP,
                content=follow_up,
                question_number=session.current_question_index + 1,
                total_questions=len(session.questions),
                evaluation=evaluation.to_dict(),
                metadata={
                    "follow_up_count": session.current_follow_up_count,
                    "priority_gap": evaluation.gap_analysis.priority_gap.value if evaluation.gap_analysis.priority_gap else None,
                },
            )
        
        # Move to next question
        next_question = session.next_question()
        
        if next_question is None:
            # Interview complete
            session.status = SessionStatus.COMPLETED
            session.add_message(
                role=MessageRole.SYSTEM,
                content="Interview completed.",
            )
            self.session_store.update(session)
            
            return InterviewerResponse(
                type=ResponseType.COMPLETE,
                content="Excellent! That concludes our interview. Thank you for your thoughtful responses.",
                evaluation=evaluation.to_dict(),
            )
        
        # Ask next question
        session.add_message(
            role=MessageRole.INTERVIEWER,
            content=next_question,
            metadata={"question_number": session.current_question_index + 1},
        )
        
        self.session_store.update(session)
        
        return InterviewerResponse(
            type=ResponseType.QUESTION,
            content=next_question,
            question_number=session.current_question_index + 1,
            total_questions=len(session.questions),
            evaluation=evaluation.to_dict(),
        )
    
    def _build_conversation_history(self, session: InterviewSession) -> List[Dict]:
        """
        Build conversation history for context-aware evaluation.
        
        Returns recent messages in a format suitable for the grilling engine.
        """
        history = []
        
        # Get last 10 messages for context
        recent_messages = session.conversation[-10:] if len(session.conversation) > 10 else session.conversation
        
        for msg in recent_messages:
            history.append({
                "role": msg.role.value,
                "content": msg.content,
                "is_follow_up": msg.is_follow_up,
                "timestamp": msg.timestamp.isoformat(),
            })
        
        return history
    
    async def skip_question(
        self,
        session: InterviewSession,
    ) -> InterviewerResponse:
        """Skip current question and move to next."""
        session.add_message(
            role=MessageRole.CANDIDATE,
            content="[Skipped]",
            metadata={"skipped": True},
        )
        
        next_question = session.next_question()
        
        if next_question is None:
            session.status = SessionStatus.COMPLETED
            self.session_store.update(session)
            
            return InterviewerResponse(
                type=ResponseType.COMPLETE,
                content="Interview complete! Thank you for your responses.",
            )
        
        session.add_message(
            role=MessageRole.INTERVIEWER,
            content=next_question,
            metadata={"question_number": session.current_question_index + 1},
        )
        
        self.session_store.update(session)
        
        return InterviewerResponse(
            type=ResponseType.QUESTION,
            content=next_question,
            question_number=session.current_question_index + 1,
            total_questions=len(session.questions),
        )
    
    async def end_interview(
        self,
        session: InterviewSession,
        reason: str = "ended by user",
    ) -> InterviewerResponse:
        """End the interview early."""
        session.status = SessionStatus.CANCELLED
        session.add_message(
            role=MessageRole.SYSTEM,
            content=f"Interview ended: {reason}",
        )
        
        self.session_store.update(session)
        
        return InterviewerResponse(
            type=ResponseType.COMPLETE,
            content="Interview ended. Thank you for your time.",
            metadata={"reason": reason},
        )
    
    async def get_interview_summary(
        self,
        session: InterviewSession,
    ) -> Dict:
        """Generate a detailed summary of the interview."""
        questions_asked = session.current_question_index
        total_questions = len(session.questions)
        
        # Count messages by type
        candidate_messages = [
            m for m in session.conversation
            if m.role == MessageRole.CANDIDATE and "[Skipped]" not in m.content
        ]
        follow_ups = [
            m for m in session.conversation
            if m.is_follow_up
        ]
        
        # Collect gap statistics from metadata
        all_gaps = []
        for msg in session.conversation:
            if msg.metadata and "detected_gaps" in msg.metadata:
                all_gaps.extend(msg.metadata["detected_gaps"])
        
        gap_frequency = {}
        for gap in all_gaps:
            gap_frequency[gap] = gap_frequency.get(gap, 0) + 1
        
        return {
            "session_id": session.session_id,
            "resume_id": session.resume_id,
            "mode": session.mode,
            "status": session.status.value,
            "questions_asked": questions_asked,
            "total_questions": total_questions,
            "answers_given": len(candidate_messages),
            "follow_ups_asked": len(follow_ups),
            "duration_seconds": (session.updated_at - session.created_at).total_seconds(),
            "conversation_length": len(session.conversation),
            "gap_statistics": gap_frequency,
            "grilling_intensity": len(follow_ups) / max(questions_asked, 1),
        }
    
    async def _generate_questions(
        self,
        resume_id: str,
        mode: str,
        focus_areas: List[str],
        num_questions: int,
    ) -> List[str]:
        """Generate interview questions using RAG."""
        try:
            # Build prompt using retriever
            focus_area = focus_areas[0] if focus_areas else None
            
            # Map mode to question_type for retriever
            # mode can be "hr", "tech", or "mixed"
            # retriever expects "hr", "tech", or "mixed" (same values, so direct pass)
            
            prompt = self.retriever.build_prompt(
                resume_id=resume_id,
                focus_area=focus_area,
                question_type=mode,  # Direct pass - values match now
                n_questions=num_questions,
            )
            
            if mode == "tech":
                mode_description = "TECHNICAL"
                mode_rules = """STRICT RULES FOR TECHNICAL QUESTIONS:
✓ DO ask about: architecture, algorithms, implementation details, technical trade-offs, debugging, optimization
✓ DO reference: specific technologies, frameworks, projects from the resume
✗ DO NOT ask: behavioral questions, STAR situations, team dynamics, soft skills

Example good questions:
- "Walk me through the architecture of your Energy Forecasting System. Why did you choose LSTM over other models?"
- "In your GKE troubleshooting role, what specific debugging methodologies did you use to achieve 45% faster resolution?"

Example BAD questions (avoid these):
- "Tell me about a time when you worked with a difficult team member" (behavioral)
- "How do you handle stress?" (behavioral)"""

            elif mode == "hr":
                mode_description = "BEHAVIORAL/HR"
                mode_rules = """STRICT RULES FOR BEHAVIORAL QUESTIONS:
✓ DO ask about: experiences, situations, teamwork, leadership, conflict, challenges
✓ DO use: "Tell me about a time...", "Describe a situation...", "Give me an example..."
✗ DO NOT ask: technical implementation details, code, algorithms, architecture

Example good questions:
- "Tell me about a time when you had to mentor new team members. What was your approach?"
- "Describe a situation where you had to deliver bad news to a stakeholder. How did you handle it?"

Example BAD questions (avoid these):
- "How did you implement the LSTM model?" (technical)
- "What technologies did you use?" (technical)"""

            else:  # mixed
                mode_description = "MIXED (Technical + Behavioral)"
                mode_rules = f"""STRICT RULES FOR MIXED INTERVIEW:
Generate exactly {num_questions // 2} TECHNICAL questions and {num_questions - (num_questions // 2)} BEHAVIORAL questions.

TECHNICAL questions: architecture, implementation, debugging, trade-offs
BEHAVIORAL questions: experiences, teamwork, challenges, leadership

Clearly separate the two types - do NOT mix them in the same question."""

            system_prompt = f"""You are an expert {mode_description} interviewer conducting a rigorous mock interview.

{mode_rules}

Generate exactly {num_questions} questions that will DEEPLY PROBE the candidate's experience.

CRITICAL REQUIREMENTS:
1. Questions MUST be SPECIFIC to this candidate's resume
2. Reference actual projects, companies, or experiences mentioned
3. Avoid generic questions anyone could answer
4. Each question should require detailed, specific answers
5. Questions should be challenging but answerable from their experience

FORMAT:
Return ONLY the questions, one per line, numbered 1 to {num_questions}.
No explanations, no other text, no markdown formatting."""

            import time
            
            # Add slight randomization to prevent identical questions
            diversity_note = f"\n\n[Generation ID: {int(time.time() * 1000) % 10000}]"
            
            response = await self.llm.generate(
                prompt=prompt + diversity_note,
                system_prompt=system_prompt,
                temperature=0.85,  # 增加隨機性
                max_tokens=2000,
            )
            
            # Parse questions from response
            questions = self._parse_questions(response, num_questions)

            if len(questions) < num_questions:
                print(f"Warning: Only parsed {len(questions)} questions, expected {num_questions}")
                
                default_questions = self._get_default_questions(mode)
                
                while len(questions) < num_questions and default_questions:
                    questions.append(default_questions.pop(0))
        
            return questions[:num_questions]
        
        except Exception as e:
            print(f"Error generating questions: {e}")
            return self._get_default_questions(mode)[:num_questions]
    
    def _get_default_questions(self, mode: str) -> List[str]:
        """Get default questions based on interview mode."""
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
        else:  # mixed
            return [
                "Walk me through your most impactful project. What was your specific technical role and what were the measurable outcomes?",
                "Tell me about a technical decision you made that required buy-in from non-technical stakeholders. How did you approach it?",
                "Describe a time when you had to debug a critical issue under pressure. What was your technical approach and how did you manage the stress?",
                "Give me an example of when you had to learn a new technology quickly for a project. What was your learning strategy?",
                "Tell me about a time you improved a system or process. What technical changes did you make and what was the impact on the team?",
            ]

    def _parse_questions(self, response: str, expected_count: int) -> List[str]:
        """Parse questions from LLM response."""
        import re
        
        questions = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering (1., 1), 1:, etc.)
            cleaned = re.sub(r'^[\d]+[.)\-:]\s*', '', line)
            # Remove bullet points
            cleaned = re.sub(r'^\*+\s*', '', cleaned)
            # Remove "Question X:" prefix
            cleaned = re.sub(r'^Question\s*\d*[.:]\s*', '', cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.strip()
            
            # Validate it looks like a question
            if cleaned and len(cleaned) > 20:
                # Should end with ? or be long enough to be a question
                if cleaned.endswith('?') or len(cleaned) > 40:
                    if not cleaned.endswith('?'):
                        cleaned += '?'
                    questions.append(cleaned)
        
        return questions[:expected_count]
    
    async def _get_resume_context(
        self,
        resume_id: str,
        question: str,
    ) -> str:
        """Get relevant resume context for a question."""
        try:
            chunks = self.retriever.retrieve(
                resume_id=resume_id,
                focus_area=question,
                n_chunks=3,
            )
            
            context_parts = []
            for chunk in chunks:
                content = chunk.get("content", "")
                if content:
                    context_parts.append(content)
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            print(f"Error getting resume context: {e}")
            return ""