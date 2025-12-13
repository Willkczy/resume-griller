"""
Interview Agent for Interview Coach.
Orchestrates the interview flow, question generation, and grilling logic.
"""

from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from backend.app.db.session_store import (
    InterviewSession,
    SessionStatus,
    MessageRole,
    get_session_store,
)
from backend.app.core.grilling_engine import GrillingEngine, AnswerEvaluation
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
    Orchestrates the interview process.
    
    Responsibilities:
    - Generate questions based on resume
    - Manage interview flow
    - Coordinate with GrillingEngine for follow-ups
    - Track conversation state
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
        Process a candidate's answer and decide next action.
        
        This is the core "grilling" logic:
        1. Evaluate the answer
        2. If insufficient, ask follow-up
        3. If sufficient (or max follow-ups reached), move to next question
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
        
        # Evaluate the answer
        evaluation = await self.grilling_engine.evaluate_answer(
            question=current_question,
            answer=answer,
            resume_context=resume_context,
            question_type=session.mode,
        )
        
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
            )
            
            session.increment_follow_up()
            session.add_message(
                role=MessageRole.INTERVIEWER,
                content=follow_up,
                is_follow_up=True,
                metadata={"follow_up_number": session.current_follow_up_count},
            )
            
            self.session_store.update(session)
            
            return InterviewerResponse(
                type=ResponseType.FOLLOW_UP,
                content=follow_up,
                question_number=session.current_question_index + 1,
                total_questions=len(session.questions),
                evaluation=evaluation.to_dict(),
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
        """Generate a summary of the interview."""
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
            prompt = self.retriever.build_prompt(
                resume_id=resume_id,
                focus_area=focus_area,
                question_type=mode,
                n_questions=num_questions,
            )
            
            system_prompt = f"""You are an expert {mode} interviewer. Generate exactly {num_questions} interview questions.

Rules:
1. Questions must be specific to the candidate's resume
2. Questions should probe for depth and specifics
3. Include a mix of verification and exploratory questions
4. For technical mode: focus on skills, projects, technical decisions
5. For HR/behavioral mode: focus on experiences, teamwork, leadership
6. For mixed mode: balance both types

IMPORTANT: Return ONLY the questions, one per line, numbered 1 to {num_questions}.
Do not include any other text, explanations, or formatting.

Example format:
1. Tell me about your experience with Python and how you used it in your projects?
2. Can you describe a challenging technical problem you solved?
3. How did you handle team collaboration in your previous role?"""

            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=2000,
            )
            
            # Parse questions from response
            questions = self._parse_questions(response, num_questions)

            if len(questions) < num_questions:
                print(f"Warning: Only parsed {len(questions)} questions, expected {num_questions}")
                print(f"LLM Response: {response[:500]}...")
            
                default_questions = [
                    "Can you walk me through your most significant project and your role in it?",
                    "What technical challenges have you faced and how did you overcome them?",
                    "How do you approach learning new technologies?",
                    "Describe a situation where you had to collaborate with a difficult team member.",
                    "What are you most proud of in your career so far?",
                ]
            
                while len(questions) < num_questions and default_questions:
                    questions.append(default_questions.pop(0))
        
            return questions[:num_questions]
        
        except Exception as e:
            print(f"Error generating questions: {e}")
            return [
                "Can you tell me about your background and experience?",
                "What project are you most proud of and why?",
                "How do you handle challenging technical problems?",
            ][:num_questions]

    
    def _parse_questions(self, response: str, expected_count: int) -> List[str]:
        """Parse questions from LLM response."""
        import re
        
        questions = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering
            cleaned = re.sub(r'^[\d]+[.)\-:]\s*', '', line)
            cleaned = re.sub(r'^\*+\s*', '', cleaned)
            cleaned = cleaned.strip()
            
            if cleaned and len(cleaned) > 15:  # Minimum question length
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