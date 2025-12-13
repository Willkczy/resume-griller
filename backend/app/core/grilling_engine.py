"""
Grilling Engine for Interview Coach.
Evaluates candidate answers and generates follow-up questions.
"""

from typing import Optional, List, Dict
from dataclasses import dataclass
import json
import re

from backend.app.services.llm_service import BaseLLMService


@dataclass
class AnswerEvaluation:
    """Evaluation result of a candidate's answer."""
    is_sufficient: bool
    score: float  # 0.0 - 1.0
    missing_elements: List[str]
    strengths: List[str]
    suggested_follow_up: Optional[str]
    reasoning: str

    def to_dict(self) -> Dict:
        return {
            "is_sufficient": self.is_sufficient,
            "score": self.score,
            "missing_elements": self.missing_elements,
            "strengths": self.strengths,
            "suggested_follow_up": self.suggested_follow_up,
            "reasoning": self.reasoning,
        }


class GrillingEngine:
    """
    Evaluates interview answers and generates follow-up questions.
    """
    
    MIN_ANSWER_LENGTH = 50
    SUFFICIENT_SCORE_THRESHOLD = 0.6
    
    def __init__(self, llm_service: BaseLLMService):
        self.llm = llm_service
    
    async def evaluate_answer(
        self,
        question: str,
        answer: str,
        resume_context: str = "",
        question_type: str = "mixed",
    ) -> AnswerEvaluation:
        """Evaluate a candidate's answer."""
        
        # Quick check: very short answers
        if len(answer.strip()) < self.MIN_ANSWER_LENGTH:
            return AnswerEvaluation(
                is_sufficient=False,
                score=0.2,
                missing_elements=["detailed explanation", "specific examples"],
                strengths=[],
                suggested_follow_up="Could you elaborate on that? Please provide more details and specific examples.",
                reasoning="Answer is too brief to evaluate properly.",
            )
        
        # Use LLM to evaluate
        evaluation = await self._llm_evaluate(
            question=question,
            answer=answer,
            resume_context=resume_context,
            question_type=question_type,
        )
        
        return evaluation
    
    async def _llm_evaluate(
        self,
        question: str,
        answer: str,
        resume_context: str,
        question_type: str,
    ) -> AnswerEvaluation:
        """Use LLM to evaluate the answer."""
        
        system_prompt = """You are an interview evaluator. Return ONLY valid JSON, nothing else."""

        prompt = f"""Evaluate this interview answer and return JSON:

Question: {question}
Answer: {answer}

Return this exact JSON structure with your evaluation:
{{"is_sufficient": true, "score": 0.7, "missing_elements": [], "strengths": ["example strength"], "suggested_follow_up": null, "reasoning": "Brief reason"}}

Rules:
- is_sufficient: true if answer is good enough, false if needs follow-up
- score: 0.0-1.0 (0.6+ is passing)
- If is_sufficient is false, provide suggested_follow_up question
- Return ONLY the JSON, no other text"""

        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=500,
            )
            
            evaluation_data = self._parse_json_response(response)
            
            return AnswerEvaluation(
                is_sufficient=evaluation_data.get("is_sufficient", False),
                score=float(evaluation_data.get("score", 0.5)),
                missing_elements=evaluation_data.get("missing_elements", []),
                strengths=evaluation_data.get("strengths", []),
                suggested_follow_up=evaluation_data.get("suggested_follow_up"),
                reasoning=evaluation_data.get("reasoning", ""),
            )
            
        except Exception as e:
            print(f"LLM evaluation failed: {e}")
            return self._fallback_evaluation(answer)
    
    def _fallback_evaluation(self, answer: str) -> AnswerEvaluation:
        """Fallback evaluation based on answer length."""
        answer_length = len(answer.strip())
        
        if answer_length < 100:
            return AnswerEvaluation(
                is_sufficient=False,
                score=0.3,
                missing_elements=["more details", "specific examples"],
                strengths=[],
                suggested_follow_up="Could you provide more specific details and examples?",
                reasoning="Answer appears brief.",
            )
        elif answer_length < 300:
            return AnswerEvaluation(
                is_sufficient=True,
                score=0.65,
                missing_elements=[],
                strengths=["provided some detail"],
                suggested_follow_up=None,
                reasoning="Answer has reasonable detail.",
            )
        else:
            return AnswerEvaluation(
                is_sufficient=True,
                score=0.8,
                missing_elements=[],
                strengths=["detailed response"],
                suggested_follow_up=None,
                reasoning="Answer is detailed.",
            )
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response."""
        response = response.strip()
        
        # Remove markdown code blocks
        if "```" in response:
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*', '', response)
            response = response.strip()
        
        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Find JSON object in response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not parse JSON: {response[:200]}")
    
    async def generate_follow_up(
        self,
        question: str,
        answer: str,
        evaluation: AnswerEvaluation,
        conversation_history: List[Dict] = None,
    ) -> str:
        """Generate a follow-up question."""
        
        if evaluation.suggested_follow_up:
            return evaluation.suggested_follow_up
        
        missing = ", ".join(evaluation.missing_elements) if evaluation.missing_elements else "more details"
        
        prompt = f"""Generate ONE follow-up interview question.

Original Question: {question}
Candidate's Answer: {answer}
Missing: {missing}

Generate a specific follow-up question that asks for concrete examples or details.
Return ONLY the question, nothing else."""

        try:
            follow_up = await self.llm.generate(
                prompt=prompt,
                system_prompt="You are an interviewer. Generate one follow-up question.",
                temperature=0.7,
                max_tokens=100,
            )
            return follow_up.strip().strip('"')
        except Exception as e:
            return f"Could you provide more specific details about {missing}?"
    
    def should_grill(self, evaluation: AnswerEvaluation, follow_up_count: int, max_follow_ups: int) -> bool:
        """Decide whether to ask a follow-up question."""
        if follow_up_count >= max_follow_ups:
            return False
        
        if not evaluation.is_sufficient:
            return True
        
        if evaluation.score < self.SUFFICIENT_SCORE_THRESHOLD:
            return True
        
        return False