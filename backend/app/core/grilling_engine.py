"""
Enhanced Grilling Engine based on "Requirements Elicitation Follow-Up Question Generation" paper.

Key features:
1. Mistake-Guided Framework (14 mistake types as gap labels)
2. Multi-dimensional evaluation (Relevancy, Clarity, Informativeness)
3. Context-aware follow-up generation
4. Resume consistency checking
5. Strict evaluation with improved fallback logic
6. Hybrid model support (Groq preprocessing + Custom Model execution)
"""

from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import re

from backend.app.services.llm_service import BaseLLMService


# ============== Gap/Mistake Types (Based on Paper Section III-C) ==============

class GapType(str, Enum):
    """
    Gap types based on the paper's 14 interviewer mistake types.
    Converted from "mistakes to avoid" to "gaps in answer".
    """
    # From "Follow-up questions" category
    NO_SPECIFIC_EXAMPLE = "no_specific_example"
    NO_ALTERNATIVES_CONSIDERED = "no_alternatives"
    UNCLEAR_STATEMENT = "unclear_statement"
    CONTRADICTORY_STATEMENT = "contradictory"
    MISSING_TACIT_KNOWLEDGE = "missing_tacit"
    
    # From "Question framing" category
    TOO_GENERIC = "too_generic"
    TOO_LONG_UNFOCUSED = "too_long_unfocused"
    USES_JARGON_WITHOUT_EXPLANATION = "unexplained_jargon"
    MISSING_TECHNICAL_DEPTH = "no_tech_depth"
    INAPPROPRIATE_SCOPE = "wrong_scope"
    PROVIDES_SOLUTION_NOT_EXPERIENCE = "solution_not_exp"
    MIXES_MULTIPLE_TOPICS = "mixed_topics"
    VAGUE_MULTIPLE_INTERPRETATIONS = "vague_ambiguous"
    NO_CONCRETE_MEANING = "no_concrete_meaning"
    
    # Additional gaps for interview context
    NO_METRICS = "no_metrics"
    UNCLEAR_PERSONAL_ROLE = "unclear_role"
    NO_RESULT_OUTCOME = "no_outcome"
    RESUME_INCONSISTENT = "resume_inconsistent"


@dataclass
class GapAnalysis:
    """Analysis of gaps/deficiencies in an answer."""
    detected_gaps: List[GapType] = field(default_factory=list)
    gap_details: Dict[GapType, str] = field(default_factory=dict)
    severity: float = 0.0
    priority_gap: Optional[GapType] = None
    
    def to_dict(self) -> Dict:
        return {
            "detected_gaps": [g.value for g in self.detected_gaps],
            "gap_details": {g.value: d for g, d in self.gap_details.items()},
            "severity": self.severity,
            "priority_gap": self.priority_gap.value if self.priority_gap else None,
        }


@dataclass
class DetailedScores:
    """Multi-dimensional scoring based on paper's evaluation criteria."""
    relevancy: float = 0.0
    clarity: float = 0.0
    informativeness: float = 0.0
    specificity: float = 0.0
    quantification: float = 0.0
    depth: float = 0.0
    completeness: float = 0.0
    
    @property
    def overall_score(self) -> float:
        """Weighted average of all dimensions."""
        weights = {
            'relevancy': 0.15,
            'clarity': 0.10,
            'informativeness': 0.15,
            'specificity': 0.20,
            'quantification': 0.15,
            'depth': 0.15,
            'completeness': 0.10,
        }
        return (
            self.relevancy * weights['relevancy'] +
            self.clarity * weights['clarity'] +
            self.informativeness * weights['informativeness'] +
            self.specificity * weights['specificity'] +
            self.quantification * weights['quantification'] +
            self.depth * weights['depth'] +
            self.completeness * weights['completeness']
        )
    
    def get_weak_dimensions(self, threshold: float = 0.5) -> List[str]:
        """Get dimensions scoring below threshold."""
        weak = []
        if self.relevancy < threshold:
            weak.append("relevancy")
        if self.clarity < threshold:
            weak.append("clarity")
        if self.informativeness < threshold:
            weak.append("informativeness")
        if self.specificity < threshold:
            weak.append("specificity")
        if self.quantification < threshold:
            weak.append("quantification")
        if self.depth < threshold:
            weak.append("depth")
        if self.completeness < threshold:
            weak.append("completeness")
        return weak
    
    def to_dict(self) -> Dict:
        return {
            "relevancy": self.relevancy,
            "clarity": self.clarity,
            "informativeness": self.informativeness,
            "specificity": self.specificity,
            "quantification": self.quantification,
            "depth": self.depth,
            "completeness": self.completeness,
            "overall": self.overall_score,
        }


@dataclass
class AnswerEvaluation:
    """Complete evaluation result of a candidate's answer."""
    is_sufficient: bool
    score: float
    detailed_scores: DetailedScores
    gap_analysis: GapAnalysis
    missing_elements: List[str]
    strengths: List[str]
    suggested_follow_up: Optional[str]
    reasoning: str
    follow_up_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "is_sufficient": self.is_sufficient,
            "score": self.score,
            "detailed_scores": self.detailed_scores.to_dict(),
            "gap_analysis": self.gap_analysis.to_dict(),
            "missing_elements": self.missing_elements,
            "strengths": self.strengths,
            "suggested_follow_up": self.suggested_follow_up,
            "reasoning": self.reasoning,
        }


# ============== Follow-up Question Templates ==============

FOLLOWUP_TEMPLATES: Dict[GapType, List[str]] = {
    GapType.NO_SPECIFIC_EXAMPLE: [
        "Can you walk me through a specific instance where you actually did this?",
        "Could you give me a concrete example from your experience?",
        "Tell me about a particular project or situation where you applied this.",
    ],
    GapType.NO_ALTERNATIVES_CONSIDERED: [
        "What other approaches did you consider, and why did you choose this one?",
        "Were there alternative solutions you evaluated? What were the trade-offs?",
    ],
    GapType.UNCLEAR_STATEMENT: [
        "Could you clarify what you mean by that?",
        "I want to make sure I understand - can you elaborate?",
    ],
    GapType.TOO_GENERIC: [
        "That's a good general answer. Can you make it more specific to your actual experience?",
        "How did this play out specifically in your work? Give me a real example.",
    ],
    GapType.NO_METRICS: [
        "What were the measurable outcomes? Any specific numbers you can share?",
        "How did you measure success? What metrics improved?",
    ],
    GapType.UNCLEAR_PERSONAL_ROLE: [
        "What was your specific role in this? What did YOU personally contribute?",
        "When you say 'we,' what parts were you directly responsible for?",
    ],
    GapType.NO_RESULT_OUTCOME: [
        "What was the end result? How did this turn out?",
        "What happened after you implemented this? What was the impact?",
    ],
    GapType.MISSING_TECHNICAL_DEPTH: [
        "Can you go deeper into the technical implementation?",
        "What specific technologies, algorithms, or methods did you use?",
    ],
    GapType.PROVIDES_SOLUTION_NOT_EXPERIENCE: [
        "That's a good theoretical approach. Have you actually implemented something like this?",
        "Can you tell me about a time you actually did this?",
    ],
    GapType.VAGUE_MULTIPLE_INTERPRETATIONS: [
        "Could you be more specific? What exactly do you mean?",
        "Can you give me more concrete details about this?",
    ],
    GapType.RESUME_INCONSISTENT: [
        "Can you help me understand how this connects to what's in your resume?",
        "I noticed something different in your resume. Could you clarify?",
    ],
}


# ============== Main Grilling Engine ==============

class GrillingEngine:
    """
    Enhanced Grilling Engine implementing paper's Mistake-Guided Framework.
    
    Supports two modes:
    1. Standard mode: Uses full LLM prompts (for API models like Groq/Gemini)
    2. Hybrid mode: Uses compact prompts (for Custom Model with limited context)
    
    In Hybrid mode:
    - Groq has already preprocessed context (resume summary, question contexts)
    - Custom Model receives compact evaluation prompts
    - Follow-up generation uses templates + short LLM calls
    """
    
    # Thresholds
    MIN_ANSWER_LENGTH = 50
    SUFFICIENT_SCORE_THRESHOLD = 0.58
    HIGH_SCORE_THRESHOLD = 0.72
    
    # Follow-up settings
    FORCE_FIRST_FOLLOWUP = True
    MIN_FOLLOWUPS_PER_QUESTION = 1
    
    def __init__(
        self,
        llm_service: BaseLLMService,
        model_type: str = "api",
        prepared_context: Optional[Dict] = None,
    ):
        """
        Initialize GrillingEngine.
        
        Args:
            llm_service: LLM service for evaluation
            model_type: "api" for full prompts, "custom" for compact prompts
            prepared_context: Pre-processed context from Groq (Hybrid mode)
        """
        self.llm = llm_service
        self.model_type = model_type
        self.prepared_context = prepared_context or {}
        
        # Hybrid mode settings
        if model_type == "custom":
            self.use_compact_prompts = True
            self.max_prompt_chars = 2500
            print(f"[GrillingEngine] Hybrid mode: using compact prompts")
        else:
            self.use_compact_prompts = False
            self.max_prompt_chars = 8000
            print(f"[GrillingEngine] Standard mode: using full prompts")
    
    def set_prepared_context(self, context: Dict):
        """Set prepared context from Groq preprocessing."""
        self.prepared_context = context
    
    def get_question_context(self, question_index: int) -> str:
        """Get pre-prepared context for a specific question."""
        contexts = self.prepared_context.get("question_contexts", [])
        if 0 <= question_index < len(contexts):
            return contexts[question_index]
        return ""
    
    def get_resume_summary(self) -> str:
        """Get pre-prepared resume summary."""
        return self.prepared_context.get("resume_summary", "")
    
    async def evaluate_answer(
        self,
        question: str,
        answer: str,
        resume_context: str = "",
        question_type: str = "mixed",
        conversation_history: List[Dict] = None,
        follow_up_count: int = 0,
        question_index: int = 0,
    ) -> AnswerEvaluation:
        """
        Comprehensive answer evaluation with gap detection.
        
        Routes to appropriate evaluation method based on model_type.
        """
        # Quick check: very short answers
        if len(answer.strip()) < self.MIN_ANSWER_LENGTH:
            return self._create_short_answer_evaluation(answer, follow_up_count)
        
        # Choose evaluation method
        try:
            if self.use_compact_prompts:
                # Hybrid mode: use compact evaluation for Custom Model
                evaluation = await self._llm_evaluate_compact(
                    question=question,
                    answer=answer,
                    question_index=question_index,
                    conversation_history=conversation_history,
                    follow_up_count=follow_up_count,
                )
            else:
                # Standard mode: use full evaluation for API models
                evaluation = await self._llm_evaluate(
                    question=question,
                    answer=answer,
                    resume_context=resume_context,
                    question_type=question_type,
                    conversation_history=conversation_history,
                    follow_up_count=follow_up_count,
                )
        except Exception as e:
            print(f"[GrillingEngine] Evaluation failed: {e}")
            evaluation = self._fallback_evaluation(answer, question, follow_up_count)
        
        # Apply forced first follow-up rule
        if self.FORCE_FIRST_FOLLOWUP and follow_up_count < self.MIN_FOLLOWUPS_PER_QUESTION:
            if evaluation.score < self.HIGH_SCORE_THRESHOLD:
                evaluation.is_sufficient = False
                if not evaluation.suggested_follow_up:
                    if self.use_compact_prompts:
                        evaluation.suggested_follow_up = await self._generate_followup_compact(
                            question, answer, evaluation.gap_analysis, conversation_history
                        )
                    else:
                        evaluation.suggested_follow_up = await self._generate_gap_based_followup(
                            question, answer, evaluation.gap_analysis, question_type
                        )
        
        return evaluation
    
    async def _llm_evaluate_compact(
        self,
        question: str,
        answer: str,
        question_index: int = 0,
        conversation_history: List[Dict] = None,
        follow_up_count: int = 0,
    ) -> AnswerEvaluation:
        """
        Compact evaluation for Custom Model with limited context.
        
        Uses pre-prepared context from Groq preprocessing.
        """
        # Get pre-prepared context
        question_context = self.get_question_context(question_index)
        resume_summary = self.get_resume_summary()
        
        # Build minimal conversation context
        conv_text = ""
        if conversation_history:
            recent = conversation_history[-4:]  # Last 2 Q&A pairs
            conv_parts = []
            for msg in recent:
                role = "Q" if msg.get("role") == "interviewer" else "A"
                content = msg.get("content", "")[:80]
                conv_parts.append(f"{role}: {content}")
            if conv_parts:
                conv_text = "\nRecent: " + " | ".join(conv_parts)
        
        # Compact evaluation prompt
        prompt = f"""Rate this interview answer (0.0-1.0).

Q: {question[:150]}
A: {answer[:500]}

Expected: {question_context[:200] if question_context else resume_summary[:200]}
{conv_text}

Evaluate: specificity, metrics, depth, relevance
Return JSON only:
{{"score": 0.X, "gap": "type", "sufficient": false, "followup": "question"}}

Valid gaps: no_specific_example, no_metrics, too_generic, unclear_role, no_outcome, no_tech_depth"""

        print(f"[GrillingEngine] Compact eval prompt: {len(prompt)} chars")
        
        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=None,
                max_tokens=200,
                temperature=0.2,
            )
            
            return self._parse_compact_evaluation(response, follow_up_count)
            
        except Exception as e:
            print(f"[GrillingEngine] Compact evaluation error: {e}")
            return self._fallback_evaluation(answer, question, follow_up_count)
    
    def _parse_compact_evaluation(self, response: str, follow_up_count: int) -> AnswerEvaluation:
        """Parse compact evaluation response from Custom Model."""
        try:
            # Find JSON in response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                score = float(data.get("score", 0.5))
                score = max(0.0, min(1.0, score))
                
                gap_str = data.get("gap", "no_specific_example")
                is_sufficient = data.get("sufficient", score >= self.SUFFICIENT_SCORE_THRESHOLD)
                followup = data.get("followup", "")
                
                # Map gap string to GapType
                try:
                    gap_type = GapType(gap_str)
                except ValueError:
                    gap_type = GapType.NO_SPECIFIC_EXAMPLE
                
                return AnswerEvaluation(
                    is_sufficient=is_sufficient,
                    score=score,
                    detailed_scores=DetailedScores(
                        relevancy=score,
                        clarity=score,
                        informativeness=score,
                        specificity=score * 0.9,
                        quantification=score * 0.8,
                        depth=score,
                        completeness=score * 0.9,
                    ),
                    gap_analysis=GapAnalysis(
                        detected_gaps=[gap_type],
                        gap_details={gap_type: "Identified by compact evaluation"},
                        severity=1.0 - score,
                        priority_gap=gap_type,
                    ),
                    missing_elements=[],
                    strengths=[],
                    suggested_follow_up=followup if followup else None,
                    reasoning="Compact evaluation (Hybrid mode)",
                    follow_up_count=follow_up_count,
                )
                
        except Exception as e:
            print(f"[GrillingEngine] Parse compact error: {e}")
        
        # Fallback
        return self._fallback_evaluation("", "", follow_up_count)
    
    async def _generate_followup_compact(
        self,
        question: str,
        answer: str,
        gap_analysis: GapAnalysis,
        conversation_history: List[Dict] = None,
    ) -> str:
        """Generate follow-up using compact prompt for Custom Model."""
        priority_gap = gap_analysis.priority_gap
        
        # First try template
        if priority_gap and priority_gap in FOLLOWUP_TEMPLATES:
            return FOLLOWUP_TEMPLATES[priority_gap][0]
        
        # Build conversation context
        conv_text = ""
        if conversation_history:
            recent_answers = [
                msg.get("content", "")[:60]
                for msg in conversation_history[-4:]
                if msg.get("role") == "candidate"
            ]
            if recent_answers:
                conv_text = f"\nPrev: {' | '.join(recent_answers)}"
        
        # Compact follow-up prompt
        prompt = f"""Generate 1 follow-up question.

Q: {question[:100]}
A: {answer[:150]}
Gap: {priority_gap.value if priority_gap else 'needs detail'}
{conv_text}

Follow-up:"""

        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=None,
                max_tokens=80,
                temperature=0.7,
            )
            
            followup = response.strip()
            if followup and not followup.endswith("?"):
                followup += "?"
            return followup
            
        except Exception as e:
            print(f"[GrillingEngine] Compact followup error: {e}")
            return FOLLOWUP_TEMPLATES.get(
                priority_gap,
                ["Could you provide more specific details?"]
            )[0]
    
    def _create_short_answer_evaluation(self, answer: str, follow_up_count: int) -> AnswerEvaluation:
        """Create evaluation for answers that are too short."""
        return AnswerEvaluation(
            is_sufficient=False,
            score=0.15,
            detailed_scores=DetailedScores(
                relevancy=0.3,
                clarity=0.4,
                informativeness=0.1,
                specificity=0.1,
                quantification=0.0,
                depth=0.1,
                completeness=0.1,
            ),
            gap_analysis=GapAnalysis(
                detected_gaps=[GapType.TOO_GENERIC, GapType.NO_SPECIFIC_EXAMPLE],
                gap_details={
                    GapType.TOO_GENERIC: "Answer is too brief",
                    GapType.NO_SPECIFIC_EXAMPLE: "No examples provided",
                },
                severity=0.9,
                priority_gap=GapType.NO_SPECIFIC_EXAMPLE,
            ),
            missing_elements=["detailed explanation", "specific examples"],
            strengths=[],
            suggested_follow_up="Your answer is quite brief. Could you elaborate with specific examples?",
            reasoning="Answer is too short to properly evaluate.",
            follow_up_count=follow_up_count,
        )
    
    async def _llm_evaluate(
        self,
        question: str,
        answer: str,
        resume_context: str,
        question_type: str,
        conversation_history: List[Dict] = None,
        follow_up_count: int = 0,
    ) -> AnswerEvaluation:
        """Full LLM-based evaluation for API models."""
        
        history_context = self._format_conversation_history(conversation_history)
        
        system_prompt = """You are a STRICT interview evaluator. Your job is to:
1. Score the answer on multiple dimensions
2. Identify specific GAPS (deficiencies) in the answer
3. Determine if follow-up questions are needed

BE STRICT: Most interview answers need improvement. Look for:
- Vague statements without concrete examples
- Missing quantifiable results or metrics
- Lack of technical depth (for tech questions)
- Incomplete STAR structure
- Claims not backed by evidence
- Unclear personal role vs team contribution

A passing score (0.65+) requires:
- Specific, concrete examples
- Clear personal contribution
- Some form of measurable outcome
- Appropriate depth for the question type

Return ONLY valid JSON."""

        prompt = f"""Evaluate this interview answer:

QUESTION: {question}

ANSWER: {answer}

{f"RESUME CONTEXT: {resume_context[:800]}" if resume_context else ""}

{f"CONVERSATION: {history_context}" if history_context else ""}

Follow-up #{follow_up_count + 1}. Question type: {question_type}

Return JSON:
{{
  "scores": {{
    "relevancy": 0.0, "clarity": 0.0, "informativeness": 0.0,
    "specificity": 0.0, "quantification": 0.0, "depth": 0.0, "completeness": 0.0
  }},
  "detected_gaps": ["gap_type1"],
  "gap_details": {{"gap_type1": "detail"}},
  "priority_gap": "gap_type",
  "missing_elements": [],
  "strengths": [],
  "is_sufficient": false,
  "suggested_follow_up": "follow-up question",
  "reasoning": "explanation"
}}

Valid gaps: no_specific_example, no_alternatives, unclear_statement, too_generic,
no_metrics, unclear_role, no_outcome, no_tech_depth, solution_not_exp, vague_ambiguous"""

        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=1500,
        )
        
        return self._parse_llm_evaluation(response, follow_up_count)
    
    def _format_conversation_history(self, history: List[Dict] = None) -> str:
        """Format conversation history for context."""
        if not history:
            return ""
        
        recent = history[-6:] if len(history) > 6 else history
        formatted = []
        
        for msg in recent:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:200]
            is_followup = msg.get('is_follow_up', False)
            
            prefix = "Q" if role == 'interviewer' else "A"
            if is_followup:
                prefix = "Follow-up Q"
            
            formatted.append(f"{prefix}: {content}...")
        
        return "\n".join(formatted)
    
    def _parse_llm_evaluation(self, response: str, follow_up_count: int) -> AnswerEvaluation:
        """Parse full LLM evaluation response."""
        try:
            data = self._parse_json_response(response)
            
            # Parse scores
            scores_data = data.get("scores", {})
            detailed_scores = DetailedScores(
                relevancy=float(scores_data.get("relevancy", 0.5)),
                clarity=float(scores_data.get("clarity", 0.5)),
                informativeness=float(scores_data.get("informativeness", 0.5)),
                specificity=float(scores_data.get("specificity", 0.4)),
                quantification=float(scores_data.get("quantification", 0.3)),
                depth=float(scores_data.get("depth", 0.4)),
                completeness=float(scores_data.get("completeness", 0.4)),
            )
            
            # Parse gaps
            detected_gaps = []
            for gap_str in data.get("detected_gaps", []):
                try:
                    detected_gaps.append(GapType(gap_str))
                except ValueError:
                    gap_mapping = {
                        "no_specific_example": GapType.NO_SPECIFIC_EXAMPLE,
                        "no_example": GapType.NO_SPECIFIC_EXAMPLE,
                        "generic": GapType.TOO_GENERIC,
                        "no_metrics": GapType.NO_METRICS,
                        "unclear_role": GapType.UNCLEAR_PERSONAL_ROLE,
                        "no_outcome": GapType.NO_RESULT_OUTCOME,
                        "no_depth": GapType.MISSING_TECHNICAL_DEPTH,
                    }
                    if gap_str in gap_mapping:
                        detected_gaps.append(gap_mapping[gap_str])
            
            # Parse gap details
            gap_details = {}
            for gap_str, detail in data.get("gap_details", {}).items():
                try:
                    gap_type = GapType(gap_str)
                    gap_details[gap_type] = detail
                except ValueError:
                    pass
            
            # Parse priority gap
            priority_gap = None
            priority_str = data.get("priority_gap")
            if priority_str:
                try:
                    priority_gap = GapType(priority_str)
                except ValueError:
                    pass
            
            if not priority_gap and detected_gaps:
                priority_gap = detected_gaps[0]
            
            gap_analysis = GapAnalysis(
                detected_gaps=detected_gaps,
                gap_details=gap_details,
                severity=1.0 - detailed_scores.overall_score,
                priority_gap=priority_gap,
            )
            
            # Determine if sufficient
            overall_score = detailed_scores.overall_score
            is_sufficient = (
                overall_score >= self.SUFFICIENT_SCORE_THRESHOLD and
                data.get("is_sufficient", False) and
                len(detected_gaps) <= 2
            )
            
            return AnswerEvaluation(
                is_sufficient=is_sufficient,
                score=overall_score,
                detailed_scores=detailed_scores,
                gap_analysis=gap_analysis,
                missing_elements=data.get("missing_elements", []),
                strengths=data.get("strengths", []),
                suggested_follow_up=data.get("suggested_follow_up"),
                reasoning=data.get("reasoning", ""),
                follow_up_count=follow_up_count,
            )
            
        except Exception as e:
            print(f"[GrillingEngine] Parse error: {e}")
            return self._fallback_evaluation("", "", follow_up_count)
    
    def _fallback_evaluation(
        self,
        answer: str,
        question: str,
        follow_up_count: int,
    ) -> AnswerEvaluation:
        """Heuristic-based fallback evaluation."""
        answer_lower = answer.lower() if answer else ""
        word_count = len(answer.split()) if answer else 0
        
        # Check for indicators
        has_numbers = bool(re.search(
            r'\d+%|\d+\s*(percent|users|times|hours|days|months|years|million|k\b)',
            answer_lower
        ))
        
        specificity_indicators = [
            'for example', 'specifically', 'instance', 'when i',
            'i built', 'i designed', 'i led', 'my role',
        ]
        has_specificity = sum(1 for ind in specificity_indicators if ind in answer_lower)
        
        result_indicators = [
            'resulted in', 'achieved', 'improved', 'reduced',
            'increased', 'saved', 'outcome', 'impact',
        ]
        has_results = sum(1 for ind in result_indicators if ind in answer_lower)
        
        # Calculate score
        base_score = 0.3
        if word_count > 30:
            base_score += 0.05
        if word_count > 60:
            base_score += 0.05
        if has_specificity >= 1:
            base_score += 0.10
        if has_numbers:
            base_score += 0.10
        if has_results >= 1:
            base_score += 0.08
        
        base_score = min(base_score, 0.70)
        
        # Determine gaps
        detected_gaps = []
        gap_details = {}
        
        if has_specificity < 1:
            detected_gaps.append(GapType.NO_SPECIFIC_EXAMPLE)
            gap_details[GapType.NO_SPECIFIC_EXAMPLE] = "No concrete examples"
        
        if not has_numbers:
            detected_gaps.append(GapType.NO_METRICS)
            gap_details[GapType.NO_METRICS] = "No metrics mentioned"
        
        if has_results < 1:
            detected_gaps.append(GapType.NO_RESULT_OUTCOME)
            gap_details[GapType.NO_RESULT_OUTCOME] = "No outcome stated"
        
        priority_gap = detected_gaps[0] if detected_gaps else GapType.NO_SPECIFIC_EXAMPLE
        
        return AnswerEvaluation(
            is_sufficient=base_score >= self.SUFFICIENT_SCORE_THRESHOLD and follow_up_count >= 1,
            score=base_score,
            detailed_scores=DetailedScores(
                relevancy=0.5,
                clarity=0.5 if word_count > 30 else 0.3,
                informativeness=0.4,
                specificity=0.3 + (0.2 * min(has_specificity, 2)),
                quantification=0.5 if has_numbers else 0.2,
                depth=0.4,
                completeness=0.3,
            ),
            gap_analysis=GapAnalysis(
                detected_gaps=detected_gaps,
                gap_details=gap_details,
                severity=1.0 - base_score,
                priority_gap=priority_gap,
            ),
            missing_elements=[gap_details.get(g, str(g)) for g in detected_gaps[:3]],
            strengths=["Attempted to answer"],
            suggested_follow_up=FOLLOWUP_TEMPLATES.get(priority_gap, ["Could you elaborate?"])[0],
            reasoning="Fallback heuristic evaluation",
            follow_up_count=follow_up_count,
        )
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response."""
        response = response.strip()
        
        if "```" in response:
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*', '', response)
            response = response.strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not parse JSON: {response[:300]}")
    
    async def _generate_gap_based_followup(
        self,
        question: str,
        answer: str,
        gap_analysis: GapAnalysis,
        question_type: str,
    ) -> str:
        """Generate follow-up for API models."""
        priority_gap = gap_analysis.priority_gap
        
        # Try template first
        if priority_gap and priority_gap in FOLLOWUP_TEMPLATES:
            templates = FOLLOWUP_TEMPLATES[priority_gap]
            template = templates[0]
            if "{" not in template:
                return template
        
        # Generate using LLM
        gaps_str = ", ".join([g.value for g in gap_analysis.detected_gaps[:3]])
        gap_detail = gap_analysis.gap_details.get(priority_gap, "needs more detail")
        
        prompt = f"""Generate ONE follow-up interview question.

Original Question: {question}
Answer: {answer}

Gaps: {gaps_str}
Priority: {priority_gap.value if priority_gap else 'general'}
Detail: {gap_detail}

Generate a pointed follow-up that:
1. Addresses the priority gap
2. References their answer
3. Pushes for specifics

Return ONLY the question."""

        try:
            follow_up = await self.llm.generate(
                prompt=prompt,
                system_prompt="Generate one incisive follow-up question.",
                temperature=0.7,
                max_tokens=150,
            )
            
            follow_up = follow_up.strip().strip('"').strip("'")
            if not follow_up.endswith('?'):
                follow_up += '?'
            
            return follow_up
            
        except Exception as e:
            print(f"[GrillingEngine] Follow-up generation failed: {e}")
            if priority_gap and priority_gap in FOLLOWUP_TEMPLATES:
                return FOLLOWUP_TEMPLATES[priority_gap][0]
            return "Could you provide more specific details?"
    
    async def generate_follow_up(
        self,
        question: str,
        answer: str,
        evaluation: AnswerEvaluation,
        conversation_history: List[Dict] = None,
        question_type: str = "mixed",
    ) -> str:
        """Generate a contextual follow-up question."""
        # If evaluation has a good follow-up, use it
        if evaluation.suggested_follow_up and len(evaluation.suggested_follow_up) > 20:
            return evaluation.suggested_follow_up
        
        # Route to appropriate method
        if self.use_compact_prompts:
            return await self._generate_followup_compact(
                question, answer, evaluation.gap_analysis, conversation_history
            )
        else:
            return await self._generate_gap_based_followup(
                question, answer, evaluation.gap_analysis, question_type
            )
    
    async def check_resume_consistency(
        self,
        answer: str,
        resume_context: str,
        question: str,
    ) -> Tuple[bool, List[str]]:
        """Check if answer is consistent with resume (API mode only)."""
        if not resume_context or self.use_compact_prompts:
            return True, []
        
        prompt = f"""Compare answer with resume.

RESUME: {resume_context[:800]}
QUESTION: {question}
ANSWER: {answer}

Return JSON:
{{"is_consistent": true, "inconsistencies": []}}"""

        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt="Verify consistency.",
                temperature=0.2,
                max_tokens=300,
            )
            
            data = self._parse_json_response(response)
            return data.get("is_consistent", True), data.get("inconsistencies", [])
            
        except Exception as e:
            print(f"[GrillingEngine] Consistency check failed: {e}")
            return True, []
    
    def should_grill(
        self,
        evaluation: AnswerEvaluation,
        follow_up_count: int,
        max_follow_ups: int,
    ) -> bool:
        """Decide whether to ask a follow-up question."""
        if follow_up_count >= max_follow_ups:
            return False
        
        if self.FORCE_FIRST_FOLLOWUP and follow_up_count < self.MIN_FOLLOWUPS_PER_QUESTION:
            if evaluation.score < self.HIGH_SCORE_THRESHOLD:
                return True
        
        if not evaluation.is_sufficient:
            return True
        
        if evaluation.score < self.SUFFICIENT_SCORE_THRESHOLD:
            return True
        
        significant_gaps = [
            GapType.NO_SPECIFIC_EXAMPLE,
            GapType.NO_METRICS,
            GapType.UNCLEAR_PERSONAL_ROLE,
            GapType.NO_RESULT_OUTCOME,
            GapType.RESUME_INCONSISTENT,
        ]
        for gap in evaluation.gap_analysis.detected_gaps:
            if gap in significant_gaps and follow_up_count < 2:
                return True
        
        return False