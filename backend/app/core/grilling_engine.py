"""
Enhanced Grilling Engine based on "Requirements Elicitation Follow-Up Question Generation" paper.

Key features:
1. Mistake-Guided Framework (14 mistake types as gap labels)
2. Multi-dimensional evaluation (Relevancy, Clarity, Informativeness)
3. Context-aware follow-up generation
4. Resume consistency checking
5. Strict evaluation with improved fallback logic
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
    
    Follow-up Question Gaps:
    """
    # From "Follow-up questions" category
    NO_SPECIFIC_EXAMPLE = "no_specific_example"          # Failed to provide concrete examples
    NO_ALTERNATIVES_CONSIDERED = "no_alternatives"       # Didn't discuss alternative approaches
    UNCLEAR_STATEMENT = "unclear_statement"              # Said something unclear/ambiguous
    CONTRADICTORY_STATEMENT = "contradictory"            # Said something contradictory
    MISSING_TACIT_KNOWLEDGE = "missing_tacit"            # Has knowledge but didn't share it
    
    # From "Question framing" category - adapted for answer evaluation
    TOO_GENERIC = "too_generic"                          # Answer is generic, not domain-specific
    TOO_LONG_UNFOCUSED = "too_long_unfocused"           # Rambling without clear point
    USES_JARGON_WITHOUT_EXPLANATION = "unexplained_jargon"  # Uses terms without explaining
    MISSING_TECHNICAL_DEPTH = "no_tech_depth"            # Lacks technical details
    INAPPROPRIATE_SCOPE = "wrong_scope"                  # Answer scope doesn't match question
    PROVIDES_SOLUTION_NOT_EXPERIENCE = "solution_not_exp"  # Gives hypothetical instead of real experience
    MIXES_MULTIPLE_TOPICS = "mixed_topics"               # Jumps between unrelated topics
    VAGUE_MULTIPLE_INTERPRETATIONS = "vague_ambiguous"   # Could mean multiple things
    NO_CONCRETE_MEANING = "no_concrete_meaning"          # Can't extract concrete information
    
    # Additional gaps for interview context
    NO_METRICS = "no_metrics"                            # No quantifiable results
    UNCLEAR_PERSONAL_ROLE = "unclear_role"               # Team vs individual contribution unclear
    NO_RESULT_OUTCOME = "no_outcome"                     # Didn't mention the result
    RESUME_INCONSISTENT = "resume_inconsistent"     # Doesn't match resume claims


@dataclass
class GapAnalysis:
    """Analysis of gaps/deficiencies in an answer."""
    detected_gaps: List[GapType] = field(default_factory=list)
    gap_details: Dict[GapType, str] = field(default_factory=dict)  # Gap -> specific detail
    severity: float = 0.0  # 0-1, overall severity
    priority_gap: Optional[GapType] = None  # Most important gap to address
    
    def to_dict(self) -> Dict:
        return {
            "detected_gaps": [g.value for g in self.detected_gaps],
            "gap_details": {g.value: d for g, d in self.gap_details.items()},
            "severity": self.severity,
            "priority_gap": self.priority_gap.value if self.priority_gap else None,
        }


# ============== Multi-dimensional Evaluation ==============

@dataclass
class DetailedScores:
    """
    Multi-dimensional scoring based on paper's evaluation criteria.
    Plus additional dimensions for technical interviews.
    """
    # Paper's three core dimensions
    relevancy: float = 0.0      # Does it answer the question asked?
    clarity: float = 0.0        # Is it clear and understandable?
    informativeness: float = 0.0  # Does it provide useful information?
    
    # Additional dimensions for interview evaluation
    specificity: float = 0.0    # Concrete examples vs generic statements
    quantification: float = 0.0  # Metrics, numbers, measurable outcomes
    depth: float = 0.0          # Technical/professional depth
    completeness: float = 0.0   # STAR method completeness
    
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
    score: float  # 0.0 - 1.0 overall score
    detailed_scores: DetailedScores
    gap_analysis: GapAnalysis
    missing_elements: List[str]
    strengths: List[str]
    suggested_follow_up: Optional[str]
    reasoning: str
    follow_up_count: int = 0  # How many follow-ups have been asked

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


# ============== Follow-up Question Templates (Mistake-Guided) ==============

FOLLOWUP_TEMPLATES: Dict[GapType, List[str]] = {
    GapType.NO_SPECIFIC_EXAMPLE: [
        "Can you walk me through a specific instance where you actually did this?",
        "Could you give me a concrete example from your experience?",
        "Tell me about a particular project or situation where you applied this.",
    ],
    GapType.NO_ALTERNATIVES_CONSIDERED: [
        "What other approaches did you consider, and why did you choose this one?",
        "Were there alternative solutions you evaluated? What were the trade-offs?",
        "What would you have done differently if you had more time or resources?",
    ],
    GapType.UNCLEAR_STATEMENT: [
        "Could you clarify what you mean by '{unclear_part}'?",
        "I want to make sure I understand - can you elaborate on '{unclear_part}'?",
        "What specifically do you mean when you say '{unclear_part}'?",
    ],
    GapType.CONTRADICTORY_STATEMENT: [
        "Earlier you mentioned {point_a}, but then you said {point_b}. Can you help me understand how these fit together?",
        "I noticed you mentioned both {point_a} and {point_b} - could you clarify?",
    ],
    GapType.MISSING_TACIT_KNOWLEDGE: [
        "Based on your experience, what insights or lessons learned would you share about this?",
        "What's something that someone new to this wouldn't know but is crucial?",
        "What did you learn from this that isn't obvious from the outside?",
    ],
    GapType.TOO_GENERIC: [
        "That's a good general answer. Can you make it more specific to your actual experience?",
        "How did this play out specifically in your work? Give me a real example.",
        "Can you ground this in a specific situation you actually faced?",
    ],
    GapType.NO_METRICS: [
        "What were the measurable outcomes? Any specific numbers you can share?",
        "How did you measure success? What metrics improved?",
        "Can you quantify the impact? For example, percentage improvements, time saved, or revenue affected?",
    ],
    GapType.UNCLEAR_PERSONAL_ROLE: [
        "What was your specific role in this? What did YOU personally contribute?",
        "When you say 'we,' what parts were you directly responsible for?",
        "Can you separate your individual contributions from the team's work?",
    ],
    GapType.NO_RESULT_OUTCOME: [
        "What was the end result? How did this turn out?",
        "What happened after you implemented this? What was the impact?",
        "How did this story end? What were the outcomes?",
    ],
    GapType.MISSING_TECHNICAL_DEPTH: [
        "Can you go deeper into the technical implementation?",
        "What specific technologies, algorithms, or methods did you use?",
        "Walk me through the technical architecture or approach.",
    ],
    GapType.INAPPROPRIATE_SCOPE: [
        "Let me refocus - specifically regarding {original_topic}, can you address that directly?",
        "That's interesting, but I'd like to hear more specifically about {original_topic}.",
    ],
    GapType.PROVIDES_SOLUTION_NOT_EXPERIENCE: [
        "That's a good theoretical approach. Have you actually implemented something like this?",
        "Can you tell me about a time you actually did this, not just how you would do it?",
        "Do you have real experience with this, or is this how you would approach it hypothetically?",
    ],
    GapType.VAGUE_MULTIPLE_INTERPRETATIONS: [
        "Could you be more specific? What exactly do you mean?",
        "Can you give me more concrete details about this?",
        "I need more specifics - can you elaborate?",
    ],
    GapType.RESUME_INCONSISTENT: [
        "In your resume, you mentioned {resume_claim}. Can you tell me more about how that connects to what you just described?",
        "I noticed your resume says {resume_claim}. How does that relate to what you're telling me now?",
    ],
}


# ============== Main Grilling Engine ==============

class GrillingEngine:
    """
    Enhanced Grilling Engine implementing paper's Mistake-Guided Framework.
    
    Key improvements:
    1. Gap/Mistake type detection (14 types from paper)
    2. Multi-dimensional scoring (7 dimensions)
    3. Context-aware evaluation (uses conversation history)
    4. Resume consistency checking
    5. Strict evaluation with better fallback
    6. Forced first follow-up mechanism
    """
    
    # Thresholds
    MIN_ANSWER_LENGTH = 50
    SUFFICIENT_SCORE_THRESHOLD = 0.65   # Must score above this to pass
    HIGH_SCORE_THRESHOLD = 0.80         # Excellent answer, may skip follow-up
    
    # Follow-up settings
    FORCE_FIRST_FOLLOWUP = True         # Always ask at least one follow-up
    MIN_FOLLOWUPS_PER_QUESTION = 1      # Minimum before moving on
    
    def __init__(self, llm_service: BaseLLMService):
        self.llm = llm_service
    
    async def evaluate_answer(
        self,
        question: str,
        answer: str,
        resume_context: str = "",
        question_type: str = "mixed",
        conversation_history: List[Dict] = None,
        follow_up_count: int = 0,
    ) -> AnswerEvaluation:
        """
        Comprehensive answer evaluation with gap detection.
        
        Args:
            question: The interview question
            answer: Candidate's answer
            resume_context: Relevant resume excerpts
            question_type: 'hr', 'tech', or 'mixed'
            conversation_history: Previous Q&A exchanges
            follow_up_count: Number of follow-ups already asked for this question
        """
        
        # Quick check: very short answers
        if len(answer.strip()) < self.MIN_ANSWER_LENGTH:
            return self._create_short_answer_evaluation(answer, follow_up_count)
        
        # Main LLM evaluation
        try:
            evaluation = await self._llm_evaluate(
                question=question,
                answer=answer,
                resume_context=resume_context,
                question_type=question_type,
                conversation_history=conversation_history,
                follow_up_count=follow_up_count,
            )
        except Exception as e:
            print(f"LLM evaluation failed: {e}")
            evaluation = self._fallback_evaluation(answer, question, follow_up_count)
        
        # Apply forced first follow-up rule
        if self.FORCE_FIRST_FOLLOWUP and follow_up_count < self.MIN_FOLLOWUPS_PER_QUESTION:
            if evaluation.score < self.HIGH_SCORE_THRESHOLD:
                evaluation.is_sufficient = False
                if not evaluation.suggested_follow_up:
                    evaluation.suggested_follow_up = await self._generate_gap_based_followup(
                        question, answer, evaluation.gap_analysis, question_type
                    )
        
        return evaluation
    
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
                    GapType.TOO_GENERIC: "Answer is too brief to evaluate",
                    GapType.NO_SPECIFIC_EXAMPLE: "No examples provided",
                },
                severity=0.9,
                priority_gap=GapType.NO_SPECIFIC_EXAMPLE,
            ),
            missing_elements=["detailed explanation", "specific examples", "concrete details"],
            strengths=[],
            suggested_follow_up="Your answer is quite brief. Could you elaborate with specific examples from your experience?",
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
        """
        Comprehensive LLM-based evaluation with gap detection.
        """
        
        # Build conversation context
        history_context = self._format_conversation_history(conversation_history)
        
        system_prompt = """You are a STRICT interview evaluator. Your job is to:
1. Score the answer on multiple dimensions
2. Identify specific GAPS (deficiencies) in the answer
3. Determine if follow-up questions are needed

BE STRICT: Most interview answers need improvement. Look for:
- Vague statements without concrete examples
- Missing quantifiable results or metrics
- Lack of technical depth (for tech questions)
- Incomplete STAR structure (Situation, Task, Action, Result)
- Claims not backed by evidence
- Unclear personal role vs team contribution

A passing score (0.65+) requires:
- Specific, concrete examples (not generic statements)
- Clear personal contribution
- Some form of measurable outcome or result
- Appropriate depth for the question type

Return ONLY valid JSON."""

        prompt = f"""Evaluate this interview answer comprehensively:

QUESTION: {question}

CANDIDATE'S ANSWER: {answer}

{f"RELEVANT RESUME CONTEXT: {resume_context[:800]}" if resume_context else ""}

{f"CONVERSATION HISTORY: {history_context}" if history_context else ""}

This is follow-up attempt #{follow_up_count + 1} for this question.
Question type: {question_type}

Evaluate and return this JSON structure:
{{
  "scores": {{
    "relevancy": 0.0,
    "clarity": 0.0,
    "informativeness": 0.0,
    "specificity": 0.0,
    "quantification": 0.0,
    "depth": 0.0,
    "completeness": 0.0
  }},
  "detected_gaps": ["gap_type1", "gap_type2"],
  "gap_details": {{
    "gap_type1": "specific detail about this gap"
  }},
  "priority_gap": "most_important_gap_type",
  "missing_elements": ["list", "of", "missing", "things"],
  "strengths": ["any", "strengths"],
  "is_sufficient": false,
  "suggested_follow_up": "A specific follow-up question targeting the priority gap",
  "reasoning": "Brief explanation of the evaluation"
}}

VALID GAP TYPES (use these exact values):
- no_specific_example: No concrete examples given
- no_alternatives: Didn't discuss alternative approaches
- unclear_statement: Something was unclear
- contradictory: Said something contradictory
- missing_tacit: Has knowledge but didn't share insights
- too_generic: Answer is generic, not specific
- unexplained_jargon: Uses terms without explaining
- no_tech_depth: Lacks technical details
- wrong_scope: Answer doesn't match question scope
- solution_not_exp: Hypothetical instead of real experience
- mixed_topics: Jumps between unrelated topics
- vague_ambiguous: Could mean multiple things
- no_metrics: No quantifiable results
- unclear_role: Team vs individual unclear
- no_outcome: Didn't mention the result
- resume_inconsistent: Doesn't match resume

SCORING GUIDE (0.0 to 1.0):
- 0.0-0.3: Poor/Missing
- 0.4-0.5: Below average
- 0.6-0.7: Adequate
- 0.8-0.9: Good
- 1.0: Excellent

Be strict but fair. Most first-attempt answers should score 0.4-0.6."""

        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=1000,
        )
        
        return self._parse_llm_evaluation(response, follow_up_count)
    
    def _format_conversation_history(self, history: List[Dict] = None) -> str:
        """Format conversation history for context."""
        if not history:
            return ""
        
        # Take last 6 messages (3 exchanges)
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
        """Parse LLM response into AnswerEvaluation."""
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
                    # Try to map common variations
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
                len(detected_gaps) <= 2  # Allow minor gaps
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
            print(f"Failed to parse LLM evaluation: {e}")
            raise
    
    def _fallback_evaluation(
        self,
        answer: str,
        question: str,
        follow_up_count: int,
    ) -> AnswerEvaluation:
        """
        Improved fallback evaluation using heuristics.
        Much stricter than before - not just based on length.
        """
        answer_lower = answer.lower()
        word_count = len(answer.split())
        
        # Check for specific indicators
        has_numbers = bool(re.search(
            r'\d+%|\d+\s*(percent|users|customers|times|hours|days|months|years|million|thousand|k\b|x\b)',
            answer_lower
        ))
        
        specificity_indicators = [
            'for example', 'specifically', 'in particular', 'such as',
            'instance', 'when i', 'we implemented', 'i built', 'i designed',
            'i created', 'i developed', 'i led', 'i managed', 'my role',
            'i was responsible', 'i personally'
        ]
        has_specificity = sum(1 for ind in specificity_indicators if ind in answer_lower)
        
        result_indicators = [
            'resulted in', 'led to', 'achieved', 'improved', 'reduced',
            'increased', 'saved', 'outcome', 'result was', 'success',
            'impact', 'effect'
        ]
        has_results = sum(1 for ind in result_indicators if ind in answer_lower)
        
        technical_indicators = [
            'algorithm', 'architecture', 'database', 'api', 'framework',
            'library', 'deploy', 'server', 'cloud', 'aws', 'docker',
            'kubernetes', 'sql', 'python', 'java', 'react', 'node'
        ]
        has_technical = sum(1 for ind in technical_indicators if ind in answer_lower)
        
        # Calculate scores based on indicators
        base_score = 0.3
        
        # Length contribution (diminishing returns)
        if word_count > 30:
            base_score += 0.05
        if word_count > 60:
            base_score += 0.05
        if word_count > 100:
            base_score += 0.05
        
        # Specificity contribution
        if has_specificity >= 1:
            base_score += 0.10
        if has_specificity >= 2:
            base_score += 0.05
        
        # Numbers/metrics contribution
        if has_numbers:
            base_score += 0.10
        
        # Results contribution
        if has_results >= 1:
            base_score += 0.08
        if has_results >= 2:
            base_score += 0.04
        
        # Technical depth (for tech questions)
        if has_technical >= 1:
            base_score += 0.05
        if has_technical >= 3:
            base_score += 0.05
        
        # Cap the score
        base_score = min(base_score, 0.75)
        
        # Determine gaps
        detected_gaps = []
        gap_details = {}
        
        if has_specificity < 1:
            detected_gaps.append(GapType.NO_SPECIFIC_EXAMPLE)
            gap_details[GapType.NO_SPECIFIC_EXAMPLE] = "No concrete examples detected"
        
        if not has_numbers:
            detected_gaps.append(GapType.NO_METRICS)
            gap_details[GapType.NO_METRICS] = "No quantifiable metrics mentioned"
        
        if has_results < 1:
            detected_gaps.append(GapType.NO_RESULT_OUTCOME)
            gap_details[GapType.NO_RESULT_OUTCOME] = "No clear outcome or result stated"
        
        if word_count < 50 or has_specificity < 1:
            detected_gaps.append(GapType.TOO_GENERIC)
            gap_details[GapType.TOO_GENERIC] = "Answer lacks specific details"
        
        # Priority gap
        priority_order = [
            GapType.NO_SPECIFIC_EXAMPLE,
            GapType.NO_METRICS,
            GapType.NO_RESULT_OUTCOME,
            GapType.TOO_GENERIC,
        ]
        priority_gap = None
        for gap in priority_order:
            if gap in detected_gaps:
                priority_gap = gap
                break
        
        # Is sufficient? Stricter criteria
        is_sufficient = (
            base_score >= self.SUFFICIENT_SCORE_THRESHOLD and
            has_specificity >= 1 and
            (has_numbers or has_results >= 1) and
            follow_up_count >= self.MIN_FOLLOWUPS_PER_QUESTION
        )
        
        # Generate follow-up based on priority gap
        suggested_follow_up = None
        if priority_gap and priority_gap in FOLLOWUP_TEMPLATES:
            templates = FOLLOWUP_TEMPLATES[priority_gap]
            suggested_follow_up = templates[0]  # Use first template
        else:
            suggested_follow_up = "Could you provide more specific details with concrete examples and measurable outcomes?"
        
        return AnswerEvaluation(
            is_sufficient=is_sufficient,
            score=base_score,
            detailed_scores=DetailedScores(
                relevancy=0.5,
                clarity=0.5 if word_count > 30 else 0.3,
                informativeness=0.4 + (0.1 if has_results else 0),
                specificity=0.3 + (0.2 * min(has_specificity, 2)),
                quantification=0.5 if has_numbers else 0.2,
                depth=0.3 + (0.1 * min(has_technical, 3)),
                completeness=0.3 + (0.1 if has_specificity else 0) + (0.1 if has_results else 0),
            ),
            gap_analysis=GapAnalysis(
                detected_gaps=detected_gaps,
                gap_details=gap_details,
                severity=1.0 - base_score,
                priority_gap=priority_gap,
            ),
            missing_elements=[gap_details.get(g, str(g)) for g in detected_gaps[:3]],
            strengths=self._identify_strengths(has_specificity, has_numbers, has_results, has_technical),
            suggested_follow_up=suggested_follow_up,
            reasoning="Evaluated using heuristic analysis.",
            follow_up_count=follow_up_count,
        )
    
    def _identify_strengths(
        self,
        has_specificity: int,
        has_numbers: bool,
        has_results: int,
        has_technical: int,
    ) -> List[str]:
        """Identify strengths in the answer."""
        strengths = []
        if has_specificity >= 1:
            strengths.append("Provides some specific details")
        if has_numbers:
            strengths.append("Includes quantifiable metrics")
        if has_results >= 1:
            strengths.append("Mentions outcomes/results")
        if has_technical >= 2:
            strengths.append("Shows technical knowledge")
        if not strengths:
            strengths.append("Attempted to answer the question")
        return strengths
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response with robust error handling."""
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
        
        raise ValueError(f"Could not parse JSON: {response[:300]}")
    
    async def _generate_gap_based_followup(
        self,
        question: str,
        answer: str,
        gap_analysis: GapAnalysis,
        question_type: str,
    ) -> str:
        """Generate a follow-up question based on detected gaps."""
        
        priority_gap = gap_analysis.priority_gap
        
        # Try to use template first
        if priority_gap and priority_gap in FOLLOWUP_TEMPLATES:
            templates = FOLLOWUP_TEMPLATES[priority_gap]
            # Pick a template and try to fill in placeholders
            template = templates[0]
            
            # Simple placeholder filling
            if "{" not in template:
                return template
        
        # Generate using LLM for more context-aware follow-up
        gaps_str = ", ".join([g.value for g in gap_analysis.detected_gaps[:3]])
        gap_detail = gap_analysis.gap_details.get(priority_gap, "needs more detail")
        
        prompt = f"""Generate ONE specific follow-up interview question.

Original Question: {question}
Candidate's Answer: {answer}

Identified Gaps: {gaps_str}
Priority Gap: {priority_gap.value if priority_gap else 'general improvement needed'}
Gap Detail: {gap_detail}

Generate a pointed follow-up question that:
1. Directly addresses the priority gap
2. References something specific from their answer
3. Cannot be answered with yes/no
4. Pushes for concrete details, examples, or metrics

Return ONLY the question, nothing else."""

        try:
            follow_up = await self.llm.generate(
                prompt=prompt,
                system_prompt="You are a skilled interviewer. Generate one incisive follow-up question.",
                temperature=0.7,
                max_tokens=150,
            )
            
            follow_up = follow_up.strip().strip('"').strip("'")
            if not follow_up.endswith('?'):
                follow_up += '?'
            
            return follow_up
            
        except Exception as e:
            print(f"Follow-up generation failed: {e}")
            # Return template-based fallback
            if priority_gap and priority_gap in FOLLOWUP_TEMPLATES:
                return FOLLOWUP_TEMPLATES[priority_gap][0]
            return "Could you provide more specific details and concrete examples?"
    
    async def generate_follow_up(
        self,
        question: str,
        answer: str,
        evaluation: AnswerEvaluation,
        conversation_history: List[Dict] = None,
        question_type: str = "mixed",
    ) -> str:
        """
        Generate a contextual follow-up question.
        
        Uses the gap analysis to create targeted follow-ups.
        """
        
        # If evaluation already has a good follow-up, use it
        if evaluation.suggested_follow_up and len(evaluation.suggested_follow_up) > 20:
            return evaluation.suggested_follow_up
        
        # Generate based on gaps
        return await self._generate_gap_based_followup(
            question=question,
            answer=answer,
            gap_analysis=evaluation.gap_analysis,
            question_type=question_type,
        )
    
    async def check_resume_consistency(
        self,
        answer: str,
        resume_context: str,
        question: str,
    ) -> Tuple[bool, List[str]]:
        """
        Check if the answer is consistent with resume claims.
        
        Returns:
            Tuple of (is_consistent, list of inconsistencies)
        """
        if not resume_context:
            return True, []
        
        prompt = f"""Compare this interview answer with the resume context.

RESUME CONTEXT:
{resume_context[:1000]}

INTERVIEW QUESTION: {question}

CANDIDATE'S ANSWER: {answer}

Check for:
1. Contradictions between answer and resume
2. Claims in answer not supported by resume
3. Important resume details that were omitted or contradicted

Return JSON:
{{
  "is_consistent": true,
  "inconsistencies": ["list of specific inconsistencies if any"],
  "omitted_details": ["important resume details not mentioned"]
}}

If everything is consistent, return is_consistent: true with empty lists.
Return ONLY the JSON."""

        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt="You are verifying consistency between interview answers and resume claims.",
                temperature=0.2,
                max_tokens=300,
            )
            
            data = self._parse_json_response(response)
            is_consistent = data.get("is_consistent", True)
            inconsistencies = data.get("inconsistencies", [])
            
            return is_consistent, inconsistencies
            
        except Exception as e:
            print(f"Resume consistency check failed: {e}")
            return True, []  # Assume consistent on error
    
    def should_grill(
        self,
        evaluation: AnswerEvaluation,
        follow_up_count: int,
        max_follow_ups: int,
    ) -> bool:
        """
        Decide whether to ask a follow-up question.
        
        Enhanced logic:
        1. Always follow up on first attempt (unless excellent)
        2. Continue if significant gaps detected
        3. Continue if score below threshold
        4. Respect max follow-up limit
        """
        # Hard limit
        if follow_up_count >= max_follow_ups:
            return False
        
        # Force first follow-up (unless excellent)
        if self.FORCE_FIRST_FOLLOWUP and follow_up_count < self.MIN_FOLLOWUPS_PER_QUESTION:
            if evaluation.score < self.HIGH_SCORE_THRESHOLD:
                return True
        
        # Continue if not sufficient
        if not evaluation.is_sufficient:
            return True
        
        # Continue if score below threshold
        if evaluation.score < self.SUFFICIENT_SCORE_THRESHOLD:
            return True
        
        # Continue if significant gaps remain
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