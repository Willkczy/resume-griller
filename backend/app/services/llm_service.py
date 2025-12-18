"""
LLM Service Abstraction for Resume Griller.
Supports Claude, OpenAI, Gemini, Groq, Custom (vLLM), and Hybrid mode.

Hybrid Mode:
- Uses Groq for preprocessing (summarization, question generation)
- Uses Custom Model for interview execution (evaluation, follow-ups)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from abc import ABC, abstractmethod
from typing import Optional, AsyncIterator, Dict, List
import asyncio

from backend.app.config import settings


class BaseLLMService(ABC):
    """Abstract base class for LLM services."""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Generate a streaming response from the LLM."""
        pass


class AnthropicService(BaseLLMService):
    """Claude API service."""
    
    def __init__(self):
        try:
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
            self.model = settings.ANTHROPIC_MODEL
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "You are a helpful assistant.",
            messages=messages,
        )
        
        return response.content[0].text
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        messages = [{"role": "user", "content": prompt}]
        
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "You are a helpful assistant.",
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text


class OpenAIService(BaseLLMService):
    """OpenAI API service."""
    
    def __init__(self):
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            self.model = settings.OPENAI_MODEL
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        return response.choices[0].message.content
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class GeminiService(BaseLLMService):
    """Google Gemini API service."""
    
    def __init__(self):
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(
                model_name=settings.GEMINI_MODEL,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 1024,
                }
            )
            self.genai = genai
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate response using Gemini."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        loop = asyncio.get_event_loop()
        
        def _generate():
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config,
            )
            return response.text
        
        return await loop.run_in_executor(None, _generate)
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Generate streaming response using Gemini."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        loop = asyncio.get_event_loop()
        
        def _generate_stream():
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config,
                stream=True,
            )
            return response
        
        response = await loop.run_in_executor(None, _generate_stream)
        
        def _get_chunks():
            chunks = []
            for chunk in response:
                if chunk.text:
                    chunks.append(chunk.text)
            return chunks
        
        chunks = await loop.run_in_executor(None, _get_chunks)
        for chunk in chunks:
            yield chunk


class GroqService(BaseLLMService):
    """Groq API service - Ultra fast inference with Llama models."""
    
    def __init__(self):
        try:
            from groq import AsyncGroq
            self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
            self.model = settings.GROQ_MODEL
            print(f"Groq initialized with model: {self.model}")
        except ImportError:
            raise ImportError(
                "groq package not installed. "
                "Run: uv add groq"
            )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate response using Groq."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        return response.choices[0].message.content
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Generate streaming response using Groq."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class CustomModelService(BaseLLMService):
    """
    Custom LoRA model deployed on GCP VM via vLLM.
    Uses OpenAI-compatible completions API.
    
    Note: Model has limited context window (1024 tokens).
    Should be used with pre-processed/condensed prompts.
    """
    
    # Model limits
    MAX_CONTEXT_TOKENS = 1024
    MAX_PROMPT_TOKENS = 700
    MAX_OUTPUT_TOKENS = 300
    CHARS_PER_TOKEN = 4
    
    def __init__(self):
        self.base_url = settings.CUSTOM_MODEL_URL
        self.model = settings.CUSTOM_MODEL_NAME
        self.timeout = settings.CUSTOM_MODEL_TIMEOUT
        print(f"Custom Model Service initialized")
        print(f"  URL: {self.base_url}")
        print(f"  Model: {self.model}")
        print(f"  Max context: {self.MAX_CONTEXT_TOKENS} tokens")
    
    def _format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format prompt for Mistral Instruct model."""
        max_chars = self.MAX_PROMPT_TOKENS * self.CHARS_PER_TOKEN
        
        if system_prompt:
            system_short = system_prompt[:300] if len(system_prompt) > 300 else system_prompt
            combined = f"{system_short}\n\n{prompt}"
        else:
            combined = prompt
        
        if len(combined) > max_chars:
            print(f"[CustomModel] Truncating prompt from {len(combined)} to {max_chars} chars")
            combined = combined[:max_chars]
        
        return f"[INST] {combined} [/INST]"
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate response using custom vLLM model."""
        import httpx
        
        max_tokens = min(max_tokens, self.MAX_OUTPUT_TOKENS)
        formatted_prompt = self._format_prompt(prompt, system_prompt)
        
        print(f"[CustomModel] Request to {self.base_url}/completions")
        print(f"[CustomModel] Prompt: {len(formatted_prompt)} chars (~{len(formatted_prompt)//self.CHARS_PER_TOKEN} tokens)")
        
        payload = {
            "model": self.model,
            "prompt": formatted_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        try:
            async with httpx.AsyncClient(timeout=float(self.timeout)) as client:
                response = await client.post(
                    f"{self.base_url}/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                
                if response.status_code != 200:
                    print(f"[CustomModel] Error {response.status_code}: {response.text[:200]}")
                
                response.raise_for_status()
                
                result = response.json()
                text = result["choices"][0]["text"].strip()
                
                print(f"[CustomModel] Response: {len(text)} chars")
                return text
                
        except httpx.ConnectError as e:
            print(f"[CustomModel] Connection error: {e}")
            raise
        except Exception as e:
            print(f"[CustomModel] Error: {type(e).__name__}: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Generate streaming response (fallback to non-streaming)."""
        full_response = await self.generate(prompt, system_prompt, max_tokens, temperature)
        
        words = full_response.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")


class HybridModelService(BaseLLMService):
    """
    Hybrid model service that uses:
    - Groq (fast API) for preprocessing: summarization, question generation
    - Custom Model (fine-tuned) for interview execution: evaluation, follow-ups
    
    This approach:
    1. Leverages Groq's large context window for complex preprocessing
    2. Distills information into compact prompts for Custom Model
    3. Uses Custom Model's specialized training for interview tasks
    """
    
    def __init__(self):
        # Initialize both services
        self.preprocessor = GroqService()  # For heavy lifting
        self.interviewer = CustomModelService()  # For interview execution
        
        # Storage for prepared context
        self._prepared_contexts: Dict[str, Dict] = {}
        
        print("[HybridModel] Initialized with Groq (preprocessor) + Custom (interviewer)")
    
    async def prepare_interview_context(
        self,
        session_id: str,
        resume_text: str,
        mode: str,
        num_questions: int,
    ) -> Dict:
        """
        Phase 1: Use Groq to prepare all interview materials.
        
        This creates:
        1. Condensed resume summary (~200 tokens)
        2. Interview questions tailored to resume
        3. Per-question context for evaluation
        
        Returns:
            Dict with prepared context to be stored in session
        """
        print(f"[HybridModel] Preparing interview context for session {session_id}")
        
        # Step 1: Generate condensed resume summary
        summary_prompt = f"""Analyze this resume and create a CONDENSED summary (max 200 words) focusing on:
1. Key technical skills
2. Most impressive achievements with metrics
3. Notable projects
4. Years of experience and domain

Resume:
{resume_text[:4000]}

Provide ONLY the summary, no explanations:"""

        resume_summary = await self.preprocessor.generate(
            prompt=summary_prompt,
            system_prompt="You are a resume analyzer. Be extremely concise.",
            max_tokens=300,
            temperature=0.3,
        )
        
        print(f"[HybridModel] Generated resume summary: {len(resume_summary)} chars")
        
        # Step 2: Generate interview questions
        if mode == "tech":
            mode_instruction = "technical questions about architecture, implementation, debugging, and trade-offs"
        elif mode == "hr":
            mode_instruction = "behavioral STAR questions about teamwork, leadership, challenges, and growth"
        else:
            mode_instruction = f"a mix of {num_questions//2} technical and {num_questions - num_questions//2} behavioral questions"
        
        questions_prompt = f"""Based on this resume summary, generate exactly {num_questions} {mode_instruction}.

Resume Summary:
{resume_summary}

Requirements:
1. Questions must be SPECIFIC to this candidate's experience
2. Reference actual projects, technologies, or achievements mentioned
3. Each question should probe for depth and specifics
4. Questions should be challenging but fair

Return ONLY the questions, numbered 1 to {num_questions}:"""

        questions_response = await self.preprocessor.generate(
            prompt=questions_prompt,
            system_prompt=f"You are an expert {mode} interviewer. Generate specific, probing questions.",
            max_tokens=1500,
            temperature=0.8,
        )
        
        # Parse questions
        questions = self._parse_questions(questions_response, num_questions)
        print(f"[HybridModel] Generated {len(questions)} questions")
        
        # Step 3: Create per-question evaluation context
        question_contexts = []
        for i, question in enumerate(questions):
            # Create a compact context for each question
            context_prompt = f"""For this interview question, extract the KEY FACTS from the resume that are relevant.

Question: {question}

Resume Summary:
{resume_summary}

List 3-5 bullet points of relevant facts the candidate should mention. Be concise:"""

            context = await self.preprocessor.generate(
                prompt=context_prompt,
                system_prompt="Extract only the most relevant facts. Be very concise.",
                max_tokens=200,
                temperature=0.3,
            )
            question_contexts.append(context)
        
        print(f"[HybridModel] Created {len(question_contexts)} question contexts")
        
        # Store the prepared context
        prepared = {
            "session_id": session_id,
            "resume_summary": resume_summary,
            "questions": questions,
            "question_contexts": question_contexts,
            "mode": mode,
        }
        
        self._prepared_contexts[session_id] = prepared
        
        return prepared
    
    def get_prepared_context(self, session_id: str) -> Optional[Dict]:
        """Get prepared context for a session."""
        return self._prepared_contexts.get(session_id)
    
    def set_prepared_context(self, session_id: str, context: Dict):
        """Set prepared context for a session (e.g., loaded from session store)."""
        self._prepared_contexts[session_id] = context
    
    async def evaluate_answer(
        self,
        session_id: str,
        question_index: int,
        question: str,
        answer: str,
        conversation_history: List[Dict] = None,
    ) -> Dict:
        """
        Phase 2: Use Custom Model to evaluate an answer.
        
        Uses the pre-prepared compact context.
        """
        context = self._prepared_contexts.get(session_id, {})
        
        # Get the relevant context for this question
        question_context = ""
        if context and question_index < len(context.get("question_contexts", [])):
            question_context = context["question_contexts"][question_index]
        
        # Build conversation context (last 2 exchanges only)
        conv_text = ""
        if conversation_history:
            recent = conversation_history[-4:]  # Last 2 Q&A pairs
            conv_parts = []
            for msg in recent:
                role = "Q" if msg.get("role") == "interviewer" else "A"
                content = msg.get("content", "")[:100]
                conv_parts.append(f"{role}: {content}")
            if conv_parts:
                conv_text = "\nRecent: " + " | ".join(conv_parts)
        
        # Compact evaluation prompt for Custom Model
        eval_prompt = f"""Rate this answer (0.0-1.0).

Q: {question[:150]}
A: {answer[:400]}

Expected: {question_context[:200]}
{conv_text}

Score: specificity, metrics, depth, relevance
JSON only: {{"score": 0.X, "gap": "type", "followup": "question if needed"}}
Gaps: no_specific_example, no_metrics, too_generic, unclear_role, no_outcome"""

        try:
            response = await self.interviewer.generate(
                prompt=eval_prompt,
                system_prompt=None,
                max_tokens=200,
                temperature=0.2,
            )
            
            return self._parse_evaluation(response)
            
        except Exception as e:
            print(f"[HybridModel] Evaluation error: {e}")
            return {
                "score": 0.5,
                "gap": "no_specific_example",
                "followup": "Could you provide more specific details?",
            }
    
    async def generate_followup(
        self,
        session_id: str,
        question: str,
        answer: str,
        gap_type: str,
        conversation_history: List[Dict] = None,
    ) -> str:
        """
        Phase 2: Use Custom Model to generate a follow-up question.
        """
        context = self._prepared_contexts.get(session_id, {})
        resume_summary = context.get("resume_summary", "")[:200]
        
        # Build conversation context
        conv_text = ""
        if conversation_history:
            recent_answers = [
                msg.get("content", "")[:80]
                for msg in conversation_history[-4:]
                if msg.get("role") == "candidate"
            ]
            if recent_answers:
                conv_text = f"\nPrevious answers: {' | '.join(recent_answers)}"
        
        # Compact follow-up prompt
        followup_prompt = f"""Generate 1 follow-up question.

Q: {question[:120]}
A: {answer[:200]}
Gap: {gap_type.replace('_', ' ')}
Context: {resume_summary[:150]}
{conv_text}

Follow-up:"""

        try:
            response = await self.interviewer.generate(
                prompt=followup_prompt,
                system_prompt=None,
                max_tokens=100,
                temperature=0.7,
            )
            
            followup = response.strip()
            if not followup.endswith("?"):
                followup += "?"
            return followup
            
        except Exception as e:
            print(f"[HybridModel] Follow-up generation error: {e}")
            # Fallback templates
            templates = {
                "no_specific_example": "Can you give me a specific example from your experience?",
                "no_metrics": "What were the measurable results or impact?",
                "too_generic": "Can you be more specific about what YOU did?",
                "unclear_role": "What was your personal contribution to this?",
                "no_outcome": "What was the final outcome or result?",
            }
            return templates.get(gap_type, "Could you elaborate on that?")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        Default generate method - routes to appropriate service.
        
        For long prompts (>2000 chars): Use Groq
        For short prompts: Use Custom Model
        """
        if len(prompt) > 2000:
            print("[HybridModel] Using Groq for long prompt")
            return await self.preprocessor.generate(prompt, system_prompt, max_tokens, temperature)
        else:
            print("[HybridModel] Using Custom Model for short prompt")
            return await self.interviewer.generate(prompt, system_prompt, min(max_tokens, 300), temperature)
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Generate streaming response."""
        if len(prompt) > 2000:
            async for chunk in self.preprocessor.generate_stream(prompt, system_prompt, max_tokens, temperature):
                yield chunk
        else:
            async for chunk in self.interviewer.generate_stream(prompt, system_prompt, min(max_tokens, 300), temperature):
                yield chunk
    
    def _parse_questions(self, response: str, expected_count: int) -> List[str]:
        """Parse questions from LLM response."""
        import re
        
        questions = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            cleaned = re.sub(r'^[\d]+[.)\-:]\s*', '', line)
            cleaned = re.sub(r'^\*+\s*', '', cleaned)
            cleaned = re.sub(r'^Question\s*\d*[.:]\s*', '', cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.strip()
            
            if cleaned and len(cleaned) > 20:
                if not cleaned.endswith('?'):
                    cleaned += '?'
                questions.append(cleaned)
        
        return questions[:expected_count]
    
    def _parse_evaluation(self, response: str) -> Dict:
        """Parse evaluation response from Custom Model."""
        import json
        import re
        
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "score": float(data.get("score", 0.5)),
                    "gap": data.get("gap", "no_specific_example"),
                    "followup": data.get("followup", ""),
                }
        except Exception as e:
            print(f"[HybridModel] Parse error: {e}")
        
        return {
            "score": 0.5,
            "gap": "no_specific_example",
            "followup": "",
        }


class LLMServiceFactory:
    """Factory to create LLM service based on configuration."""
    
    _instance: Optional[BaseLLMService] = None
    _hybrid_instance: Optional[HybridModelService] = None
    
    @classmethod
    def get_service(cls) -> BaseLLMService:
        if cls._instance is None:
            cls._instance = cls._create_service()
        return cls._instance
    
    @classmethod
    def get_hybrid_service(cls) -> HybridModelService:
        """Get or create the hybrid model service."""
        if cls._hybrid_instance is None:
            cls._hybrid_instance = HybridModelService()
        return cls._hybrid_instance
    
    @classmethod
    def _create_service(cls) -> BaseLLMService:
        # API mode
        if settings.LLM_PROVIDER == "anthropic":
            if not settings.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not set")
            print("Initializing Anthropic (Claude) Service")
            return AnthropicService()
        
        elif settings.LLM_PROVIDER == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set")
            print("Initializing OpenAI Service")
            return OpenAIService()
        
        elif settings.LLM_PROVIDER == "gemini":
            if not settings.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY not set")
            print("Initializing Google Gemini Service")
            return GeminiService()
        
        elif settings.LLM_PROVIDER == "groq":
            if not settings.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not set")
            print("Initializing Groq Service")
            return GroqService()
        
        elif settings.LLM_PROVIDER == "custom":
            if not settings.CUSTOM_MODEL_URL:
                raise ValueError("CUSTOM_MODEL_URL not set")
            print("Initializing Custom Model Service (GCP VM)")
            return CustomModelService()
        
        elif settings.LLM_PROVIDER == "hybrid":
            print("Initializing Hybrid Model Service")
            return cls.get_hybrid_service()
        
        else:
            raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")
    
    @classmethod
    def reset(cls):
        cls._instance = None
        cls._hybrid_instance = None


def get_llm_service() -> BaseLLMService:
    """Get the configured LLM service."""
    return LLMServiceFactory.get_service()