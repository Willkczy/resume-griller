"""
LLM Service Abstraction for Resume Griller.
Supports Claude, OpenAI, Gemini, Groq, Custom (vLLM), and Hybrid mode.

Hybrid Mode:
- Uses Groq for preprocessing (summarization, question generation)
- Custom compact evaluation/follow-up helpers report unsupported under M2 contracts
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from backend.app.config import settings
from backend.app.models.model_execution import TokenUsage
from backend.app.models.model_outputs import (
    MODEL_OUTPUT_SCHEMA_VERSION,
    InvalidOutput,
    QuestionBatch,
    UnsupportedMode,
    parse_questions,
)
from backend.app.services.model_execution import (
    ModelResult,
    call_model,
    chat_usage,
    token_count,
)


class BaseLLMService(ABC):
    """Abstract base class for LLM services."""

    provider = None

    @property
    def model_name(self) -> str | None:
        return self.model if isinstance(self.model, str) else self.model.model_name

    def execution_identity(self, prompt: str) -> tuple[str | None, str | None]:
        return self.provider, self.model_name

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ModelResult:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Generate a streaming response from the LLM."""
        pass


class AnthropicService(BaseLLMService):
    """Claude API service."""

    provider = "anthropic"

    def __init__(self):
        try:
            from anthropic import AsyncAnthropic

            self.client = AsyncAnthropic(
                api_key=settings.ANTHROPIC_API_KEY,
                timeout=settings.LLM_REQUEST_TIMEOUT_SECONDS,
                max_retries=0,
            )
            self.model = settings.ANTHROPIC_MODEL
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ModelResult:
        messages = [{"role": "user", "content": prompt}]

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "You are a helpful assistant.",
            messages=messages,
        )

        usage = response.usage
        return ModelResult(
            text="".join(
                block.text for block in response.content if block.type == "text"
            ),
            provider=self.provider,
            configured_model=self.model_name,
            model=response.model,
            provider_request_id=getattr(response, "_request_id", None),
            finish_reason=response.stop_reason,
            usage=TokenUsage(
                input_tokens=token_count(getattr(usage, "input_tokens", None)),
                output_tokens=token_count(getattr(usage, "output_tokens", None)),
                cache_read_input_tokens=token_count(
                    getattr(usage, "cache_read_input_tokens", None)
                ),
                cache_creation_input_tokens=token_count(
                    getattr(usage, "cache_creation_input_tokens", None)
                ),
            ),
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
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

    provider = "openai"

    def __init__(self):
        try:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                timeout=settings.LLM_REQUEST_TIMEOUT_SECONDS,
                max_retries=0,
            )
            self.model = settings.OPENAI_MODEL
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ModelResult:
        messages = [
            {
                "role": "system",
                "content": system_prompt or "You are a helpful assistant.",
            },
            {"role": "user", "content": prompt},
        ]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return ModelResult(
            text=response.choices[0].message.content or "",
            provider=self.provider,
            configured_model=self.model_name,
            model=response.model,
            provider_request_id=getattr(response, "_request_id", None),
            finish_reason=response.choices[0].finish_reason,
            usage=chat_usage(response.usage),
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        messages = [
            {
                "role": "system",
                "content": system_prompt or "You are a helpful assistant.",
            },
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

    provider = "gemini"

    def __init__(self):
        try:
            import google.generativeai as genai

            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(
                model_name=settings.GEMINI_MODEL,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 1024,
                },
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
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ModelResult:
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
                request_options={
                    "timeout": settings.LLM_REQUEST_TIMEOUT_SECONDS,
                    "retry": None,
                },
            )
            # The protobuf wrapper supplies an empty usage object when absent.
            usage = (
                response.usage_metadata
                if "usage_metadata" in response._result
                else None
            )
            candidate = response.candidates[0] if response.candidates else None
            reason = candidate.finish_reason.name if candidate else None
            text = (
                "".join(part.text for part in candidate.content.parts)
                if candidate
                else ""
            )
            return ModelResult(
                text=text,
                provider=self.provider,
                configured_model=self.model_name,
                model=response.model_version or None,
                finish_reason=reason,
                usage=TokenUsage(
                    input_tokens=token_count(
                        getattr(usage, "prompt_token_count", None)
                    ),
                    output_tokens=token_count(
                        getattr(usage, "candidates_token_count", None)
                    ),
                    total_tokens=token_count(getattr(usage, "total_token_count", None)),
                ),
            )

        return await loop.run_in_executor(None, _generate)

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
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
                request_options={
                    "timeout": settings.LLM_REQUEST_TIMEOUT_SECONDS,
                    "retry": None,
                },
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

    provider = "groq"

    def __init__(self):
        try:
            from groq import AsyncGroq

            self.client = AsyncGroq(
                api_key=settings.GROQ_API_KEY,
                timeout=settings.LLM_REQUEST_TIMEOUT_SECONDS,
                max_retries=0,
            )
            self.model = settings.GROQ_MODEL
        except ImportError:
            raise ImportError("groq package not installed. " "Run: uv add groq")

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ModelResult:
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

        return ModelResult(
            text=response.choices[0].message.content or "",
            provider=self.provider,
            configured_model=self.model_name,
            model=response.model,
            provider_request_id=getattr(getattr(response, "x_groq", None), "id", None),
            finish_reason=response.choices[0].finish_reason,
            usage=chat_usage(response.usage),
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
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

    provider = "custom"

    def __init__(self):
        self.base_url = settings.CUSTOM_MODEL_URL
        self.model = settings.CUSTOM_MODEL_NAME
        self.timeout = settings.CUSTOM_MODEL_TIMEOUT

    def _format_prompt(self, prompt: str, system_prompt: str | None = None) -> str:
        """Format prompt for Mistral Instruct model."""
        max_chars = self.MAX_PROMPT_TOKENS * self.CHARS_PER_TOKEN

        if system_prompt:
            system_short = (
                system_prompt[:300] if len(system_prompt) > 300 else system_prompt
            )
            combined = f"{system_short}\n\n{prompt}"
        else:
            combined = prompt

        if len(combined) > max_chars:
            combined = combined[:max_chars]

        return f"[INST] {combined} [/INST]"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ModelResult:
        """Generate response using custom vLLM model."""
        import httpx

        max_tokens = min(max_tokens, self.MAX_OUTPUT_TOKENS)
        formatted_prompt = self._format_prompt(prompt, system_prompt)

        payload = {
            "model": self.model,
            "prompt": formatted_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        async with httpx.AsyncClient(timeout=float(self.timeout)) as client:
            response = await client.post(
                f"{self.base_url}/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            result = response.json()
            return ModelResult(
                text=result["choices"][0].get("text", "").strip(),
                provider=self.provider,
                configured_model=self.model_name,
                model=result.get("model"),
                provider_request_id=response.headers.get("x-request-id"),
                finish_reason=result["choices"][0].get("finish_reason"),
                usage=chat_usage(result.get("usage")),
            )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Generate streaming response (fallback to non-streaming)."""
        full_response = await self.generate(
            prompt, system_prompt, max_tokens, temperature
        )

        words = full_response.text.split()
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

    @property
    def model_name(self) -> None:
        return None

    def execution_identity(self, prompt: str) -> tuple[str | None, str | None]:
        selected = self.preprocessor if len(prompt) > 2000 else self.interviewer
        return selected.provider, selected.model_name

    def __init__(self):
        # Initialize both services
        self.preprocessor = GroqService()  # For heavy lifting
        self.interviewer = CustomModelService()  # For interview execution

        # Storage for prepared context
        self._prepared_contexts: dict[str, dict] = {}

    async def prepare_interview_context(
        self,
        session_id: str,
        resume_text: str,
        mode: str,
        num_questions: int,
    ) -> dict:
        """
        Phase 1: Use Groq to prepare all interview materials.

        This creates:
        1. Condensed resume summary (~200 tokens)
        2. Interview questions tailored to resume
        3. Per-question context for evaluation

        Returns:
            Dict with prepared context to be stored in session
        """

        # Step 1: Generate condensed resume summary
        summary_prompt = f"""Analyze this resume and create a CONDENSED summary (max 200 words) focusing on:
1. Key technical skills
2. Most impressive achievements with metrics
3. Notable projects
4. Years of experience and domain

Resume:
{resume_text[:4000]}

Provide ONLY the summary, no explanations:"""

        records = []
        summary_call = await call_model(
            self.preprocessor,
            operation="resume_summary",
            prompt_version="hybrid-summary-v1",
            prompt=summary_prompt,
            system_prompt="You are a resume analyzer. Be extremely concise.",
            max_tokens=300,
            temperature=0.3,
        )

        resume_summary = self._prepared_text(summary_call, records)

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

Question prompt version: hybrid-questions-v2.
Return ONLY JSON: {{"questions": ["question text", "another question"]}}"""

        questions_call = await call_model(
            self.preprocessor,
            operation="question_generation",
            prompt_version="hybrid-questions-v2",
            output_schema_version=MODEL_OUTPUT_SCHEMA_VERSION,
            prompt=questions_prompt,
            system_prompt=f"You are an expert {mode} interviewer. Generate specific, probing questions.",
            max_tokens=1500,
            temperature=0.8,
        )

        # Parse questions
        parsed = questions_call.failure or parse_questions(
            questions_call.text, expected_count=num_questions
        )
        questions_call.record(
            "valid" if isinstance(parsed, QuestionBatch) else parsed, records
        )
        if not isinstance(parsed, QuestionBatch):
            raise ValueError("Invalid prepared question output")
        questions = parsed.questions

        # Step 3: Create per-question evaluation context
        question_contexts = []
        for question in questions:
            # Create a compact context for each question
            context_prompt = f"""For this interview question, extract the KEY FACTS from the resume that are relevant.

Question: {question}

Resume Summary:
{resume_summary}

List 3-5 bullet points of relevant facts the candidate should mention. Be concise:"""

            context_call = await call_model(
                self.preprocessor,
                operation="question_context",
                prompt_version="hybrid-question-context-v1",
                prompt=context_prompt,
                system_prompt="Extract only the most relevant facts. Be very concise.",
                max_tokens=200,
                temperature=0.3,
            )
            question_contexts.append(self._prepared_text(context_call, records))

        # Store the prepared context
        prepared = {
            "session_id": session_id,
            "resume_summary": resume_summary,
            "model_executions": records,
            "questions": questions,
            "question_contexts": question_contexts,
            "mode": mode,
        }

        self._prepared_contexts[session_id] = prepared

        return prepared

    @staticmethod
    def _prepared_text(call, records: list[dict]) -> str:
        outcome = call.failure or (
            "valid" if call.text.strip() else InvalidOutput(code="schema_violation")
        )
        call.record(outcome, records)
        if outcome != "valid":
            raise RuntimeError("Interview preparation unavailable")
        return call.text.strip()

    def get_prepared_context(self, session_id: str) -> dict | None:
        """Get prepared context for a session."""
        return self._prepared_contexts.get(session_id)

    def set_prepared_context(self, session_id: str, context: dict):
        """Set prepared context for a session (e.g., loaded from session store)."""
        self._prepared_contexts[session_id] = context

    async def evaluate_answer(
        self,
        session_id: str,
        question_index: int,
        question: str,
        answer: str,
        conversation_history: list[dict] = None,
    ) -> dict:
        """Legacy compact evaluation is explicitly unavailable under M2 contracts."""
        return UnsupportedMode().model_dump(mode="json")

    async def generate_followup(
        self,
        session_id: str,
        question: str,
        answer: str,
        gap_type: str,
        conversation_history: list[dict] = None,
    ) -> dict:
        """No follow-up judgment is manufactured for an unsupported evaluator."""
        return UnsupportedMode().model_dump(mode="json")

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ModelResult:
        """
        Default generate method - routes to appropriate service.

        For long prompts (>2000 chars): Use Groq
        For short prompts: Use Custom Model
        """
        if len(prompt) > 2000:
            return await self.preprocessor.generate(
                prompt, system_prompt, max_tokens, temperature
            )
        else:
            return await self.interviewer.generate(
                prompt, system_prompt, min(max_tokens, 300), temperature
            )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Generate streaming response."""
        if len(prompt) > 2000:
            async for chunk in self.preprocessor.generate_stream(
                prompt, system_prompt, max_tokens, temperature
            ):
                yield chunk
        else:
            async for chunk in self.interviewer.generate_stream(
                prompt, system_prompt, min(max_tokens, 300), temperature
            ):
                yield chunk

    def _parse_questions(self, response: str, expected_count: int) -> list[str]:
        """Apply the same question contract used by the graph."""
        result = parse_questions(response, expected_count=expected_count)
        if not isinstance(result, QuestionBatch):
            raise ValueError("Invalid prepared question output")
        return result.questions


class LLMServiceFactory:
    """Factory to create LLM service based on configuration."""

    _instance: BaseLLMService | None = None
    _hybrid_instance: HybridModelService | None = None

    @classmethod
    def get_service(cls, provider: str | None = None) -> BaseLLMService:
        """Get LLM service. If provider is specified, creates a fresh instance (not cached)."""
        if provider and provider != settings.LLM_PROVIDER:
            return cls._create_service(provider_override=provider)
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
    def _create_service(cls, provider_override: str | None = None) -> BaseLLMService:
        provider = provider_override or settings.LLM_PROVIDER
        # API mode
        if provider == "anthropic":
            if not settings.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not set")
            return AnthropicService()

        elif provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set")
            return OpenAIService()

        elif provider == "gemini":
            if not settings.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY not set")
            return GeminiService()

        elif provider == "groq":
            if not settings.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not set")
            return GroqService()

        elif provider == "custom":
            if not settings.CUSTOM_MODEL_URL:
                raise ValueError("CUSTOM_MODEL_URL not set")
            return CustomModelService()

        elif provider == "hybrid":
            return cls.get_hybrid_service()

        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    @classmethod
    def reset(cls):
        cls._instance = None
        cls._hybrid_instance = None


def get_llm_service() -> BaseLLMService:
    """Get the configured LLM service."""
    return LLMServiceFactory.get_service()
