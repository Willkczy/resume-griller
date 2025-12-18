"""
LLM Service Abstraction for Resume Griller.
Supports Claude, OpenAI, Gemini, and Local (LoRA) inference.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from abc import ABC, abstractmethod
from typing import Optional, AsyncIterator
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
        # Combine system prompt with user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Run in thread pool (Gemini SDK is synchronous)
        loop = asyncio.get_event_loop()
        
        def _generate():
            # Update generation config for this request
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
        
        # Get the stream
        response = await loop.run_in_executor(None, _generate_stream)
        
        # Yield chunks
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
    Uses OpenAI-compatible API format.
    
    Requirements:
    - IAP tunnel must be running: 
      gcloud compute start-iap-tunnel instance-20251217-192430 8000 \
          --local-host-port=localhost:8001 --zone=us-east1-b
    """
    
    def __init__(self):
        try:
            from openai import AsyncOpenAI
            
            self.client = AsyncOpenAI(
                base_url=settings.CUSTOM_MODEL_URL,
                api_key="not-needed",  # vLLM doesn't require API key
                timeout=settings.CUSTOM_MODEL_TIMEOUT,
            )
            self.model = settings.CUSTOM_MODEL_NAME
            print(f"Custom Model Service initialized")
            print(f"  URL: {settings.CUSTOM_MODEL_URL}")
            print(f"  Model: {self.model}")
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: uv add openai"
            )
    
    def _format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format prompt for Mistral Instruct model."""
        if system_prompt:
            return f"[INST] {system_prompt}\n\n{prompt} [/INST]"
        else:
            return f"[INST] {prompt} [/INST]"
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate response using custom vLLM model."""
        formatted_prompt = self._format_prompt(prompt, system_prompt)
        
        try:
            response = await self.client.completions.create(
                model=self.model,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            return response.choices[0].text.strip()
            
        except Exception as e:
            print(f"Custom Model error: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Generate streaming response using custom vLLM model."""
        formatted_prompt = self._format_prompt(prompt, system_prompt)
        
        try:
            stream = await self.client.completions.create(
                model=self.model,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices[0].text:
                    yield chunk.choices[0].text
                    
        except Exception as e:
            print(f"Custom Model streaming error: {e}")
            # Fallback to non-streaming
            full_response = await self.generate(prompt, system_prompt, max_tokens, temperature)
            yield full_response

class LocalLLMService(BaseLLMService):
    """Local LoRA model service."""
    
    def __init__(self):
        self.generator = None
        self._loaded = False
    
    def _ensure_loaded(self):
        if not self._loaded:
            from rag.generator import InterviewGenerator
            self.generator = InterviewGenerator(device=settings.LOCAL_MODEL_DEVICE)
            self.generator.load_model()
            self._loaded = True
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        loop = asyncio.get_event_loop()
        
        def _generate():
            self._ensure_loaded()
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            return self.generator.generate(
                prompt=full_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
        
        return await loop.run_in_executor(None, _generate)
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        # Local model doesn't support true streaming
        full_response = await self.generate(prompt, system_prompt, max_tokens, temperature)
        
        words = full_response.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.02)


class LLMServiceFactory:
    """Factory to create LLM service based on configuration."""
    
    _instance: Optional[BaseLLMService] = None
    
    @classmethod
    def get_service(cls) -> BaseLLMService:
        if cls._instance is None:
            cls._instance = cls._create_service()
        return cls._instance
    
    @classmethod
    def _create_service(cls) -> BaseLLMService:
        if settings.LLM_MODE == "local":
            print("Initializing Local LLM Service (LoRA model)")
            return LocalLLMService()
        
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
        
        else:
            raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")
    
    @classmethod
    def reset(cls):
        cls._instance = None


def get_llm_service() -> BaseLLMService:
    """Get the configured LLM service."""
    return LLMServiceFactory.get_service()