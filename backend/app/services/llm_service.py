"""
LLM Service Abstraction for Resume Griller.
Supports both API-based (Claude/OpenAI) and Local (LoRA) inference.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, AsyncIterator
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
        """Generate response using Claude API."""
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
        """Generate streaming response using Claude API."""
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
        """Generate response using OpenAI API."""
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
        """Generate streaming response using OpenAI API."""
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


class LocalLLMService(BaseLLMService):
    """Local LoRA model service using the existing generator."""
    
    def __init__(self):
        self.generator = None
        self._loaded = False
    
    def _ensure_loaded(self):
        """Lazy load the model."""
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
        """Generate response using local LoRA model."""
        # Run in thread pool to not block async loop
        loop = asyncio.get_event_loop()
        
        def _generate():
            self._ensure_loaded()
            # Combine system prompt with user prompt if provided
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
        """
        Local model doesn't support true streaming.
        Simulate streaming by yielding chunks of the full response.
        """
        full_response = await self.generate(prompt, system_prompt, max_tokens, temperature)
        
        # Simulate streaming by yielding word by word
        words = full_response.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.02)  # Small delay to simulate streaming


class LLMServiceFactory:
    """Factory to create appropriate LLM service based on configuration."""
    
    _instance: Optional[BaseLLMService] = None
    
    @classmethod
    def get_service(cls) -> BaseLLMService:
        """Get or create LLM service instance."""
        if cls._instance is None:
            cls._instance = cls._create_service()
        return cls._instance
    
    @classmethod
    def _create_service(cls) -> BaseLLMService:
        """Create LLM service based on settings."""
        if settings.LLM_MODE == "local":
            print("Initializing Local LLM Service (LoRA model)")
            return LocalLLMService()
        
        # API mode
        if settings.LLM_PROVIDER == "anthropic":
            if not settings.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not set in environment")
            print("Initializing Anthropic (Claude) Service")
            return AnthropicService()
        
        elif settings.LLM_PROVIDER == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set in environment")
            print("Initializing OpenAI Service")
            return OpenAIService()
        
        else:
            raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")
    
    @classmethod
    def reset(cls):
        """Reset the service instance (useful for testing)."""
        cls._instance = None


# Convenience function
def get_llm_service() -> BaseLLMService:
    """Get the configured LLM service."""
    return LLMServiceFactory.get_service()