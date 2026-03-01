"""API model interface for LLM inference."""

import os
import time
import tiktoken
from typing import Optional
import google.generativeai as genai


class APIModel:
    """Wrapper for API-based language models."""

    def __init__(self, model_name: str, provider: str, api_key_env: str):
        """
        Initialize API model.

        Args:
            model_name: Model identifier (e.g., "gemini-2.0-flash-exp")
            provider: API provider ("google", "openai", etc.)
            api_key_env: Environment variable name containing API key
        """
        self.model_name = model_name
        self.provider = provider
        self.api_key = os.getenv(api_key_env)

        if not self.api_key:
            raise ValueError(
                f"API key not found in environment variable: {api_key_env}"
            )

        # Initialize provider
        if provider == "google":
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Token counter (approximate for non-OpenAI models)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> tuple[str, int]:
        """
        Generate text from prompt.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences

        Returns:
            Tuple of (generated_text, token_count)
        """
        if self.provider == "google":
            return self._generate_google(prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _generate_google(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> tuple[str, int]:
        """Generate using Google Gemini API."""
        max_retries = 3
        retry_delay = 2.0

        for attempt in range(max_retries):
            try:
                generation_config = genai.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                )

                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                )

                text = response.text

                # Count tokens (approximate if tokenizer unavailable)
                if self.tokenizer:
                    token_count = len(self.tokenizer.encode(text))
                else:
                    # Rough estimate: ~4 chars per token
                    token_count = len(text) // 4

                return text, token_count

            except Exception as e:
                if attempt < max_retries - 1:
                    # Retry with exponential backoff
                    time.sleep(retry_delay * (2**attempt))
                    continue
                else:
                    raise RuntimeError(
                        f"API call failed after {max_retries} retries: {e}"
                    )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            return len(text) // 4
