import asyncio
from typing import Optional

class LLMClient:
    """
    A mock LLM client for testing and demonstration purposes.
    This class simulates the behavior of a real LLM client without
    requiring an actual LLM connection or API keys.
    It returns pre-defined responses and simulates network latency.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def check_connection(self) -> bool:
        """Simulates a successful connection check."""
        print("Simulating LLM connection check...")
        await asyncio.sleep(0.1)
        return True

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Simulates LLM response generation."""
        print(f"Simulating LLM generation for prompt: {prompt[:50]}...")
        await asyncio.sleep(0.2)
        return f"This is a simulated response to the prompt: '{prompt}'"