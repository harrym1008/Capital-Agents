from typing import Optional, Any
from openai import OpenAI

from llm.llm_client import BaseLLMClient


# Generic OpenAI-compatible HTTP client connecting to self-hosted or third-party endpoints
class OpenAICompatibleClient(BaseLLMClient):
    def __init__(self, baseUrl: str, apiKey: Optional[str] = None, model: str = "default", costTracker: Optional[Any] = None):
        self.baseUrl = baseUrl.rstrip("/") if baseUrl else ""
        self.apiKey = apiKey if apiKey and apiKey.strip() else "EMPTY"
        super().__init__(defaultModel=model, allowParallel=True, costTracker=costTracker)

    def _createOpenaiClient(self) -> OpenAI:
        return OpenAI(
            base_url=self.baseUrl,
            api_key=self.apiKey,
        )

    def _applyRateLimit(self):
        # To be added later
        pass
