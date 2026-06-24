import time
from typing import Optional

from openai import OpenAI

from llm.llm_client import BaseLLMClient
from cli.ansi import ANSI
from collectors.rate_limiter import RateLimiter


class GroqClient(BaseLLMClient):
    def __init__(self, apiKey: str, model: str = "gpt-oss-120b"):
        self.apiKey = apiKey
        self.rateLimiter = RateLimiter("groq", 30, 60)  # Conservative free-tier limiter
        super().__init__(defaultModel=model)

    def _createOpenaiClient(self) -> OpenAI:
        return OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.apiKey
        )
    
    def _getExtraBody(self, thinkingBudget: Optional[int] = None):
        if thinkingBudget is None:
            return {
                "reasoning_effort": "none"
            }

        if thinkingBudget <= 256:
            reasoningEffort = "low"
        elif thinkingBudget <= 1024:
            reasoningEffort = "medium"
        else:
            reasoningEffort = "high"

        return {
            "reasoning_effort": reasoningEffort
        }

    def _applyRateLimit(self):
        waitTime = self.rateLimiter.getWaitTime()
        if waitTime > 0.05:
            print(f"{ANSI.DIM}[Rate Limited by Groq - waiting {waitTime:.1f} seconds]{ANSI.RESET}")
            time.sleep(waitTime)
