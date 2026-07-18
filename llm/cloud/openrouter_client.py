import time
from typing import Optional

from openai import OpenAI

from llm.llm_client import BaseLLMClient
from cli.ansi import ANSI
from collectors.rate_limiter import RateLimiter


class OpenRouterClient(BaseLLMClient):
    def __init__(self, apiKey: str, model: str):
        self.apiKey = apiKey
        self.rateLimiter = RateLimiter("openrouter", 20, 60)  # 20 requests per minute max for free tier
        super().__init__(defaultModel=model, allowParallel=True)

    def _createOpenaiClient(self) -> OpenAI:
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.apiKey,
            default_headers={
                "HTTP-Referer": "https://localhost:3000",
                "X-Title": "CapitalAgents"
            }
        )
    
    def _getExtraBody(self, thinkingBudget: Optional[int] = None):
        extraBody = {}
        if thinkingBudget is not None:
            if thinkingBudget == 0:
                extraBody = {
                    "reasoning": {
                        "enabled": False
                    }
                }
            else:
                effort = "low" if thinkingBudget < 512 else ("medium" if thinkingBudget < 2048 else "high")
                extraBody = {
                    "reasoning": {
                        "enabled": True,
                        "max_tokens": thinkingBudget,
                        # "effort": effort
                    }
                }
        
        return extraBody    

    def _applyRateLimit(self):
        waitTime = self.rateLimiter.getWaitTime()
        if waitTime > 0.05:
            print(f"{ANSI.DIM}[Rate Limited by OpenRouter - waiting {waitTime:.1f} seconds]{ANSI.RESET}")
            time.sleep(waitTime)
