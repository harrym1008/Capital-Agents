import time
from typing import Optional, Any

from openai import OpenAI

from llm.llm_client import BaseLLMClient
from cli.ansi import ANSI
from collectors.rate_limiter import RateLimiter


class OpenRouterClient(BaseLLMClient):
    def __init__(self, apiKey: str, model: str, providerRouter: Optional[str] = None, costTracker: Optional[Any] = None):
        self.apiKey = apiKey
        self.providerRouter = providerRouter
        super().__init__(defaultModel=model, allowParallel=True, costTracker=costTracker)
        if model.endswith(":free"):
            self.rateLimiter = RateLimiter("openrouter", 20, 60)  # 20 requests per minute max for free tier
        else:
            self.rateLimiter = RateLimiter("openrouter", 10, 1)  # No limit for paid tier (10 a second is safe)

    def _createOpenaiClient(self) -> OpenAI:
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.apiKey,
            default_headers={
                "X-Title": "CapitalAgents"
            }
        )
    
    def _getStreamOptions(self):
        return {"include_usage": True}      # Ask OpenRouter to include usage stats

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

        if self.providerRouter and self.providerRouter.strip() and self.providerRouter.strip().lower() != "auto":
            extraBody["provider"] = {
                "order": [self.providerRouter.strip()],
                "allow_fallbacks": True
            }
        
        return extraBody
    

    def _applyRateLimit(self):
        if self.rateLimiter is not None:
            waitTime = self.rateLimiter.getWaitTime()
            if waitTime > 0.05:
                print(f"{ANSI.DIM}[Rate Limited by OpenRouter - waiting {waitTime:.1f} seconds]{ANSI.RESET}")
                time.sleep(waitTime)
