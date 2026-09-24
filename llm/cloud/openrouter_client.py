import time
from typing import Optional, Any

from openai import OpenAI

from llm.llm_client import BaseLLMClient
from collectors.rate_limiter import RateLimiter


# Client wrapper for OpenRouter API with rate-limiting and thinking budget controls
class OpenRouterClient(BaseLLMClient):
    def __init__(self, apiKey: str, model: str, providerRouter: Optional[str] = None, costTracker: Optional[Any] = None):
        self.apiKey = apiKey
        self.providerRouter = providerRouter
        super().__init__(defaultModel=model, allowParallel=True, costTracker=costTracker)
        if model.endswith(":free"):
            self.rateLimiter = RateLimiter("openrouter", 20, 60)    # 20 requests per minute max for free tier
        else:
            self.rateLimiter = RateLimiter("openrouter", 10, 1)     # No limit for paid tier (10 a second is safe)

    def _createOpenaiClient(self) -> OpenAI:
        # Initialise OpenAI client configured for OpenRouter base endpoint
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.apiKey,
            default_headers={
                "X-Title": "CapitalAgents"
            }
        )
    
    def _getStreamOptions(self):
        return {"include_usage": True}   # Explicitly request usage stats


    def _getExtraBody(self, thinkingBudget: Optional[int] = None):
        # Format reasoning tokens configuration and upstream provider routing preferences
        extraBody = {}
        if thinkingBudget is not None:
            if thinkingBudget <= 0:
                extraBody["reasoning"] = {
                    "enabled": False,
                    "max_tokens": 0
                }
                extraBody["thinking"] = {
                    "type": "disabled"
                }
                extraBody["chat_template_kwargs"] = {
                    "enable_thinking": False
                }
            else:
                extraBody["reasoning"] = {
                    "enabled": True,
                    "max_tokens": thinkingBudget
                }
                extraBody["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinkingBudget
                }
                extraBody["chat_template_kwargs"] = {
                    "enable_thinking": True,
                    "thinking_budget": thinkingBudget
                }

        # Apply specific provider routing constraint
        if self.providerRouter and self.providerRouter.strip() and self.providerRouter.strip().lower() != "auto":
            extraBody["provider"] = {
                "order": [self.providerRouter.strip()],
                "allow_fallbacks": True
            }
        
        return extraBody
    

    def _applyRateLimit(self):
        # Block thread if current call volume exceeds rate limit bucket
        if self.rateLimiter is not None:
            waitTime = self.rateLimiter.getWaitTime()
            if waitTime > 0.05:
                print(f"[Rate Limited by OpenRouter - waiting {waitTime:.1f} seconds]")
                time.sleep(waitTime)
