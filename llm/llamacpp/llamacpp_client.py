from openai import OpenAI
from typing import Optional

from llm.llm_client import BaseLLMClient
from llm.llamacpp.llamacpp_init import LlamaCppProcessInitiator


class LlamaCppClient(BaseLLMClient):
    def __init__(self, processInitiator: LlamaCppProcessInitiator):
        self.processInitiator = processInitiator
        super().__init__(defaultModel="model")

    def _createOpenaiClient(self) -> OpenAI:
        return OpenAI(base_url=self.processInitiator.apiUrl, api_key="xyz")  # API key is unused
    
    def _getExtraBody(self, thinkingBudget: Optional[int] = None):
        extraBody = {}
        if thinkingBudget is not None:
            extraBody["thinking_budget_tokens"] = thinkingBudget
            extraBody["reasoning_budget"] = thinkingBudget            

            if thinkingBudget <= 256:
                reasoningEffort = "low"
            elif thinkingBudget <= 2048:
                reasoningEffort = "medium"
            else:
                reasoningEffort = "high"
            extraBody["reasoning_effort"] = reasoningEffort
            
        return extraBody
    
    def _applyRateLimit(self):
        pass    # No local rate limiting