from openai import OpenAI
from typing import Optional, Any

from llm.llm_client import BaseLLMClient
from llm.llamacpp.llamacpp_init import LlamaCppProcessInitiator


# Local LLM client communicating with managed llama-server process
class LlamaCppClient(BaseLLMClient):
    def __init__(self, processInitiator: LlamaCppProcessInitiator, allowParallel=True, costTracker: Optional[Any] = None):
        self.processInitiator = processInitiator
        super().__init__(defaultModel="model", allowParallel=allowParallel, costTracker=costTracker)

    def _createOpenaiClient(self) -> OpenAI:
        # Connect to local llama-server HTTP endpoint
        return OpenAI(base_url=self.processInitiator.apiUrl, api_key="xyz")
    
    def _getExtraBody(self, thinkingBudget: Optional[int] = None):
        # Configure reasoning token budgets and effort levels for llama-server
        extraBody = {}
        if thinkingBudget is not None:
            if thinkingBudget <= 0:
                extraBody["thinking_budget_tokens"] = 0
                extraBody["reasoning_budget"] = 0
                extraBody["reasoning_effort"] = "none"
                extraBody["enable_thinking"] = False
                extraBody["chat_template_kwargs"] = {
                    "enable_thinking": False
                }
            else:
                extraBody["thinking_budget_tokens"] = thinkingBudget
                extraBody["reasoning_budget"] = thinkingBudget            

                if thinkingBudget <= 256:
                    reasoningEffort = "low"
                elif thinkingBudget <= 2048:
                    reasoningEffort = "medium"
                else:
                    reasoningEffort = "high"
                extraBody["reasoning_effort"] = reasoningEffort
                extraBody["chat_template_kwargs"] = {
                    "enable_thinking": True
                }
            
        return extraBody
    
    def _applyRateLimit(self):
        # No rate limiting required for self-hosted local server
        pass