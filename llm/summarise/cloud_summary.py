import os
import time
from dotenv import load_dotenv
load_dotenv()

from typing import Optional
from openai import OpenAI

from llm.llm_client import BaseLLMClient, ResponsePrintMode
from collectors.rate_limiter import RateLimiter

from llm.summarise.local_summary import SummaryType, SYSTEM_PROMPTS


SUMMARY_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"


class OpenRouterSummaryClient(BaseLLMClient):
    def __init__(self, apiKey: Optional[str] = None):
        if apiKey is None:
            apiKey = os.getenv("OPENROUTER_API_KEY")
        if not apiKey:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

        self.apiKey = apiKey
        self.rateLimiter = RateLimiter("openrouter", 20, 60)  # 20 requests per minute max for free tier

        super().__init__(defaultModel=SUMMARY_MODEL, allowParallel=True)

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
        return {
            "chat_template_kwargs": {
                "enable_thinking": True,
                "low_effort": True
            }
        }

    def _applyRateLimit(self):
        waitTime = self.rateLimiter.getWaitTime()
        if waitTime > 0.05:
            time.sleep(waitTime)
    

    def summariseText(self,
                      text: str,
                      summaryType: SummaryType,
                      targetWords: int = 100,
                      bulletPoints: int = 5):
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPTS[summaryType].format(bulletPoints=bulletPoints, targetWords=targetWords)
            },
            {
                "role": "user",
                "content": text
            }
        ]

        try:
            self._applyRateLimit()
            stream = self.openaiClient.chat.completions.create(
                model=self.defaultModel,
                messages=messages,
                temperature=0.25,
                extra_body=self._getExtraBody(),
                stream=True
            )
            content, _, _ = self.handleResponseStream(stream, responsePrint=ResponsePrintMode.SILENT)
            return content.strip()

        except Exception as e:
            print(f"Unexpected error: {e}")
            return f"Failed to summarise text due to an unexpected error:\n{e}"
