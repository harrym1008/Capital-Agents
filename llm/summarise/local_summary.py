from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from openai import OpenAI, APIStatusError, APIConnectionError
from enum import Enum
import math

from llm.llm_client import BaseLLMClient, ResponsePrintMode
from llm.llamacpp.llamacpp_init import LlamaCppProcessInitiator
from llm.llamacpp.llamacpp_args import LlamaCppModel


class SummaryType(Enum):
    GENERAL = 1
    EARNINGS_REPORT = 2


SYSTEM_PROMPTS = {
    SummaryType.GENERAL: """You are an expert summarisation assistant.
    
Produce accurate, concise summaries while preserving the most important information.

Rules:
- Output a short title of the full text (3-5 words, no punctuation) followed by exactly {bulletPoints} bullet points.
- Each bullet point should be a complete sentence or two.
- Use no more than {targetWords} words in total.
- Keep important facts, names, dates, numbers, conclusions etc.
- Do not add information that is not present in the original text.
- Rewrite in your own words whenever possible.
- Do not include any markdown formatting - except '-', use these for the bullet points.
- Return only the title, the bullet points and nothing else.""",

    SummaryType.EARNINGS_REPORT: """You are a financial document summarisation assistant.

Your task is to produce concise, factual summaries of the provided section from a company's earnings report or SEC filing.

Rules:
- Output a short title of the full text (3-5 words, no punctuation) followed by exactly {bulletPoints} bullet points.
- Each bullet point should be a complete sentence or two.
- Use no more than {targetWords} words in total.
- Preserve all important quantitative information, including percentages, dollar amounts, dates, growth rates, margins, and guidance.
- Focus only on material facts, business performance, strategy, risks, outlook, operations, and notable events.
- Do not speculate or infer information that is not explicitly stated.
- Ignore boilerplate, legal language, repetition, and stylistic wording unless it contains material information.
- Do not omit significant negative information.
- Do not include any markdown formatting - except '-', use these for the bullet points.
- Rewrite information in clear, concise language instead of copying sentences verbatim.
- If the section contains little useful information, summarise the key facts without inventing details.
- Return only the title, the bullet points and nothing else."""

}


class LlamaCppSummaryClient(BaseLLMClient):
    def __init__(self, thinkingBudget: int = 256):
        self.thinkingBudget = thinkingBudget
        self.processInitiator = LlamaCppProcessInitiator(
            serverName="summary", 
            model=LlamaCppModel.SUMMARY_MODEL, 
            printLogsToTerminal=False,
            killExistingProcesses=False
        )

        startThread = self.processInitiator.startOnAnotherThread()
        startThread.join()  
        
        super().__init__(defaultModel="model", allowParallel=True)

    def _createOpenaiClient(self):
        return OpenAI(base_url=self.processInitiator.apiUrl, api_key="xyz")  # API key is unused
    
    def _getExtraBody(self, thinkingBudget: Optional[int] = None):
        if self.thinkingBudget is None:
            return {"reasoning": {"enabled": False}}
        return {
            "thinking_budget_tokens": self.thinkingBudget,
            "reasoning_budget": self.thinkingBudget
        }
    
    def summariseText(self, 
                      text: str, 
                      summaryType: SummaryType, 
                      targetWords: int = 100, 
                      bulletPoints: int = 5,
                      iteration: int = 1):
        if iteration > 5:
            raise RuntimeError("Exceeded maximum summarisation iterations. Text is too long!")        

        maxTokens = self.thinkingBudget + (targetWords * 1.8)   # Rule of thumb is 1 token = 0.75 words... allow a bit of wiggle room for longer responses
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
            stream = self.openaiClient.chat.completions.create(
                model=self.defaultModel,
                messages=messages,
                max_tokens=maxTokens,
                temperature=0.15,
                extra_body=self._getExtraBody(),
                stream=True
            )
            content, _, _ = self.handleResponseStream(stream, responsePrint=ResponsePrintMode.SILENT)
            return content.strip()
        

        except APIStatusError as e:
            errorBody = e.body
            errorType = errorBody["type"]
            if errorType != "exceed_context_size_error":
                raise e
            
            contextWindow = errorBody["n_ctx"]
            promptTokens = errorBody["n_prompt_tokens"]

            splits = int(math.ceil(promptTokens / (contextWindow * 0.9)))   # Split into chunks that are 90% of the context window
            splitSize = int(math.ceil(len(text) / splits))
            splitTexts = [text[i:i + splitSize] for i in range(0, len(text), splitSize)]

            summaries = [None] * len(splitTexts)

            self._safePrint(f"[Iteration {iteration}] Text exceeds context window ({promptTokens} > {contextWindow}). Splitting into {len(splitTexts)} chunks and summarising in parallel...")
            with ThreadPoolExecutor(max_workers=len(splitTexts)) as executor:
                futureToIndex = {
                    executor.submit(
                        self.summariseText, 
                        text=splitTexts[i], 
                        summaryType=summaryType, 
                        targetWords=targetWords, 
                        bulletPoints=bulletPoints,
                        iteration=iteration + 1
                    ): i for i in range(len(splitTexts))
                }
                for future in as_completed(futureToIndex):
                    index = futureToIndex[future]
                    summaries[index] = future.result()
            
            # Summarise the joined summaries into a final summary
            print(f"[Iteration {iteration}] Summarising {len(summaries)} chunk summaries into a final summary (iteration {iteration})...")
            finalSummary = self.summariseText(
                text="\n".join(summaries), 
                summaryType=summaryType, 
                targetWords=targetWords, 
                bulletPoints=bulletPoints,
                iteration=iteration + 1
            )
            return finalSummary


        except Exception as e:
            print(f"Unexpected error: {e}")
            return f"Failed to summarise text due to an unexpected error:\n{e}"
          
            
    
    def stop(self):
        self.processInitiator.stop()