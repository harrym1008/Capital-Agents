from abc import ABC, abstractmethod
import json
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI
from enum import Enum

from cli.ansi import ANSI
from llm.tools_registry import Tool


class ResponsePrintMode(Enum):
    FULL = "full"
    ONLY_RESPONSE = "only_response"
    SILENT = "silent"
    ONE_TOKEN_ONLY = "one_token_only"

    def printThinking(self):
        return self == ResponsePrintMode.FULL
    
    def printResponse(self):
        return self in {ResponsePrintMode.FULL, ResponsePrintMode.ONLY_RESPONSE}


class BaseLLMClient(ABC):
    def __init__(self, defaultModel: str):
        self.defaultModel = defaultModel
        self.openaiClient = self._createOpenaiClient()


    @abstractmethod
    def _createOpenaiClient(self) -> OpenAI:
        pass

    def _getExtraBody(self, thinkingBudget: Optional[int] = None) -> Dict[str, Any]:
        return {}
    
    def _applyRateLimit(self):
        pass


    def handleResponseStream(self, 
            responseStream, 
            responsePrint: ResponsePrintMode = ResponsePrintMode.FULL):
        
        fullContent = ""
        fullReasoning = ""
        toolCallsList = []
        isThinking = False
        
        isToolCallStreaming = False

        for chunk in responseStream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # 1. Capture reasoning content tokens
            reasoningChunk = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None) 
            if reasoningChunk:                              # ^^^^  Support both llamacpp and OpenRouter naming conventions
                if isinstance(reasoningChunk, dict):
                    reasoningChunk = reasoningChunk.get("text", "")     # OpenRouter might return reasoning as a dict with a "text" key

                fullReasoning += reasoningChunk
                if responsePrint.printThinking():
                    if not isThinking:
                        print(f"\n{ANSI.DIM}[Thinking]: ", end="", flush=True)
                        isThinking = True
                    print(reasoningChunk, end="", flush=True)
                elif responsePrint == ResponsePrintMode.ONE_TOKEN_ONLY:
                    print(f"{re.sub(r'[\x00-\x1F\x7F]', '', reasoningChunk)}                 ", end="\r", flush=True)

            # 2. Capture regular response text tokens
            contentChunk = getattr(delta, "content", None)
            if contentChunk:
                fullContent += contentChunk
                if responsePrint.printResponse():
                    if isThinking:
                        print(f"\n{ANSI.RESET}[Response]: ", end="", flush=True)
                        isThinking = False
                    elif fullContent == "":
                        print("\n[Response]: ", end="", flush=True)
                    print(contentChunk, end="", flush=True)
                elif responsePrint == ResponsePrintMode.ONE_TOKEN_ONLY:
                    print(f"{re.sub(r'[\x00-\x1F\x7F]', '', contentChunk)}                 ", end="\r", flush=True)

            # 3. Assemble fragmented tool call tokens as they arrive
            toolCallsChunk = getattr(delta, "tool_calls", None)
            if toolCallsChunk:
                for toolCallDelta in toolCallsChunk:
                    index = toolCallDelta.index
                    while len(toolCallsList) <= index:
                        toolCallsList.append({
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""}
                        })
                    
                    if not isToolCallStreaming:
                        print(f"\n\n{ANSI.DIM}[Tool Calls]: ", end="", flush=True)
                        isToolCallStreaming = True

                    currentCall = toolCallsList[index]
                    if getattr(toolCallDelta, "id", None):
                        currentCall["id"] += toolCallDelta.id
                    if getattr(toolCallDelta, "function", None):
                        funcDelta = toolCallDelta.function
                        if getattr(funcDelta, "name", None):
                            if not currentCall["function"]["name"]:
                                print(f"\n{ANSI.RESET}{ANSI.BOLD}[Tool #{index}]: {funcDelta.name} -> ", end="", flush=True)
                            else:
                                print(funcDelta.name, end="", flush=True)
                            currentCall["function"]["name"] += funcDelta.name
                        if getattr(funcDelta, "arguments", None):
                            print(funcDelta.arguments, end="", flush=True)
                            currentCall["function"]["arguments"] += funcDelta.arguments

        if ((fullContent and responsePrint.printResponse()) or 
            (fullReasoning and responsePrint.printThinking())) and len(toolCallsList) == 0:
            print(ANSI.RESET)
        else:
            print(ANSI.RESET, end="")

        if len(toolCallsList) > 1:
            print()

        return fullContent, fullReasoning, toolCallsList



    def runConversation(self, 
            messageHistory: List[Dict[str, Any]], 
            availableTools: Optional[List[Tool]] = None, 
            thinkingBudget: Optional[int] = None,
            responsePrint: ResponsePrintMode = ResponsePrintMode.FULL
        ):
        toolSchemas = [tool.getToolSchema() for tool in availableTools] if availableTools else []
        maxIterations = 12
        currentIteration = 0
        accumulatedContent = ""

        while currentIteration < maxIterations:
            currentIteration += 1

            self._applyRateLimit()
            responseStream = self.openaiClient.chat.completions.create(
                model=self.defaultModel,
                messages=messageHistory,
                temperature=0.5,
                tools=toolSchemas,
                tool_choice="auto" if toolSchemas else None,
                max_tokens=8192,
                stream=True,
                extra_body=self._getExtraBody(thinkingBudget) if thinkingBudget is not None else None
            )
            content, reasoning, toolCallsList = self.handleResponseStream(responseStream, responsePrint)

            if content and content.strip():
                accumulatedContent += content + "\n"

            if not toolCallsList:
                return accumulatedContent.strip()
            
            if self.__class__.__name__ == "LlamaCppClient":
                assistantMessageDict = {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning,
                    "tool_calls": toolCallsList
                }
            else:
                assistantMessageDict = {
                    "role": "assistant",
                    "content": content,
                    "reasoning": reasoning,
                    "tool_calls": toolCallsList
                }

            messageHistory.append(assistantMessageDict)
        
            toolMap = {t.toolName: t for t in availableTools} if availableTools else {}

            for currentToolCall in toolCallsList:
                funcName = currentToolCall["function"]["name"]
                funcArgsString = currentToolCall["function"]["arguments"]

                try:
                    funcArgsDict = json.loads(funcArgsString)
                except json.JSONDecodeError:
                    # Failed to parse the tool's arguments 
                    funcArgsDict = {}

                if funcName in toolMap:
                    toolCalled = toolMap[funcName]
                    try:
                        if len(toolCallsList) > 1:
                            print(f"{ANSI.BOLD} Executing {funcName} --> {funcArgsDict}", end="", flush=True)
                        toolResult = toolCalled.executeTool(**funcArgsDict)
                        stringResult = json.dumps(toolResult)
                        if "error" in toolResult:
                            print(f" {ANSI.BOLD}{ANSI.RED}... failed: {toolResult['error']}  {ANSI.RESET}", flush=True)
                        else:
                            print(f" {ANSI.BOLD}{ANSI.GREEN}... done.  {ANSI.RESET}", flush=True)

                        # Output tool's result
                        if toolCalled.toolName == "executePythonCalculation":
                            output = toolCalled.toolLog.pop()
                            print(
                                f"\n{ANSI.BOLD}Python Execution Output: {ANSI.RESET}"
                                f"{ANSI.DIM}{output['stdout']}{ANSI.RESET}\n"
                                f"{ANSI.BOLD}\nPython Execution Variables: {ANSI.RESET}")
                            for k, v in output["variables"].items():
                                print(f"{ANSI.DIM}{k}: {ANSI.RESET}{v}")
                            print()

                    except Exception as e:
                        stringResult = json.dumps({"error": f"{e.__class__.__name__}: {e}"})
                        print(f" {ANSI.BOLD}{ANSI.RED}... failed: {e.__class__.__name__}: {e}  {ANSI.RESET}")
                else:
                    stringResult = json.dumps({"error": f"Tool {funcName} doesn't exist or not accessible by this agent."})

                messageHistory.append({
                    "role": "tool",
                    "tool_call_id": currentToolCall["id"],
                    "content": stringResult
                })

            # Inject a user prompt after all tool results are collected.
            # Without this, the model treats tool results as a passive continuation
            # of its pre-tool plan and often skips re-reasoning over the data.
            # This explicit turn forces a fresh thinking pass grounded in the actual results.
            messageHistory.append({
                "role": "user",
                "content": "All tool results have been returned. Analyse the data above carefully, extract the key figures, and now produce your response."
            })

        # If this code is reached, it means the maximum number of iterations was reached without a final response
        self._applyRateLimit()
        finalResponseStream = self.openaiClient.chat.completions.create(
            model=self.defaultModel,
            messages=messageHistory,
            temperature=0.5,
            max_tokens=8192,
            stream=True,
            extra_body=self._getExtraBody(thinkingBudget) if thinkingBudget is not None else None
        )
        finalContent, _, _ = self.handleResponseStream(finalResponseStream, responsePrint)
        
        if finalContent and finalContent.strip():
            accumulatedContent += finalContent

        return accumulatedContent.strip()