import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from enum import Enum

from openai import OpenAI
import pandas as pd

from cli.ansi import ANSI
from llm.tools.tool_registry import ToolRegistry
from llm.token_cost_tracker import TokenCostTracker

from ui.ui_hooks import (
    emitEvent, getCurrentAgent, setCurrentAgent,
    getAgentPhase, setAgentPhase, getCurrentStage, setCurrentStage,
    isStopRequested, SimulationStoppedException
)


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
    def __init__(self, defaultModel: str, allowParallel: bool = False, costTracker: Optional[TokenCostTracker] = None):
        self.defaultModel = defaultModel
        self.openaiClient: OpenAI = self._createOpenaiClient()
        self.toolCallLock = threading.Lock()
        
        self.allowParallel = allowParallel
        self.printLock = threading.Lock() if allowParallel else None

        if costTracker is not None:
            self.costTracker = costTracker
        else:
            from llm.server_manager import serverManager
            self.costTracker = serverManager.costTracker


    @abstractmethod
    def _createOpenaiClient(self) -> OpenAI:
        pass

    def _getExtraBody(self, thinkingBudget: Optional[int] = None) -> Dict[str, Any]:
        return {}

    def _getStreamOptions(self) -> Optional[Dict[str, Any]]:
        return None

    def _applyRateLimit(self):
        pass

    def newTask(self):
        return self.costTracker.newTask()

    def _safePrint(self, *args, **kwargs):
        if self.allowParallel:
            with self.printLock:
                print(*args, **kwargs)
        else:
            print(*args, **kwargs)


    def handleResponseStream(self, 
            responseStream, 
            responsePrint: ResponsePrintMode = ResponsePrintMode.FULL):
        
        fullContent = ""
        fullReasoning = ""
        toolCallsList = []
        isThinking = False
        
        isToolCallStreaming = False
        currentState = "idle"
        lastUsage = None

        for chunk in responseStream:
            if isStopRequested():
                raise SimulationStoppedException("Simulation stopped by user.")

            # Final stream chunks carries usage statistics
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                lastUsage = usage

            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # 1. Capture reasoning content tokens
            reasoningChunk = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None) 
            if reasoningChunk:                              # ^^^^  Support both llamacpp and OpenRouter naming conventions
                if isinstance(reasoningChunk, dict):
                    reasoningChunk = reasoningChunk.get("text", "")     # OpenRouter might return reasoning as a dict with a "text" key

                fullReasoning += reasoningChunk

                # Emit reasoning token
                if currentState != "reasoning":
                    if currentState == "content":
                        emitEvent("contentEnd")
                    emitEvent("reasoningStart", {"phase": getAgentPhase()})
                    currentState = "reasoning"
                emitEvent("reasoningToken", {"token": reasoningChunk, "phase": getAgentPhase()})

                if responsePrint.printThinking():
                    if not isThinking:
                        self._safePrint(f"\n{ANSI.DIM}[Thinking]: ", end="", flush=True)
                        isThinking = True
                    self._safePrint(reasoningChunk, end="", flush=True)
                elif responsePrint == ResponsePrintMode.ONE_TOKEN_ONLY:
                    self._safePrint(f"{re.sub(r'[\x00-\x1F\x7F]', '', reasoningChunk)}                 ", end="\r", flush=True)

            # 2. Capture regular response text tokens
            contentChunk = getattr(delta, "content", None)
            if contentChunk:
                fullContent += contentChunk

                # Emit content token
                if currentState != "content":
                    if currentState == "reasoning":
                        emitEvent("reasoningEnd")
                    emitEvent("contentStart", {"phase": getAgentPhase()})
                    currentState = "content"
                emitEvent("contentToken", {"token": contentChunk, "phase": getAgentPhase()})

                if responsePrint.printResponse():
                    if isThinking:
                        self._safePrint(f"\n{ANSI.RESET}[Response]: ", end="", flush=True)
                        isThinking = False
                    elif fullContent == "":
                        self._safePrint("\n[Response]: ", end="", flush=True)
                    self._safePrint(contentChunk, end="", flush=True)
                elif responsePrint == ResponsePrintMode.ONE_TOKEN_ONLY:
                    self._safePrint(f"{re.sub(r'[\x00-\x1F\x7F]', '', contentChunk)}                 ", end="\r", flush=True)

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
                        self._safePrint(f"\n\n{ANSI.DIM}[Tool Calls]: ", end="", flush=True)
                        isToolCallStreaming = True

                    currentCall = toolCallsList[index]
                    if getattr(toolCallDelta, "id", None):
                        currentCall["id"] += toolCallDelta.id
                    if getattr(toolCallDelta, "function", None):
                        funcDelta = toolCallDelta.function
                        if getattr(funcDelta, "name", None):
                            if not currentCall["function"]["name"]:
                                self._safePrint(f"\n{ANSI.RESET}{ANSI.BOLD}[Tool #{index}]: {funcDelta.name} -> ", end="", flush=True)
                            else:
                                self._safePrint(funcDelta.name, end="", flush=True)
                            currentCall["function"]["name"] += funcDelta.name
                            emitEvent("toolCallStreamStart", {
                                "index": index,
                                "toolName": currentCall["function"]["name"]
                            })
                        if getattr(funcDelta, "arguments", None):
                            self._safePrint(funcDelta.arguments, end="", flush=True)
                            currentCall["function"]["arguments"] += funcDelta.arguments
                            emitEvent("toolCallStreamToken", {
                                "index": index,
                                "token": funcDelta.arguments
                            })

        # Close active streaming states at the end of the response stream
        # Wrap in try/except so cleanup events don't crash during a stop
        try:
            if currentState == "reasoning":
                emitEvent("reasoningEnd")
            elif currentState == "content":
                emitEvent("contentEnd")
        except SimulationStoppedException:
            pass

        if ((fullContent and responsePrint.printResponse()) or 
            (fullReasoning and responsePrint.printThinking())) and len(toolCallsList) == 0:
            self._safePrint(ANSI.RESET)
        else:
            self._safePrint(ANSI.RESET, end="")

        if len(toolCallsList) > 1:
            self._safePrint()

        return fullContent, fullReasoning, toolCallsList, lastUsage


    def executeSingleToolCall(
        self,
        currentToolCall: Dict[str, Any], 
        toolRegistry: ToolRegistry, 
        timestamp: pd.Timestamp,
        agentRole=None, 
        agentColor=None, 
        stageNum=0, 
        toolIndex=0
    ):
        if agentRole:
            setCurrentAgent(agentRole, agentColor)
            setCurrentStage(stageNum)

        funcName = currentToolCall["function"]["name"]
        funcArgsString = currentToolCall["function"]["arguments"]
        callId = currentToolCall["id"]

        emitEvent("toolCallStart", {
            "toolName": funcName,
            "args": funcArgsString,
            "callId": callId,
            "toolIndex": toolIndex
        })

        stdoutOutput = ""
        variablesOutput = {}

        try:
            funcArgsDict = json.loads(funcArgsString)
        except json.JSONDecodeError:
            # Failed to parse the tool's arguments 
            funcArgsDict = {}

        if funcName in toolRegistry.tools:
            toolCalled = toolRegistry.tools[funcName]
            try:
                toolResult = toolRegistry.executeTool(funcName, timestamp, funcArgsDict)
                stringResult = json.dumps(toolResult)
                
                with self.toolCallLock:                    
                    self._safePrint(f"{ANSI.BOLD} Executing {funcName} --> {funcArgsDict}", end="")
                    if "error" in toolResult:
                        self._safePrint(f" {ANSI.BOLD}{ANSI.RED}... failed: {toolResult['error']}  {ANSI.RESET}", flush=True)
                    else:
                        self._safePrint(f" {ANSI.BOLD}{ANSI.GREEN}... done.  {ANSI.RESET}", flush=True)

                if toolCalled.name == "executePythonCalculation":
                    with self.toolCallLock:
                        output = toolCalled.toolLog.pop()
                        success = output.get("success", False)

                        if not success:
                            error = output.get("error", "Unknown error")
                            self._safePrint(f"{ANSI.BOLD}{ANSI.RED}Python Execution Error: {error}{ANSI.RESET}")
                        else:
                            stdoutOutput = output.get("stdout", "No stdout captured.")
                            variablesOutput = output.get("variables", {})

                            self._safePrint(
                                f"\n{ANSI.BOLD}Python Execution Output: {ANSI.RESET}"
                                f"\n{ANSI.DIM}{stdoutOutput}{ANSI.RESET}\n"
                                f"{ANSI.BOLD}\nPython Execution Variables: {ANSI.RESET}")
                            for k, v in variablesOutput.items():
                                self._safePrint(f"{ANSI.DIM}{k}: {ANSI.RESET}{v}")
                            self._safePrint()

            
            except SimulationStoppedException:
                raise
            except Exception as e:
                stringResult = json.dumps({"error": f"{e.__class__.__name__}: {e}"})
                with self.toolCallLock:
                    self._safePrint(f"{ANSI.BOLD} Executing {funcName} --> {funcArgsDict}", end="")
                    self._safePrint(f" {ANSI.BOLD}{ANSI.RED}... failed: {e.__class__.__name__}: {e}  {ANSI.RESET}", flush=True)
        else:
            stringResult = json.dumps({"error": f"Tool {funcName} doesn't exist or not accessible by this agent."})

        # Determine tool status
        status = "success"
        try:
            resParsed = json.loads(stringResult)
            if isinstance(resParsed, dict) and "error" in resParsed:
                status = "error"
        except Exception:
            pass

        emitEvent("toolCallEnd", {
            "toolName": funcName,
            "callId": callId,
            "status": status,
            "result": stringResult,
            "stdout": stdoutOutput,
            "variables": variablesOutput,
            "toolIndex": toolIndex
        })

        return callId, stringResult, status


    def runConversation(self, 
            messageHistory: List[Dict[str, Any]], 
            toolRegistry: ToolRegistry,
            timestamp: pd.Timestamp,
            thinkingBudget: Optional[int] = None,
            responsePrint: ResponsePrintMode = ResponsePrintMode.FULL,
            requireInitialTools: bool = False
        ):
        toolSchemas = [tool.getToolSchema() for tool in toolRegistry.tools.values()] if toolRegistry else []
        maxIterations = 10
        currentIteration = 0
        accumulatedContent = ""

        while currentIteration < maxIterations:
            currentIteration += 1

            if currentIteration == 1 and requireInitialTools and toolSchemas:
                toolChoiceSetting = "required"
                # Thinking mode on certain providers (e.g. OpenRouter / Alibaba / Qwen) does not support tool_choice="required"
                if thinkingBudget is not None and thinkingBudget > 0:
                    toolChoiceSetting = "auto"
            else:
                toolChoiceSetting = "auto" if toolSchemas else None

            self._applyRateLimit()
            responseKwargs = dict(
                model=self.defaultModel,
                messages=messageHistory,
                temperature=0.5,
                tools=toolSchemas if toolSchemas else None,
                tool_choice=toolChoiceSetting,
                max_tokens=8192,
                stream=True,
            )
            streamOptions = self._getStreamOptions()
            if streamOptions is not None:
                responseKwargs["stream_options"] = streamOptions

            extraBody = self._getExtraBody(thinkingBudget) if thinkingBudget is not None else None
            if extraBody is not None:
                responseKwargs["extra_body"] = extraBody

            try:
                responseStream = self.openaiClient.chat.completions.create(**responseKwargs)
            except Exception as reqErr:
                errStr = str(reqErr)
                if "tool_choice" in errStr and responseKwargs.get("tool_choice") == "required":
                    responseKwargs["tool_choice"] = "auto"
                    responseStream = self.openaiClient.chat.completions.create(**responseKwargs)
                else:
                    raise reqErr
            content, reasoning, toolCallsList, usage = self.handleResponseStream(responseStream, responsePrint)
            self.costTracker.recordUsage(usage)

            if content and content.strip():
                accumulatedContent += content + "\n"

            if not toolCallsList:
                return accumulatedContent.strip()
            
            assistantMessageDict = {
                "role": "assistant",
                "content": content or "",
                "tool_calls": toolCallsList
            }
            if reasoning:
                if self.__class__.__name__ == "LlamaCppClient":
                    assistantMessageDict["reasoning_content"] = reasoning
                else:
                    assistantMessageDict["reasoning"] = reasoning

            messageHistory.append(assistantMessageDict)
            
            parentAgent = getCurrentAgent()
            agentRole = parentAgent.get("role")
            agentColor = parentAgent.get("color")
            currentStageNum = getCurrentStage()


            resultsByIndex = [None] * len(toolCallsList)
            with ThreadPoolExecutor(max_workers=len(toolCallsList)) as executor:
                futureToIndex =  {executor.submit(
                    self.executeSingleToolCall, 
                    call, toolRegistry, timestamp, agentRole, agentColor, currentStageNum, idx): idx
                                  for idx, call in enumerate(toolCallsList)}
                for future in as_completed(futureToIndex):
                    idx = futureToIndex[future]
                    try:
                        toolCallId, stringResult, status = future.result()
                    except SimulationStoppedException:
                        # Cancel remaining futures and propagate the stop
                        for f in futureToIndex:
                            f.cancel()
                        raise
                    resultsByIndex[idx] = (toolCallId, stringResult, status)

                for entry in resultsByIndex:
                    if entry is None:
                        continue
                    toolCallId, stringResult, status = entry
                    messageHistory.append({
                        "role": "tool",
                        "tool_call_id": toolCallId,
                        "content": stringResult
                    })

                # Check for 'confirmBoardroomDecision' tool call and handle it
                for toolCall in toolCallsList:
                    if toolCall["function"]["name"] == "confirmBoardroomDecision":
                        # Find this tool call's id from messageHistory and if its status is 'success' assume the agent made the decision
                        for msg in messageHistory:
                            if msg.get("role") == "tool" and msg.get("tool_call_id") == toolCall["id"]:
                                if msg.get("content"):
                                    try:
                                        resultData = json.loads(msg["content"])
                                        if isinstance(resultData, dict) and resultData.get("status") == "success":
                                            # Agent has made a decision, return the accumulated content
                                            return accumulatedContent.strip()
                                    except json.JSONDecodeError:
                                        pass
                                break

        # If this code is reached, it means the maximum number of iterations was reached without a final response
        messageHistory.append({
            "role": "user",
            "content": "You have reached the maximum number of iterations without providing a final response. Please provide your final response based on the accumulated information."
        })

        self._applyRateLimit()
        finalResponseKwargs = dict(
            model=self.defaultModel,
            messages=messageHistory,
            tools=[],
            temperature=0.5,
            max_tokens=8192,
            stream=True,
        )
        finalStreamOptions = self._getStreamOptions()
        if finalStreamOptions is not None:
            finalResponseKwargs["stream_options"] = finalStreamOptions

        finalExtraBody = self._getExtraBody(thinkingBudget) if thinkingBudget is not None else None
        if finalExtraBody is not None:
            finalResponseKwargs["extra_body"] = finalExtraBody
            
        finalResponseStream = self.openaiClient.chat.completions.create(**finalResponseKwargs)
        finalContent, _, _, finalUsage = self.handleResponseStream(finalResponseStream, responsePrint)
        self.costTracker.recordUsage(finalUsage)
        
        if finalContent and finalContent.strip():
            accumulatedContent += finalContent

        return accumulatedContent.strip()