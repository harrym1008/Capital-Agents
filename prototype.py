import json
import time
import math
import os

from dotenv import load_dotenv
load_dotenv()

from enum import Enum
from typing import Callable, Any, Dict, List, Optional

from openai import OpenAI

from cli.ansi import ANSI
from llm.llamacpp_args import LlamaCppModel, THINKING_BUDGET, SUMMARISE_THINK_BUDGET
from llm.llamacpp_init import LlamaCppProcessInitiator, killExistingLlamaCppProcesses

from llm.tools_registry import Tool, buildToolsRegistry
from llm.agent_config import FinancialAgentConfig
from llm.agent_prompts import buildResearcherSysPrompt, buildUiFormatSysPrompt, getSystemPersona


class ResponsePrintMode(Enum):
    FULL = "full"
    ONLY_RESPONSE = "only_response"
    SILENT = "silent"

    def printThinking(self):
        return self == ResponsePrintMode.FULL
    
    def printResponse(self):
        return self in {ResponsePrintMode.FULL, ResponsePrintMode.ONLY_RESPONSE}


class LlamaCppClient:
    def __init__(self, processInitiator):
        self.processInitiator = processInitiator
        self.openaiClient = OpenAI(base_url=self.processInitiator.apiUrl, api_key="abcd")

    def handleResponseStream(self, 
            responseStream, 
            responsePrint: ResponsePrintMode = ResponsePrintMode.FULL):
        
        fullContent = ""
        fullReasoning = ""
        toolCallsList = []
        isThinking = False
        
        for chunk in responseStream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # 1. Capture reasoning content tokens
            reasoningChunk = getattr(delta, "reasoning_content", None)
            if reasoningChunk:
                fullReasoning += reasoningChunk
                if responsePrint.printThinking():
                    if not isThinking:
                        print(f"\n{ANSI.DIM}[Thinking]: ", end="", flush=True)
                        isThinking = True
                    print(reasoningChunk, end="", flush=True)

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
                    
                    currentCall = toolCallsList[index]
                    if getattr(toolCallDelta, "id", None):
                        currentCall["id"] += toolCallDelta.id
                    if getattr(toolCallDelta, "function", None):
                        funcDelta = toolCallDelta.function
                        if getattr(funcDelta, "name", None):
                            currentCall["function"]["name"] += funcDelta.name
                        if getattr(funcDelta, "arguments", None):
                            currentCall["function"]["arguments"] += funcDelta.arguments

        if (fullContent and responsePrint.printResponse()) or (fullReasoning and responsePrint.printThinking()):
            print(ANSI.RESET)
        else:
            print(ANSI.RESET, end="")

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

            extraBody = {}
            if thinkingBudget is not None:
                extraBody["thinking_budget_tokens"] = thinkingBudget

            responseStream = self.openaiClient.chat.completions.create(
                model="model",
                messages=messageHistory,
                temperature=0.5,
                tools=toolSchemas,
                tool_choice="auto" if toolSchemas else None,
                stream=True,
                extra_body=extraBody
            )
            content, reasoning, toolCallsList = self.handleResponseStream(responseStream, responsePrint)

            if content and content.strip():
                accumulatedContent += content + "\n"

            if not toolCallsList:
                return accumulatedContent.strip()
            
            assistantMessageDict = {
                "role": "assistant",
                "content": content,
                "reasoning_content": reasoning,
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
                    toolCall = toolMap[funcName]
                    try:
                        print(f"{ANSI.BOLD} -> Executing {funcName}(**{funcArgsDict})...", end="")
                        toolResult = toolCall.executeTool(**funcArgsDict)
                        stringResult = json.dumps(toolResult)
                        print(f" {ANSI.GREEN}done.  {ANSI.RESET}")
                    except Exception as e:
                        stringResult = json.dumps({"error": f"{e.__class__.__name__}: {e}"})
                        print(f" {ANSI.RED}failed: {e.__class__.__name__}: {e}  {ANSI.RESET}")
                else:
                    stringResult = json.dumps({"error": f"Tool {funcName} doesn't exist or not accessible by this agent."})

                messageHistory.append({
                    "role": "tool",
                    "tool_call_id": currentToolCall["id"],
                    "content": stringResult
                })

        extraBody = {}
        if thinkingBudget is not None:
            extraBody["thinking_budget_tokens"] = thinkingBudget

        # If this code is reached, it means the maximum number of iterations was reached without a final response
        finalResponseStream = self.openaiClient.chat.completions.create(
            model="model",
            messages=messageHistory,
            temperature=0.5,
            stream=True,
            extra_body=extraBody
        )
        finalContent, _, _ = self.handleResponseStream(finalResponseStream, responsePrint)
        
        if finalContent and finalContent.strip():
            accumulatedContent += finalContent

        return accumulatedContent.strip()
    

class FinancialAgent:
    def __init__(self, config: FinancialAgentConfig, apiClient: LlamaCppClient, dateStr: str):
        self.apiClient = apiClient

        self.agentRole = config.agentRole
        self.persona = config.systemPersona
        self.tools = config.tools
        self.color = config.color

        self.config = config
        self.researcherSystemMessage = buildResearcherSysPrompt(config, dateStr)
        self.uiFormatSystemMessage = buildUiFormatSysPrompt(config)

        self.messageHistory = [
            {"role": "system", "content": self.researcherSystemMessage}
        ]

    def executeInternalAnalysis(self, incomingMessage: str, responsePrint: ResponsePrintMode = ResponsePrintMode.FULL):
        self.messageHistory.append({"role": "user", "content": incomingMessage})
        print(f"\n{self.color}{ANSI.BOLD}========== [{self.agentRole}] is analyzing... =========={ANSI.RESET}", end="")
        
        rawAnalysis = self.apiClient.runConversation(self.messageHistory, self.tools, THINKING_BUDGET, responsePrint)
        self.messageHistory.append({"role": "assistant", "content": rawAnalysis})
        return rawAnalysis


    def generateUiSummary(self, rawAnalysis: str, responsePrint: ResponsePrintMode = ResponsePrintMode.SILENT):
        tempHistory = [
            {"role": "system", "content": self.uiFormatSystemMessage},
            {"role": "user", "content": f"Reformat the following raw analysis according to the instructions:\n\n{rawAnalysis}"}
        ]

        uiSummary = self.apiClient.runConversation(tempHistory, [], SUMMARISE_THINK_BUDGET, responsePrint)
        return uiSummary


    def analyseAndReply(self, 
            incomingMessage: str, 
            responsePrintRawAnalysis: ResponsePrintMode = ResponsePrintMode.FULL,
            responsePrintUiSummary: ResponsePrintMode = ResponsePrintMode.SILENT
        ):
        generatingSummaryAdvisory = responsePrintUiSummary == ResponsePrintMode.SILENT and responsePrintRawAnalysis != ResponsePrintMode.SILENT
        
        rawAnalysis = self.executeInternalAnalysis(incomingMessage, responsePrintRawAnalysis)

        if generatingSummaryAdvisory:
            print(f"\n{self.color}{ANSI.BOLD}========== [{self.agentRole}] is generating UI summary... =========={ANSI.RESET}", end="\r")

        uiSummary = self.generateUiSummary(rawAnalysis, responsePrintUiSummary)

        if generatingSummaryAdvisory:
            print(f"{self.color}{ANSI.BOLD}========== [{self.agentRole}] UI summary generation complete. =========={ANSI.RESET}\n")

        return rawAnalysis, uiSummary



class BoardroomEngine:
    def __init__(self, agents: Dict[str, FinancialAgent]):
        self.macroAnalyst = agents.get("macroAnalyst")
        self.bullAnalyst = agents.get("bullAnalyst")
        self.bearAnalyst = agents.get("bearAnalyst")
        self.aggRiskAnalyst = agents.get("aggRiskAnalyst")
        self.consRiskAnalyst = agents.get("consRiskAnalyst")
        self.portManager = agents.get("portManager")


    def executeSingleEquityRating(self, targetTicker):
        print(f"\n{'='*70}\nStarting Live Boardroom Evaluation for: {targetTicker}\n{'='*70}")
        

        def newPhaseHeader(phaseNumber, phaseName):
            if phaseNumber == 0:
                tempHeader = f"{phaseName}"
            else:
                tempHeader = f"Phase {phaseNumber}: {phaseName}"
            tempHeader = f"{'|'*5} {tempHeader} {'|'*5}"
            headerLength = len(tempHeader)
            print(f"\n{ANSI.BOLD}{'-'*headerLength}\n{tempHeader}\n{'-'*headerLength}{ANSI.RESET}\n")


        # Phase 1: Macro Environment Analysis
        newPhaseHeader(1, "Macro Environment Analysis")
        macroRaw, macroUiSummary = self.macroAnalyst.analyseAndReply(
            f"Current Phase: *PHASE 1* - Macro Environment Analysis\n"
            "Analyse the current macroeconomic environment and produce a concise summary under the rules marked for Phase 1."
        )
        
        # Phase 2: Specialist Research
        newPhaseHeader(2, f"Specialist Research on {targetTicker}")
        researchPrompt = (
            f"Macroeconomic summary produced by the Macro Analyst:\n"
            f"{macroRaw}\n\n"
            f"Current Phase: *PHASE 2* - Specialist Research on {targetTicker}\n"
            f"You must conduct your research on this ticker: {targetTicker}, under the rules marked for Phase 2. "
        )
        
        bullThesisRaw, bullThesisUiSummary = self.bullAnalyst.analyseAndReply(researchPrompt)        
        bearThesisRaw, bearThesisUiSummary = self.bearAnalyst.analyseAndReply(researchPrompt)

        # Phase 3: Senior Risk Debate
        newPhaseHeader(3, f"Senior Risk Debate on {targetTicker}")
        aggQuestionsRaw, aggQuestionsUiSummary = self.aggRiskAnalyst.analyseAndReply(
            f"Macroeconomic summary produced by the Macro Analyst:\n{macroRaw}\n\n"
            # f"Bullish Value Analyst's Thesis:\n{bullThesis}\n\n"
            f"Bearish Value Analyst's Thesis:\n{bearThesisRaw}\n\n"
            f"Current Phase: *PHASE 3* - Senior Risk Debate on {targetTicker}\n"
            f"Review the theses and targets for {targetTicker} and produce 2-3 questions challenging this thesis under the rules marked for Phase 3."
        )
        
        consQuestionsRaw, consQuestionsUiSummary = self.consRiskAnalyst.analyseAndReply(
            f"Macroeconomic summary produced by the Macro Analyst:\n{macroRaw}\n\n"
            f"Bullish Value Analyst's Thesis:\n{bullThesisRaw}\n\n"
            # f"Bearish Value Analyst's Thesis:\n{bearThesis}\n\n"
            f"Current Phase: *PHASE 3* - Senior Risk Debate on {targetTicker}\n"
            f"Review the theses and targets for {targetTicker} and produce 2-3 questions challenging this thesis under the rules marked for Phase 3."
        )
        
        # Phase 4: Analyst Defense
        newPhaseHeader(4, f"Analyst Defense on {targetTicker}")
        bullDefenseRaw, bullDefenseUiSummary = self.bullAnalyst.analyseAndReply(
            f"Questions posed by the Conservative Risk Analyst:\n{consQuestionsRaw}\n\n"
            f"Current Phase: *PHASE 4* - Analyst Defense on {targetTicker}\n"
            f"Produce your response to these questions under the rules marked for Phase 4."
        )
        bearDefenseRaw, bearDefenseUiSummary = self.bearAnalyst.analyseAndReply(
            f"Questions posed by the Aggressive Risk Analyst:\n{aggQuestionsRaw}\n\n"
            f"Current Phase: *PHASE 4* - Analyst Defense on {targetTicker}\n"
            f"Produce your response to these questions under the rules marked for Phase 4."
        )

        # Phase 5: Q&A Based Proposals
        newPhaseHeader(5, f"Q&A-Based Proposals on {targetTicker}")
        aggProposalRaw, aggProposalUiSummary = self.aggRiskAnalyst.analyseAndReply(
            f"Bearish Analyst's Response/Defense:\n{bearDefenseRaw}\n\n"
            f"Current Phase: *PHASE 5* - Q&A-Based Proposals on {targetTicker}\n"
            f"Based on the defenses, make your final proposals with justification under the rules marked for Phase 5."
        )
        consProposalRaw, consProposalUiSummary = self.consRiskAnalyst.analyseAndReply(
            f"Bullish Analyst's Response/Defense:\n{bullDefenseRaw}\n\n"
            f"Current Phase: *PHASE 5* - Q&A-Based Proposals on {targetTicker}\n"
            f"Based on the defenses, make your final proposals with justification under the rules marked for Phase 5."
        )

        # Phase 6: Final Executive Decision
        newPhaseHeader(6, f"Final Executive Decision on {targetTicker}")
        managerPrompt = (
            f"Target Asset: {targetTicker}\n"
            f"Macro Conditions:\n{macroRaw}\n\n"
            f"Aggressive Allocation Case:\n{aggProposalRaw}\n\n"
            f"Conservative Allocation Case:\n{consProposalRaw}\n\n"
            f"Current Phase: *PHASE 6* - Final Executive Decision on {targetTicker}\n"
            f"Weigh up the arguments and make the final executive decision under the rules marked for Phase 6."
        )
        finalDecisionRaw, finalDecisionUiSummary = self.portManager.analyseAndReply(managerPrompt)
        
        print()
        newPhaseHeader(0, f"Final Boardroom Summary on {targetTicker}")        
            
        separator = f"\n{ANSI.BOLD}{ANSI.DIM}{'-'*70}{ANSI.RESET}\n"
        print(
            f"\n{ANSI.BOLD}{self.macroAnalyst.color}Macro Analyst Summary:\n{ANSI.RESET}{macroUiSummary}\n"
            f"{separator}"

            f"\n{ANSI.BOLD}{self.bullAnalyst.color}Bullish Analyst Summary:\n{ANSI.RESET}{bullThesisUiSummary}\n"
            f"\n⇩\n"
            f"\n{ANSI.BOLD}{self.consRiskAnalyst.color}Conservative Risk Analyst Summary and Questions:\n{ANSI.RESET}{consQuestionsUiSummary}\n"
            f"\n⇩\n"
            f"\n{ANSI.BOLD}{self.bullAnalyst.color}Bullish Analyst Defense:\n{ANSI.RESET}{bullDefenseUiSummary}\n"
            f"\n{separator}\n"

            f"\n{ANSI.BOLD}{self.bearAnalyst.color}Bearish Analyst Summary:\n{ANSI.RESET}{bearThesisUiSummary}\n"
            f"\n⇩\n"
            f"\n{ANSI.BOLD}{self.aggRiskAnalyst.color}Aggressive Risk Analyst Summary and Questions:\n{ANSI.RESET}{aggQuestionsUiSummary}\n"
            f"\n⇩\n"
            f"\n{ANSI.BOLD}{self.bearAnalyst.color}Bearish Analyst Defense:\n{ANSI.RESET}{bearDefenseUiSummary}\n"
            f"\n{separator}\n"

            f"\n{ANSI.BOLD}{self.aggRiskAnalyst.color}Aggressive Risk Analyst Proposal:\n{ANSI.RESET}{aggProposalUiSummary}\n"
            f"\n{separator}\n"

            f"\n{ANSI.BOLD}{self.consRiskAnalyst.color}Conservative Risk Analyst Proposal:\n{ANSI.RESET}{consProposalUiSummary}\n"
            f"\n{separator}\n"

            f"\n{ANSI.BOLD}{self.portManager.color}Final Executive Decision:\n{ANSI.RESET}{finalDecisionUiSummary}\n"
            f"\n\n"
        )



if __name__ == "__main__":
    killExistingLlamaCppProcesses()

    serverProcess = LlamaCppProcessInitiator(model=LlamaCppModel.GEMMA_4_E2B)
    startThread = serverProcess.startOnAnotherThread()
    startThread.join()  

    localClient = LlamaCppClient(serverProcess)

    liveTools = buildToolsRegistry()
    toolMap = {tool.toolName: tool for tool in liveTools}

    simulatedDate = time.strftime("%Y-%m-%d", time.localtime())

    macroAgent = FinancialAgent(
        config=FinancialAgentConfig(
            agentRole="Macro Strategist",
            systemPersona=getSystemPersona("Macro Strategist"),
            tools=[toolMap["fetchMacroContext"]],
            color=ANSI.CYAN
        ),
        apiClient=localClient,
        dateStr=simulatedDate
    )

    bullAgent = FinancialAgent(
        config=FinancialAgentConfig(
            agentRole="Bullish Value Analyst",
            systemPersona=getSystemPersona("Bullish Value Analyst"),
            tools=[
                toolMap["fetchCompanyProfile"],
                toolMap["fetchStockKeyFundamentals"],
                toolMap["fetchIncomeStatement"],
                toolMap["fetchStockPricePerformance"],
                toolMap["fetchAnalystConsensus"],
                toolMap["fetchCompanyRecentNews"],
                toolMap["executePythonCalculation"]
            ],
            color=ANSI.GREEN
        ),
        apiClient=localClient,
        dateStr=simulatedDate
    )

    bearAgent = FinancialAgent(
        config=FinancialAgentConfig(
            agentRole="Bearish Risk Analyst",
            systemPersona=getSystemPersona("Bearish Risk Analyst"),
            tools=[
                toolMap["fetchCompanyProfile"],
                toolMap["fetchStockKeyFundamentals"],
                toolMap["fetchBalanceSheet"],
                toolMap["fetchCashFlowStatement"],
                toolMap["fetchStockPricePerformance"],
                toolMap["fetchAnalystConsensus"],
                toolMap["fetchCompanyRecentNews"],
                toolMap["executePythonCalculation"]
            ],
            color=ANSI.RED
        ),
        apiClient=localClient,
        dateStr=simulatedDate
    )

    aggRiskAnalystAgent = FinancialAgent(
        config=FinancialAgentConfig(
            agentRole="Aggressive Risk Analyst",
            systemPersona=getSystemPersona("Aggressive Risk Analyst"),
            tools=[
                toolMap["fetchCompanyProfile"],
                toolMap["fetchStockKeyFundamentals"],
                toolMap["fetchBalanceSheet"],
                toolMap["fetchCashFlowStatement"],
                toolMap["fetchStockPricePerformance"],
                toolMap["fetchAnalystConsensus"],
                toolMap["fetchCompanyRecentNews"],
                toolMap["executePythonCalculation"]
            ],
            color=ANSI.YELLOW
        ),
        apiClient=localClient,
        dateStr=simulatedDate
    )

    consRiskAnalystAgent = FinancialAgent(
        config=FinancialAgentConfig(
            agentRole="Conservative Risk Analyst",
            systemPersona=getSystemPersona("Conservative Risk Analyst"),
            tools=[
                toolMap["fetchCompanyProfile"],
                toolMap["fetchStockKeyFundamentals"],
                toolMap["fetchBalanceSheet"],
                toolMap["fetchCashFlowStatement"],
                toolMap["fetchStockPricePerformance"],
                toolMap["fetchAnalystConsensus"],
                toolMap["fetchCompanyRecentNews"],
                toolMap["executePythonCalculation"]
            ],
            color=ANSI.BLUE
        ),
        apiClient=localClient,
        dateStr=simulatedDate
    )

    portfolioManager = FinancialAgent(
        config=FinancialAgentConfig(
            agentRole="Impartial Portfolio Manager",
            systemPersona=getSystemPersona("Impartial Portfolio Manager"),
            tools=[
                toolMap["fetchCompanyProfile"],
                toolMap["executePythonCalculation"]
            ],
            color=ANSI.MAGENTA
        ),
        apiClient=localClient,
        dateStr=simulatedDate
    )

    boardroom = BoardroomEngine({
        "macroAnalyst": macroAgent,
        "bullAnalyst": bullAgent,
        "bearAnalyst": bearAgent,
        "aggRiskAnalyst": aggRiskAnalystAgent,
        "consRiskAnalyst": consRiskAnalystAgent,
        "portManager": portfolioManager
    })

    
    timeBefore = time.time()
    boardroom.executeSingleEquityRating("INTU")
    timeAfter = time.time()

    seconds = timeAfter - timeBefore
    print(f"\nTotal time taken for boardroom evaluation: {math.floor(seconds/60)} mins {seconds%60:.1f} secs")

    time.sleep(2)

    serverProcess.stop()
