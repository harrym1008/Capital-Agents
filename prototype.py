import json
import time
from typing import Callable, Any, Dict, List, Optional
from openai import OpenAI

from cli.ansi import ANSI
from llm.llamacpp_args import LlamaCppModel
from llm.llamacpp_init import LlamaCppProcessInitiator, killExistingLlamaCppProcesses

from llm.tools_registry import Tool, buildToolsRegistry
from llm.agent_config import FinancialAgentConfig
from llm.agent_prompts import buildSystemPrompt, getSystemPersona



class LlamaCppClient:
    def __init__(self, processInitiator):
        self.processInitiator = processInitiator
        self.openaiClient = OpenAI(base_url=self.processInitiator.apiUrl, api_key="abcd")


    def handleResponseStream(self, responseStream, onlyPrintResponse: bool = False):
        fullContent = ""
        fullReasoning = ""
        toolCallsList = []
        isThinking = False

        if onlyPrintResponse:
            isThinking = False
        
        for chunk in responseStream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # 1. Capture reasoning content tokens
            reasoningChunk = getattr(delta, "reasoning_content", None)
            if reasoningChunk and not onlyPrintResponse:
                if not isThinking:
                    print(f"\n{ANSI.DIM}[Thinking]: ", end="", flush=True)
                    isThinking = True
                print(reasoningChunk, end="", flush=True)
                fullReasoning += reasoningChunk

            # 2. Capture regular response text tokens
            contentChunk = getattr(delta, "content", None)
            if contentChunk:
                if isThinking:
                    print(f"\n{ANSI.RESET}[Response]: ", end="", flush=True)
                    isThinking = False
                elif fullContent == "":
                    print("\n[Response]: ", end="", flush=True)
                print(contentChunk, end="", flush=True)
                fullContent += contentChunk

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

        if fullContent or fullReasoning:
            print(ANSI.RESET)

        return fullContent, fullReasoning, toolCallsList



    def runConversation(self, messageHistory: List[Dict[str, Any]], availableTools: Optional[List[Tool]] = None, onlyPrintResponse: bool = False):
        toolSchemas = [tool.getToolSchema() for tool in availableTools] if availableTools else []
        maxIterations = 12
        currentIteration = 0

        while currentIteration < maxIterations:
            currentIteration += 1

            responseStream = self.openaiClient.chat.completions.create(
                model="model",
                messages=messageHistory,
                temperature=0.55,
                tools=toolSchemas,
                tool_choice="auto" if toolSchemas else None,
                stream=True
            )
            content, reasoning, toolCallsList = self.handleResponseStream(responseStream, onlyPrintResponse=onlyPrintResponse)

            if not toolCallsList:
                return content
            
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

        # If this code is reached, it means the maximum number of iterations was reached without a final response
        finalResponseStream = self.openaiClient.chat.completions.create(
            model="model",
            messages=messageHistory,
            temperature=0.55,
            stream=True
        )
        finalContent, _, _ = self.handleResponseStream(finalResponseStream, onlyPrintResponse=onlyPrintResponse)
        return finalContent
    

class FinancialAgent:
    def __init__(self, config: FinancialAgentConfig, apiClient: LlamaCppClient, dateStr: str):
        self.apiClient = apiClient

        self.agentRole = config.agentRole
        self.persona = config.systemPersona
        self.tools = config.tools
        self.color = config.color

        self.config = config
        self.systemPrompt = buildSystemPrompt(config, dateStr)

        self.messageHistory = [
            {"role": "system", "content": self.systemPrompt}
        ]

    def analyseAndReply(self, incomingMessage: str, onlyPrintResponse: bool = False):
        self.messageHistory.append({"role": "user", "content": incomingMessage})
        print(f"\n{self.color}{ANSI.BOLD}========== [{self.agentRole}] is analysing... =========={ANSI.RESET}", end="")
        response = self.apiClient.runConversation(self.messageHistory, self.tools, onlyPrintResponse=onlyPrintResponse)
        self.messageHistory.append({"role": "assistant", "content": response})
        return response



class BoardroomEngine:
    def __init__(self, agents: Dict[str, FinancialAgent]):
        self.macroAnalyst = agents.get("macroAnalyst")
        self.bullAnalyst = agents.get("bullAnalyst")
        self.bearAnalyst = agents.get("bearAnalyst")
        self.aggRiskAnalyst = agents.get("aggRiskAnalyst")
        self.consRiskAnalyst = agents.get("consRiskAnalyst")
        self.portManager = agents.get("portManager")
        self.boardSummariser = agents.get("boardSummariser")


    def executeSingleEquityRating(self, targetTicker):
        print(f"\n{'='*70}\nStarting Live Boardroom Evaluation for: {targetTicker}\n{'='*70}")
        
        # Phase 1: Macro Environment Analysis
        print("\n--- Phase 1: Macro Environment Analysis ---")
        macroSummary = self.macroAnalyst.analyseAndReply(
            f"Current Phase: *PHASE 1* - Macro Environment Analysis\n"
            "Analyse the current macroeconomic environment and produce a concise summary under the rules marked for Phase 1."
        )
        
        # Phase 2: Specialist Research
        print(f"\n--- Phase 2: Specialist Research on {targetTicker} ---")
        researchPrompt = (
            f"Macroeconomic summary produced by the Macro Analyst:\n"
            f"{macroSummary}\n\n"
            f"Current Phase: *PHASE 2* - Specialist Research on {targetTicker}\n"
            f"You must conduct your research on this ticker: {targetTicker}, under the rules marked for Phase 2. "
        )
        
        bullThesis = self.bullAnalyst.analyseAndReply(researchPrompt)        
        bearThesis = self.bearAnalyst.analyseAndReply(researchPrompt)

        # Phase 3: Senior Risk Debate
        print(f"\n--- Phase 3 & 4: Senior Risk Debate & Analyst Defense ---")
        debateContext = f"Bull Thesis:\n{bullThesis}\n\nBear Thesis:\n{bearThesis}"
        
        aggQuestions = self.aggRiskAnalyst.analyseAndReply(
            f"Macroeconomic summary produced by the Macro Analyst:\n{macroSummary}\n\n"
            # f"Bullish Value Analyst's Thesis:\n{bullThesis}\n\n"
            f"Bearish Value Analyst's Thesis:\n{bearThesis}\n\n"
            f"Current Phase: *PHASE 3* - Senior Risk Debate on {targetTicker}\n"
            f"Review the theses and targets for {targetTicker} and produce 2-3 questions challenging this thesis under the rules marked for Phase 3."
        )
        
        consQuestions = self.consRiskAnalyst.analyseAndReply(
            f"Macroeconomic summary produced by the Macro Analyst:\n{macroSummary}\n\n"
            f"Bullish Value Analyst's Thesis:\n{bullThesis}\n\n"
            # f"Bearish Value Analyst's Thesis:\n{bearThesis}\n\n"
            f"Current Phase: *PHASE 3* - Senior Risk Debate on {targetTicker}\n"
            f"Review the theses and targets for {targetTicker} and produce 2-3 questions challenging this thesis under the rules marked for Phase 3."
        )
        
        # Phase 4: Analyst Defense
        bullDefense = self.bullAnalyst.analyseAndReply(
            f"Questions posed by the Conservative Risk Analyst:\n{consQuestions}\n\n"
            f"Current Phase: *PHASE 4* - Analyst Defense on {targetTicker}\n"
            f"Produce your response to these questions under the rules marked for Phase 4."
        )
        bearDefense = self.bearAnalyst.analyseAndReply(
            f"Questions posed by the Aggressive Risk Analyst:\n{aggQuestions}\n\n"
            f"Current Phase: *PHASE 4* - Analyst Defense on {targetTicker}\n"
            f"Produce your response to these questions under the rules marked for Phase 4."
        )

        # Phase 5: Q&A Based Proposals
        print(f"\n--- Phase 5: Q&A-Based Proposals ---")
        aggProposal = self.aggRiskAnalyst.analyseAndReply(
            f"Bearish Analyst's Response/Defense:\n{bearDefense}\n\n"
            f"Current Phase: *PHASE 5* - Q&A-Based Proposals on {targetTicker}\n"
            f"Based on the defenses, make your final proposals with justification under the rules marked for Phase 5."
        )
        consProposal = self.consRiskAnalyst.analyseAndReply(
            f"Bullish Analyst's Response/Defense:\n{bullDefense}\n\n"
            f"Current Phase: *PHASE 5* - Q&A-Based Proposals on {targetTicker}\n"
            f"Based on the defenses, make your final proposals with justification under the rules marked for Phase 5."
        )

        # Phase 6: Final Executive Decision
        print(f"\n--- Phase 6: Final Executive Decision ---")
        managerPrompt = (
            f"Target Asset: {targetTicker}\n"
            f"Macro Conditions:\n{macroSummary}\n\n"
            f"Aggressive Allocation Case:\n{aggProposal}\n\n"
            f"Conservative Allocation Case:\n{consProposal}\n\n"
            f"Current Phase: *PHASE 6* - Final Executive Decision on {targetTicker}\n"
            f"Weigh up the arguments and make the final executive decision under the rules marked for Phase 6."
        )
        finalDecision = self.portManager.analyseAndReply(managerPrompt)
        
        # Phase 7: Executive Boardroom Summarization
        print(f"\n\n\n--- Phase 7: Executive Boardroom Summarization ---\n\n\n")
        summaryPrompt = (
            f"Current Phase: *PHASE 7* - Executive Summarization on {targetTicker}\n"
            f"Below is the full conversation for you to summarise:\n\n"
            f"Phase 1: Macro Analyst Summary:\n{macroSummary}\n\n"
            f"Phase 2: Bullish Analyst Summary:\n{bullThesis}\n\n"
            f"Phase 2: Bearish Analyst Summary:\n{bearThesis}\n\n"
            f"Phase 3: Aggressive Risk Analyst Questions:\n{aggQuestions}\n\n"
            f"Phase 3: Conservative Risk Analyst Questions:\n{consQuestions}\n\n"
            f"Phase 4: Bullish Analyst Defense:\n{bullDefense}\n\n"
            f"Phase 4: Bearish Analyst Defense:\n{bearDefense}\n\n"
            f"Phase 5: Aggressive Risk Analyst Summary:\n{aggProposal}\n\n"
            f"Phase 5: Conservative Risk Analyst Summary:\n{consProposal}\n\n"
            f"Phase 6: Final Executive Decision:\n{finalDecision}"
        )
        executiveSummary = self.boardSummariser.analyseAndReply(summaryPrompt, onlyPrintResponse=True)
            
        print(f"\n{'='*70}\nBoardroom Evaluation for {targetTicker} Completed.\n{'='*70}")


if __name__ == "__main__":
    killExistingLlamaCppProcesses()

    serverProcess = LlamaCppProcessInitiator(model=LlamaCppModel.GEMMA_4_12B)
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

    summariserAgent = FinancialAgent(
        config=FinancialAgentConfig(
            agentRole="Executive Boardroom Summariser",
            systemPersona=getSystemPersona("Executive Boardroom Summariser"),
            tools=[],
            color=ANSI.BOLD
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
        "portManager": portfolioManager,
        "boardSummariser": summariserAgent
    })

    # boardroom.executeSingleEquityRating("NVDA")
    # time.sleep(2)
    # boardroom.executeSingleEquityRating("AAPL")
    # time.sleep(2)
    boardroom.executeSingleEquityRating("NVDA")
    time.sleep(2)

    serverProcess.stop()
