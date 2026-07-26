from typing import Optional

import pandas as pd

from cli.ansi import ANSI

from llm.agents.agent_config import FinancialAgentConfig, THINKING_BUDGET, SUMMARISE_THINK_BUDGET, \
                                    buildResearcherSysPrompt, buildUIFormatSysPrompt
from llm.llm_client import BaseLLMClient, ResponsePrintMode
from llm.tools.tool_registry import ToolRegistry

from ui.ui_hooks import setCurrentAgent, setAgentPhase, emitEvent


SUMMARISE_ENABLED = True

class FinancialAgent:
    def __init__(self, config: FinancialAgentConfig, dateStr: str):
        self.apiClient: BaseLLMClient = None

        self.agentRole = config.agentRole
        self.persona = config.systemPersona
        self.tools = config.tools
        self.color = config.color

        self.config = config
        self.researcherSystemMessage = buildResearcherSysPrompt(config, dateStr)
        self.uiFormatSystemMessage = buildUIFormatSysPrompt(config)

        self.messageHistory = [
            {"role": "system", "content": self.researcherSystemMessage}
        ]

    def setLLMClient(self, llmClient: BaseLLMClient):
        self.apiClient = llmClient


    def executeInternalAnalysis(
            self, 
            incomingMessage: str,
            toolRegistry: ToolRegistry,
            timestamp: pd.Timestamp,
            responsePrint: ResponsePrintMode = ResponsePrintMode.FULL
        ):
        self.messageHistory.append({"role": "user", "content": incomingMessage})
        print(f"\n{self.color}{ANSI.BOLD}========== [{self.agentRole}] is analysing... =========={ANSI.RESET}", end="")
        
        rawAnalysis = self.apiClient.runConversation(
            self.messageHistory, 
            toolRegistry, 
            timestamp, 
            THINKING_BUDGET, 
            responsePrint
        )

        self.messageHistory.append({"role": "assistant", "content": rawAnalysis})
        return rawAnalysis


    def generateUISummary(
            self, 
            rawAnalysis: str, 
            responsePrint: ResponsePrintMode = ResponsePrintMode.SILENT
        ):
        tempHistory = [
            {"role": "system", "content": self.uiFormatSystemMessage},
            {"role": "user", "content": f"Reformat the following raw analysis according to the instructions:\n\n{rawAnalysis}"}
        ]

        uiSummary = self.apiClient.runConversation(
            tempHistory, 
            toolRegistry=None, 
            timestamp=pd.Timestamp.now(tz="UTC"), 
            thinkingBudget=SUMMARISE_THINK_BUDGET, 
            responsePrint=responsePrint
        )
        return uiSummary


    def analyseAndReply(self, 
            incomingMessage: str, 
            toolRegistry: ToolRegistry,
            timestamp: pd.Timestamp,
            responsePrintRawAnalysis: ResponsePrintMode = ResponsePrintMode.FULL,
            responsePrintUISummary: ResponsePrintMode = ResponsePrintMode.ONE_TOKEN_ONLY,
            summarisationOverride: Optional[bool] = None
        ):
        generateSummary = SUMMARISE_ENABLED if summarisationOverride is None else summarisationOverride
        
        # Set agent context for UI streaming
        setCurrentAgent(self.agentRole, self.color)
        setAgentPhase("raw")
        emitEvent("agentRunStart", {"agentRole": self.agentRole, "agentColor": self.color, "phase": "raw"})
        
        rawAnalysis = self.executeInternalAnalysis(incomingMessage, toolRegistry, timestamp, responsePrintRawAnalysis)        
        
        emitEvent("agentRunEnd", {"agentRole": self.agentRole, "phase": "raw"})
        
        if not generateSummary:
            return rawAnalysis, rawAnalysis

        setAgentPhase("summary")
        emitEvent("agentRunStart", {"agentRole": self.agentRole, "agentColor": self.color, "phase": "summary"})

        generatingSummaryAdvisory = responsePrintUISummary == ResponsePrintMode.SILENT and responsePrintRawAnalysis != ResponsePrintMode.SILENT
        if generatingSummaryAdvisory:
            print(f"\n{self.color}{ANSI.BOLD}========== [{self.agentRole}] is generating UI summary... =========={ANSI.RESET}", end="\r")

        uiSummary = self.generateUISummary(rawAnalysis, responsePrintUISummary)

        if generatingSummaryAdvisory or responsePrintUISummary == ResponsePrintMode.ONE_TOKEN_ONLY:
            print(f"{self.color}{ANSI.BOLD}========== [{self.agentRole}] UI summary generation complete. =========={ANSI.RESET}\n")

        emitEvent("agentRunEnd", {"agentRole": self.agentRole, "phase": "summary"})

        return rawAnalysis, uiSummary

