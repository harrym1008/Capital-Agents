from typing import Dict, List, Optional

import pandas as pd

from cli.ansi import ANSI

from llm.agents.agent_prompts import buildAgentSpecificSysPrompt, buildSummariseSysPrompt
from llm.llm_client import BaseLLMClient, ResponsePrintMode
from llmtools.tool_registry import ToolRegistry, Tool
from boardroom.boardroom_config import BoardroomConfig

from ui.ui_hooks import setCurrentAgent, setAgentPhase, emitEvent

SUMMARISE_THINK_BUDGET = 256
SUMMARISE_ENABLED = True

ANSI_TO_COLOR_NAME = {
    ANSI.CYAN: "cyan",
    ANSI.GREEN: "green",
    ANSI.RED: "red",
    ANSI.YELLOW: "yellow",
    ANSI.BLUE: "blue",
    ANSI.MAGENTA: "magenta",
    ANSI.WHITE: "white"
}


class FinancialAgent:
    def __init__(self, agentRole: str, tools: List[Tool], ansiColor: str = ANSI.RESET, dateStr: str = None, maxIterations: int = 10):
        self.llmClient: Optional[BaseLLMClient] = None

        self.agentRole = agentRole
        self.tools: List[Tool] = tools 
        self.color = ansiColor
        self.colorName = ANSI_TO_COLOR_NAME.get(ansiColor, "cyan")

        self.simulatedDateStr = dateStr
        self.messageHistory = []
        self.maxIterations = maxIterations

    @property
    def mainApiClient(self) -> Optional[BaseLLMClient]:
        return self.llmClient

    @mainApiClient.setter
    def mainApiClient(self, client: Optional[BaseLLMClient]):
        self.llmClient = client

    def setClient(self, llmClient: BaseLLMClient):
        self.llmClient = llmClient


    def getSpecificToolsStr(self) -> str:
        return ", ".join([tool.name for tool in self.tools if tool.name not in ["executePythonCalculation"]])

    def clearTools(self):
        self.tools = []

    def removeTool(self, toolName: str):
        self.tools = [tool for tool in self.tools if tool.name != toolName]

    def addTool(self, toolName: str, toolRegistry: ToolRegistry):
        tool = toolRegistry.getTool(toolName)
        if tool:
            self.tools.append(tool)
        else:
            print(f"Tool '{toolName}' not found in the registry.")


    def executeInternalAnalysis(
        self, 
        incomingMessage: str,
        toolRegistry: ToolRegistry,
        timestamp: pd.Timestamp,
        config: BoardroomConfig,
        systemPrompt: Optional[str] = None,
        requireInitialTools: bool = False
    ):
        if systemPrompt:
            if len(self.messageHistory) == 0:
                self.messageHistory.append(None)
            self.messageHistory[0] = {"role": "system", "content": systemPrompt}

            if len(self.messageHistory) > 1:
                self.messageHistory.append(
                    {
                        "role": "user", 
                        "content": "Note: the above conversation history and prior tool outputs should be used to inform your response. "
                                   "Your system prompt has been updated, you should now use this prompt to complete your next message."
                                   "Do not re-fetch data that is already present in your history unless needed for new calculations."
                    }
                )

        self.messageHistory.append({"role": "user", "content": incomingMessage})
        historyToUse = self.messageHistory

        print(f"\n{self.color}{ANSI.BOLD}========== [{self.agentRole}] is analysing... =========={ANSI.RESET}", end="")
        
        rawAnalysis = self.llmClient.runConversation(
            historyToUse, 
            toolRegistry, 
            timestamp, 
            thinkingBudget=config.thinkingBudget, 
            responsePrint=ResponsePrintMode.FULL,
            requireInitialTools=requireInitialTools,
            permittedTools=self.tools,
            maxIterations=config.maxIterations,
            temperature=config.temperature
        )

        self.messageHistory.append({"role": "assistant", "content": rawAnalysis})

        return rawAnalysis


    def generateUISummary(
        self, 
        rawAnalysis: str, 
        systemPrompt: Optional[str] = None
    ):
        tempHistory = [
            {"role": "system", "content": systemPrompt},
            {"role": "user", "content": f"Reformat the following raw analysis according to the instructions:\n\n{rawAnalysis}"}
        ]

        uiSummary = self.llmClient.runConversation(
            tempHistory, 
            toolRegistry=None, 
            timestamp=pd.Timestamp.now(tz="UTC"), 
            thinkingBudget=SUMMARISE_THINK_BUDGET, 
            responsePrint=ResponsePrintMode.ONE_TOKEN_ONLY,
            requireInitialTools=False,
            permittedTools=[]
        )
        return uiSummary


    def analyseAndReply(self, 
        incomingMessage: str, 
        toolRegistry: ToolRegistry,
        timestamp: pd.Timestamp,
        config: BoardroomConfig,
        subrole: Optional[str] = None,
        requireInitialTools: bool = False,
        summarisationOverride: Optional[bool] = None,
        sysPromptOverride: Optional[str] = None,
        modeOverride: Optional[str] = None,
    ):
        generateSummary = config.generateSummaries if summarisationOverride is None else summarisationOverride
        
        # Set agent context for UI streaming
        setCurrentAgent(self.agentRole, self.colorName)
        setAgentPhase("raw")
        emitEvent("agentRunStart", {"agentRole": self.agentRole, "agentColor": self.colorName, "phase": "raw"})
        
        mode = modeOverride or (config.modeName if hasattr(config, "modeName") else "SingleEquityRating")

        if sysPromptOverride is not None:
            sysPrompt = sysPromptOverride
        else:
            dateStr = self.simulatedDateStr if self.simulatedDateStr else timestamp.strftime("%Y-%m-%d")
            sysPrompt = buildAgentSpecificSysPrompt(
                dateStr=dateStr,
                agentRole=self.agentRole,
                agentToolsStr=self.getSpecificToolsStr(),
                mode=mode,
                subrole=subrole,
                promptArgs=config.getPromptArgs()
            )

        # Run the actual inference function
        try:
            rawAnalysis = self.executeInternalAnalysis(
                incomingMessage, toolRegistry, timestamp, config, sysPrompt, requireInitialTools
            )        
        finally:
            emitEvent("agentRunEnd", {"agentRole": self.agentRole, "phase": "raw"})
        
        if not generateSummary:
            return rawAnalysis, rawAnalysis

        setAgentPhase("summary")
        emitEvent("agentRunStart", {"agentRole": self.agentRole, "agentColor": self.colorName, "phase": "summary"})

        # Summarise raw analysis for UI display
        try:
            uiSummary = self.generateUISummary(rawAnalysis, buildSummariseSysPrompt(self.agentRole, mode, subrole, config.getPromptArgs()))
        finally:
            emitEvent("agentRunEnd", {"agentRole": self.agentRole, "phase": "summary"})

        return rawAnalysis, uiSummary



