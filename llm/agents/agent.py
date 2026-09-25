from typing import Dict, List, Optional

from colorama import Fore, Style
import pandas as pd

from llm.agents.agent_prompts import buildAgentSpecificSysPrompt, buildSummariseSysPrompt
from llm.llm_client import BaseLLMClient, ResponsePrintMode
from llmtools.tool_registry import ToolRegistry, Tool
from boardroom.boardroom_config import BoardroomConfig

from ui.ui_hooks import setCurrentAgent, setAgentPhase, emitEvent

SUMMARISE_THINK_BUDGET = 256
SUMMARISE_ENABLED = True


# Encapsules role, toolset, message history, etc. for a single agent
class FinancialAgent:
    def __init__(self, agentRole: str, tools: List[Tool], color: str = "black", dateStr: str = None, maxIterations: int = 10):
        self.llmClient: Optional[BaseLLMClient] = None

        self.agentRole = agentRole
        self.tools: List[Tool] = tools 
        self.color = color
        self.ansiCode = getattr(Fore, color.upper(), Fore.RESET)

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
        # Attach LLM inference client to agent
        self.llmClient = llmClient


    def getSpecificToolsStr(self) -> str:
        # Generate comma-separated list of assigned tool names excluding internal calculations
        return ", ".join([tool.name for tool in self.tools if tool.name not in ["executePythonCalculation"]])

    def clearTools(self):
        # Empty tool list
        self.tools = []

    def removeTool(self, toolName: str):
        # Remove specific tool by name
        self.tools = [tool for tool in self.tools if tool.name != toolName]

    def addTool(self, toolName: str, toolRegistry: ToolRegistry):
        # Look up tool in registry and append to active tools list
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
        # Execute multi-turn reasoning and tool calls for raw analysis
        # Overwrite the system prompt to ensure the agent does the correct reasoning/tools/workflow for the current stage
        if systemPrompt:
            if len(self.messageHistory) == 0:
                self.messageHistory.append(None)
            oldPrompt = self.messageHistory[0].get("content") if isinstance(self.messageHistory[0], dict) else None
            self.messageHistory[0] = {"role": "system", "content": systemPrompt}

            if len(self.messageHistory) > 1 and oldPrompt is not None and oldPrompt != systemPrompt:
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

        print(f"\n{self.ansiCode}{Style.BRIGHT}========== [{self.agentRole}] is analysing... =========={Style.RESET_ALL}", end="")
        
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
        # Reformat technical raw analysis into condensed UI executive summary
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
        # Orchestrate complete agent execution cycle: raw analysis followed by UI summarisation
        generateSummary = config.generateSummaries if summarisationOverride is None else summarisationOverride
        
        # Set agent context for UI streaming
        setCurrentAgent(self.agentRole, self.color)
        setAgentPhase("raw")
        emitEvent("agentRunStart", {"agentRole": self.agentRole, "agentColor": self.color, "phase": "raw"})
        
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

        # Run primary internal analysis
        try:
            rawAnalysis = self.executeInternalAnalysis(
                incomingMessage, toolRegistry, timestamp, config, sysPrompt, requireInitialTools
            )        
        finally:
            emitEvent("agentRunEnd", {"agentRole": self.agentRole, "phase": "raw"})
        
        if not generateSummary:
            return rawAnalysis, rawAnalysis

        setAgentPhase("summary")
        emitEvent("agentRunStart", {"agentRole": self.agentRole, "agentColor": self.color, "phase": "summary"})

        # Generate condensed executive summary for UI dashboard, if enabled
        try:
            uiSummary = self.generateUISummary(rawAnalysis, buildSummariseSysPrompt(self.agentRole, mode, subrole, config.getPromptArgs()))
        finally:
            emitEvent("agentRunEnd", {"agentRole": self.agentRole, "phase": "summary"})

        return rawAnalysis, uiSummary
