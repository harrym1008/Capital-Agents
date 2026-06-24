from cli.ansi import ANSI

from llm.agents.agent_config import FinancialAgentConfig, THINKING_BUDGET, SUMMARISE_THINK_BUDGET
from llm.agents.agent_config import buildResearcherSysPrompt, buildUIFormatSysPrompt

from llm.llm_client import BaseLLMClient, ResponsePrintMode


SUMMARISE_ENABLED = True

class FinancialAgent:
    def __init__(self, config: FinancialAgentConfig, llmClient: BaseLLMClient, dateStr: str):
        self.apiClient = llmClient

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

    def executeInternalAnalysis(self, incomingMessage: str, responsePrint: ResponsePrintMode = ResponsePrintMode.FULL):
        self.messageHistory.append({"role": "user", "content": incomingMessage})
        print(f"\n{self.color}{ANSI.BOLD}========== [{self.agentRole}] is analyzing... =========={ANSI.RESET}", end="")
        
        rawAnalysis = self.apiClient.runConversation(self.messageHistory, self.tools, THINKING_BUDGET, responsePrint)
        self.messageHistory.append({"role": "assistant", "content": rawAnalysis})
        return rawAnalysis


    def generateUISummary(self, rawAnalysis: str, responsePrint: ResponsePrintMode = ResponsePrintMode.SILENT):
        tempHistory = [
            {"role": "system", "content": self.uiFormatSystemMessage},
            {"role": "user", "content": f"Reformat the following raw analysis according to the instructions:\n\n{rawAnalysis}"}
        ]

        uiSummary = self.apiClient.runConversation(tempHistory, [], SUMMARISE_THINK_BUDGET, responsePrint)
        return uiSummary


    def analyseAndReply(self, 
            incomingMessage: str, 
            responsePrintRawAnalysis: ResponsePrintMode = ResponsePrintMode.FULL,
            responsePrintUISummary: ResponsePrintMode = ResponsePrintMode.ONE_TOKEN_ONLY
        ):
        generatingSummaryAdvisory = responsePrintUISummary == ResponsePrintMode.SILENT and responsePrintRawAnalysis != ResponsePrintMode.SILENT
        
        rawAnalysis = self.executeInternalAnalysis(incomingMessage, responsePrintRawAnalysis)
        
        if not SUMMARISE_ENABLED:
            return rawAnalysis, rawAnalysis

        if generatingSummaryAdvisory:
            print(f"\n{self.color}{ANSI.BOLD}========== [{self.agentRole}] is generating UI summary... =========={ANSI.RESET}", end="\r")

        uiSummary = self.generateUISummary(rawAnalysis, responsePrintUISummary)

        if generatingSummaryAdvisory or responsePrintUISummary == ResponsePrintMode.ONE_TOKEN_ONLY:
            print(f"{self.color}{ANSI.BOLD}========== [{self.agentRole}] UI summary generation complete. =========={ANSI.RESET}\n")

        return rawAnalysis, uiSummary
