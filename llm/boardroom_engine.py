from datetime import datetime
from typing import Dict

from concurrent.futures import ThreadPoolExecutor

from cli.ansi import ANSI

from llm.llm_client import BaseLLMClient
from llm.tools_registry import buildToolsRegistry
from llm.agents.agent import FinancialAgent
from llm.agents.agent_config import FinancialAgentConfig


class BoardroomEngine:
    def __init__(self, agents: Dict[str, FinancialAgent], allowParallel: bool=True):
        self.macroAnalyst = agents.get("macroAnalyst")
        self.bullAnalyst = agents.get("bullAnalyst")
        self.bearAnalyst = agents.get("bearAnalyst")
        self.aggRiskAnalyst = agents.get("aggRiskAnalyst")
        self.consRiskAnalyst = agents.get("consRiskAnalyst")
        self.portManager = agents.get("portManager")
        self.allowParallel = allowParallel


    def _runAgentsConcurrently(self, *tasks):
        if not self.allowParallel:
            return [task() for task in tasks]
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(task) for task in tasks]
            return [future.result() for future in futures]


    def _newPhaseHeader(self, phaseNumber, phaseName):
        if phaseNumber == 0:
            tempHeader = f"{phaseName}"
        else:
            tempHeader = f"Phase {phaseNumber}: {phaseName}"
        tempHeader = f"{'|'*5} {tempHeader} {'|'*5}"
        headerLength = len(tempHeader)
        print(f"\n{ANSI.BOLD}{'-'*headerLength}\n{tempHeader}\n{'-'*headerLength}{ANSI.RESET}\n")


    def executeFastSingleEquityRating(self, targetTicker):
        startTime = datetime.now()
        print(f"\n{'='*70}\nStarting Fast Boardroom Evaluation for: {targetTicker}\n{'='*70}")        

        # Phase 1: Macro Environment Analysis
        self._newPhaseHeader(1, "Macro Environment Analysis")
        macroRaw, macroUISummary = self.macroAnalyst.analyseAndReply(
            f"Current Phase: *PHASE 1* - Macro Environment Analysis\n"
            "Analyse the current financial environment via all three of your tools and produce a concise summary under the rules marked for Phase 1."
        )
        
        # Phase 2: Specialist Research
        self._newPhaseHeader(2, f"Specialist Research on {targetTicker}")
        researchPrompt = (
            f"Macroeconomic summary produced by the Macro Analyst:\n"
            f"{macroRaw}\n\n"
            f"Current Phase: *PHASE 2* - Specialist Research on {targetTicker}\n"
            f"You must conduct your research on this ticker: {targetTicker}, under the rules marked for Phase 2. "
        )
        
        (bullThesisRaw, bullThesisUISummary), (bearThesisRaw, bearThesisUISummary) = self._runAgentsConcurrently(
            lambda: self.bullAnalyst.analyseAndReply(researchPrompt),
            lambda: self.bearAnalyst.analyseAndReply(researchPrompt)
        )
        # Phase 3 or 6: Final Executive Decision
        self._newPhaseHeader(3, f"Final Executive Decision on {targetTicker}")
        managerPrompt = (
            f"Target Asset: {targetTicker}\n"
            f"Macro Conditions:\n{macroRaw}\n\n"
            f"Aggressive Allocation Case:\n{bullThesisRaw}\n\n"
            f"Conservative Allocation Case:\n{bearThesisRaw}\n\n"
            f"Current Phase: *PHASE 6* - Final Executive Decision on {targetTicker}\n"
            f"Weigh up the arguments and make the final executive decision under the rules marked for Phase 6."
        )
        finalDecisionRaw, finalDecisionUISummary = self.portManager.analyseAndReply(managerPrompt)


        # Phase 4 or 7: Decision Upload
        _, _ = self.portManager.analyseAndReply((
            f"Current Phase: *PHASE 7* - Decision Upload on {targetTicker}\n"
            f"Upload the final decision, weight allocation, and price targets via the 'confirmBoardroomDecision' tool, under the rules marked for Phase 7."
        ), summarisationOverride=False)
        formattedExecutiveDecision = self.portManager.tools[2].toolLog[-1]

        endTime = datetime.now()
        timeTaken = endTime - startTime

        print()
        self._newPhaseHeader(0, f"Final Boardroom Summary on {targetTicker}")                    

        separator = f"\n{ANSI.BOLD}{ANSI.DIM}{'-'*70}{ANSI.RESET}\n"
        shortConvSummary = (
            f"\n{ANSI.BOLD}{self.macroAnalyst.color}Macro Analyst Summary:\n{ANSI.RESET}{macroUISummary}\n"
            f"{separator}"

            f"\n{ANSI.BOLD}{self.bullAnalyst.color}Bullish Analyst Summary:\n{ANSI.RESET}{bullThesisUISummary}\n"
            f"\n{separator}\n"

            f"\n{ANSI.BOLD}{self.bearAnalyst.color}Bearish Analyst Summary:\n{ANSI.RESET}{bearThesisUISummary}\n"
            f"\n{separator}\n"

            f"\n{ANSI.BOLD}{self.portManager.color}Final Executive Decision:\n{ANSI.RESET}{finalDecisionUISummary}\n"
            f"\n{formattedExecutiveDecision}\n"

            f"Time taken for boardroom discussion: {timeTaken.seconds//60} mins {timeTaken.seconds%60} secs\n"
        )

        fullConvSummary = (
            f"\n{ANSI.BOLD}{self.macroAnalyst.color}Macro Analyst Summary:\n{ANSI.RESET}{macroRaw}\n"
            f"{separator}"

            f"\n{ANSI.BOLD}{self.bullAnalyst.color}Bullish Analyst Summary:\n{ANSI.RESET}{bullThesisRaw}\n"
            f"\n{separator}\n"

            f"\n{ANSI.BOLD}{self.bearAnalyst.color}Bearish Analyst Summary:\n{ANSI.RESET}{bearThesisRaw}\n"
            f"\n{separator}\n"

            f"\n{ANSI.BOLD}{self.portManager.color}Final Executive Decision:\n{ANSI.RESET}{finalDecisionRaw}\n"
            f"\n{formattedExecutiveDecision}\n"
            
            f"Time taken for boardroom discussion: {timeTaken.seconds//60} mins {timeTaken.seconds%60} secs\n"
        )
        
        print(shortConvSummary)

        with open(f"output\\{targetTicker}_fast_{startTime.strftime('%Y-%m-%d_%H-%M-%S')}.ans", "w", encoding="utf-8") as f:
            f.write(fullConvSummary)
        





    def executeCompleteSingleEquityRating(self, targetTicker):
        startTime = datetime.now()
        print(f"\n{'='*70}\nStarting Live Boardroom Evaluation for: {targetTicker}\n{'='*70}")        

        # Phase 1: Macro Environment Analysis
        self._newPhaseHeader(1, "Macro Environment Analysis")
        macroRaw, macroUISummary = self.macroAnalyst.analyseAndReply(
            f"Current Phase: *PHASE 1* - Macro Environment Analysis\n"
            "Analyse the current financial environment via all three of your tools and produce a concise summary under the rules marked for Phase 1."
        )
        
        # Phase 2: Specialist Research
        self._newPhaseHeader(2, f"Specialist Research on {targetTicker}")
        researchPrompt = (
            f"Macroeconomic summary produced by the Macro Analyst:\n"
            f"{macroRaw}\n\n"
            f"Current Phase: *PHASE 2* - Specialist Research on {targetTicker}\n"
            f"You must conduct your research on this ticker: {targetTicker}, under the rules marked for Phase 2. "
        )
        
        (bullThesisRaw, bullThesisUISummary), (bearThesisRaw, bearThesisUISummary) = self._runAgentsConcurrently(
            lambda: self.bullAnalyst.analyseAndReply(researchPrompt),
            lambda: self.bearAnalyst.analyseAndReply(researchPrompt)
        )

        # Phase 3: Senior Risk Debate
        self._newPhaseHeader(3, f"Senior Risk Debate on {targetTicker}")
        
        (aggQuestionsRaw, aggQuestionsUISummary), (consQuestionsRaw, consQuestionsUISummary) = self._runAgentsConcurrently(
            lambda: self.aggRiskAnalyst.analyseAndReply(
                f"Macroeconomic summary produced by the Macro Analyst:\n{macroRaw}\n\n"
                # f"Bullish Value Analyst's Thesis:\n{bullThesis}\n\n"
                f"Bearish Value Analyst's Thesis:\n{bearThesisRaw}\n\n"
                f"Current Phase: *PHASE 3* - Senior Risk Debate on {targetTicker}\n"
                f"Review the theses and targets for {targetTicker} and produce 2-3 questions challenging this thesis under the rules marked for Phase 3."
            ),
            lambda: self.consRiskAnalyst.analyseAndReply(
                f"Macroeconomic summary produced by the Macro Analyst:\n{macroRaw}\n\n"
                f"Bullish Value Analyst's Thesis:\n{bullThesisRaw}\n\n"
                # f"Bearish Value Analyst's Thesis:\n{bearThesis}\n\n"
                f"Current Phase: *PHASE 3* - Senior Risk Debate on {targetTicker}\n"
                f"Review the theses and targets for {targetTicker} and produce 2-3 questions challenging this thesis under the rules marked for Phase 3."
            )
        )
        
        # Phase 4: Analyst Defense
        self._newPhaseHeader(4, f"Analyst Defense on {targetTicker}")
        
        (bullDefenseRaw, bullDefenseUISummary), (bearDefenseRaw, bearDefenseUISummary) = self._runAgentsConcurrently(
            lambda: self.bullAnalyst.analyseAndReply(
                f"Questions posed by the Conservative Risk Analyst:\n{consQuestionsRaw}\n\n"
                f"Current Phase: *PHASE 4* - Analyst Defense on {targetTicker}\n"
                f"Produce your response to these questions under the rules marked for Phase 4."
            ),
            lambda: self.bearAnalyst.analyseAndReply(
                f"Questions posed by the Aggressive Risk Analyst:\n{aggQuestionsRaw}\n\n"
                f"Current Phase: *PHASE 4* - Analyst Defense on {targetTicker}\n"
                f"Produce your response to these questions under the rules marked for Phase 4."
            )
        )

        # Phase 5: Q&A Based Proposals
        self._newPhaseHeader(5, f"Q&A-Based Proposals on {targetTicker}")
        
        (aggProposalRaw, aggProposalUISummary), (consProposalRaw, consProposalUISummary) = self._runAgentsConcurrently(
            lambda: self.aggRiskAnalyst.analyseAndReply(
                f"Bearish Analyst's Response/Defense:\n{bearDefenseRaw}\n\n"
                f"Current Phase: *PHASE 5* - Q&A-Based Proposals on {targetTicker}\n"
                f"Based on the defenses, make your final proposals with justification under the rules marked for Phase 5."
            ),
            lambda: self.consRiskAnalyst.analyseAndReply(
                f"Bullish Analyst's Response/Defense:\n{bullDefenseRaw}\n\n"
                f"Current Phase: *PHASE 5* - Q&A-Based Proposals on {targetTicker}\n"
                f"Based on the defenses, make your final proposals with justification under the rules marked for Phase 5."
            )
        )

        # Phase 6: Final Executive Decision
        self._newPhaseHeader(6, f"Final Executive Decision on {targetTicker}")
        managerPrompt = (
            f"Target Asset: {targetTicker}\n"
            f"Macro Conditions:\n{macroRaw}\n\n"
            f"Aggressive Allocation Case:\n{aggProposalRaw}\n\n"
            f"Conservative Allocation Case:\n{consProposalRaw}\n\n"
            f"Current Phase: *PHASE 6* - Final Executive Decision on {targetTicker}\n"
            f"Weigh up the arguments and make the final executive decision under the rules marked for Phase 6."
        )
        finalDecisionRaw, finalDecisionUISummary = self.portManager.analyseAndReply(managerPrompt)
        
        # Phase 7: Decision Upload
        _, _ = self.portManager.analyseAndReply((
            f"Current Phase: *PHASE 7* - Decision Upload on {targetTicker}\n"
            f"Upload the final decision, weight allocation, and price targets via the 'confirmBoardroomDecision' tool, under the rules marked for Phase 7."
        ), summarisationOverride=False)
        formattedExecutiveDecision = self.portManager.tools[2].toolLog[-1]

        endTime = datetime.now()
        timeTaken = endTime - startTime

        print()
        self._newPhaseHeader(0, f"Final Boardroom Summary on {targetTicker}")   
            
        separator = f"\n{ANSI.BOLD}{ANSI.DIM}{'-'*70}{ANSI.RESET}\n"
        shortConvSummary = (
            f"\n{ANSI.BOLD}{self.macroAnalyst.color}Macro Analyst Summary:\n{ANSI.RESET}{macroUISummary}\n"
            f"{separator}"

            f"\n{ANSI.BOLD}{self.bullAnalyst.color}Bullish Analyst Summary:\n{ANSI.RESET}{bullThesisUISummary}\n"
            f"\n⇩\n"
            f"\n{ANSI.BOLD}{self.consRiskAnalyst.color}Conservative Risk Analyst Summary and Questions:\n{ANSI.RESET}{consQuestionsUISummary}\n"
            f"\n⇩\n"
            f"\n{ANSI.BOLD}{self.bullAnalyst.color}Bullish Analyst Defense:\n{ANSI.RESET}{bullDefenseUISummary}\n"
            f"\n{separator}\n"

            f"\n{ANSI.BOLD}{self.bearAnalyst.color}Bearish Analyst Summary:\n{ANSI.RESET}{bearThesisUISummary}\n"
            f"\n⇩\n"
            f"\n{ANSI.BOLD}{self.aggRiskAnalyst.color}Aggressive Risk Analyst Summary and Questions:\n{ANSI.RESET}{aggQuestionsUISummary}\n"
            f"\n⇩\n"
            f"\n{ANSI.BOLD}{self.bearAnalyst.color}Bearish Analyst Defense:\n{ANSI.RESET}{bearDefenseUISummary}\n"
            f"\n{separator}\n"

            f"\n{ANSI.BOLD}{self.aggRiskAnalyst.color}Aggressive Risk Analyst Proposal:\n{ANSI.RESET}{aggProposalUISummary}\n"
            f"\n{separator}\n"

            f"\n{ANSI.BOLD}{self.consRiskAnalyst.color}Conservative Risk Analyst Proposal:\n{ANSI.RESET}{consProposalUISummary}\n"
            f"\n{separator}\n"

            f"\n{ANSI.BOLD}{self.portManager.color}Final Executive Decision:\n{ANSI.RESET}{finalDecisionUISummary}\n"
            f"\n{formattedExecutiveDecision}\n"

            f"Time taken for boardroom discussion: {timeTaken.seconds//60} mins {timeTaken.seconds%60} secs\n"
        )

        fullConvSummary = (
            f"\n{ANSI.BOLD}{self.macroAnalyst.color}Macro Analyst Summary:\n{ANSI.RESET}{macroRaw}\n"
            f"{separator}"

            f"\n{ANSI.BOLD}{self.bullAnalyst.color}Bullish Analyst Summary:\n{ANSI.RESET}{bullThesisRaw}\n"
            f"\n⇩\n"
            f"\n{ANSI.BOLD}{self.consRiskAnalyst.color}Conservative Risk Analyst Summary and Questions:\n{ANSI.RESET}{consQuestionsRaw}\n"
            f"\n⇩\n"
            f"\n{ANSI.BOLD}{self.bullAnalyst.color}Bullish Analyst Defense:\n{ANSI.RESET}{bullDefenseRaw}\n"
            f"\n{separator}\n"

            f"\n{ANSI.BOLD}{self.bearAnalyst.color}Bearish Analyst Summary:\n{ANSI.RESET}{bearThesisRaw}\n"
            f"\n⇩\n"
            f"\n{ANSI.BOLD}{self.aggRiskAnalyst.color}Aggressive Risk Analyst Summary and Questions:\n{ANSI.RESET}{aggQuestionsRaw}\n"
            f"\n⇩\n"
            f"\n{ANSI.BOLD}{self.bearAnalyst.color}Bearish Analyst Defense:\n{ANSI.RESET}{bearDefenseRaw}\n"
            f"\n{separator}\n"

            f"\n{ANSI.BOLD}{self.aggRiskAnalyst.color}Aggressive Risk Analyst Proposal:\n{ANSI.RESET}{aggProposalRaw}\n"
            f"\n{separator}\n"

            f"\n{ANSI.BOLD}{self.consRiskAnalyst.color}Conservative Risk Analyst Proposal:\n{ANSI.RESET}{consProposalRaw}\n"
            f"\n{separator}\n"

            f"\n{ANSI.BOLD}{self.portManager.color}Final Executive Decision:\n{ANSI.RESET}{finalDecisionRaw}\n"
            f"\n{formattedExecutiveDecision}\n"

            f"Time taken for boardroom discussion: {timeTaken.seconds//60} mins {timeTaken.seconds%60} secs\n"
        )
        
        print(shortConvSummary)

        with open(f"output\\{targetTicker}_full_{startTime.strftime('%Y-%m-%d_%H-%M-%S')}.ans", "w", encoding="utf-8") as f:
            f.write(fullConvSummary)       
        


    def executeSingleEquityRating(self, targetTicker, fastMode=False):
        if fastMode:
            self.executeFastSingleEquityRating(targetTicker)
        else:
            self.executeCompleteSingleEquityRating(targetTicker)



def boardroomGenerator(simulatedDate: str, llmClient: BaseLLMClient):
    toolRegistry = buildToolsRegistry()
    toolMap = {tool.toolName: tool for tool in toolRegistry}

    macroAgent = FinancialAgent(
        config=FinancialAgentConfig(
            agentRole="Macro Strategist",
            tools=[
                toolMap["fetchMacroIndicators"],
                toolMap["fetchMacroNews"],
                toolMap["fetchMacroFredSeries"],
                toolMap["executePythonCalculation"]
            ],
            color=ANSI.CYAN
        ),
        llmClient=llmClient,
        dateStr=simulatedDate
    )

    bullAgent = FinancialAgent(
        config=FinancialAgentConfig(
            agentRole="Bullish Value Analyst",
            tools=[
                toolMap["fetchCompanyProfile"],
                toolMap["fetchCompanyValuationMetrics"],
                toolMap["fetchIncomeStatement"],
                toolMap["fetchBalanceSheet"],
                toolMap["fetchCashFlowStatement"],
                toolMap["fetchStockPricePerformance"],
                toolMap["fetchAnalystConsensus"],
                toolMap["fetchCompanyRecentNews"],
                toolMap["executePythonCalculation"]
            ],
            color=ANSI.GREEN
        ),
        llmClient=llmClient,
        dateStr=simulatedDate
    )

    bearAgent = FinancialAgent(
        config=FinancialAgentConfig(
            agentRole="Bearish Risk Analyst",
            tools=[
                toolMap["fetchCompanyProfile"],
                toolMap["fetchCompanyValuationMetrics"],
                toolMap["fetchIncomeStatement"],
                toolMap["fetchBalanceSheet"],
                toolMap["fetchCashFlowStatement"],
                toolMap["fetchStockPricePerformance"],
                toolMap["fetchAnalystConsensus"],
                toolMap["fetchCompanyRecentNews"],
                toolMap["executePythonCalculation"]
            ],
            color=ANSI.RED
        ),
        llmClient=llmClient,
        dateStr=simulatedDate
    )

    aggRiskAnalystAgent = FinancialAgent(
        config=FinancialAgentConfig(
            agentRole="Aggressive Risk Analyst",
            tools=[
                toolMap["fetchCompanyProfile"],
                toolMap["fetchCompanyValuationMetrics"],
                toolMap["fetchBalanceSheet"],
                toolMap["fetchCashFlowStatement"],
                toolMap["fetchStockPricePerformance"],
                toolMap["fetchAnalystConsensus"],
                toolMap["fetchCompanyRecentNews"],
                toolMap["executePythonCalculation"]
            ],
            color=ANSI.YELLOW
        ),
        llmClient=llmClient,
        dateStr=simulatedDate
    )

    consRiskAnalystAgent = FinancialAgent(
        config=FinancialAgentConfig(
            agentRole="Conservative Risk Analyst",
            tools=[
                toolMap["fetchCompanyProfile"],
                toolMap["fetchCompanyValuationMetrics"],
                toolMap["fetchBalanceSheet"],
                toolMap["fetchCashFlowStatement"],
                toolMap["fetchStockPricePerformance"],
                toolMap["fetchAnalystConsensus"],
                toolMap["fetchCompanyRecentNews"],
                toolMap["executePythonCalculation"]
            ],
            color=ANSI.BLUE
        ),
        llmClient=llmClient,
        dateStr=simulatedDate
    )

    portfolioManager = FinancialAgent(
        config=FinancialAgentConfig(
            agentRole="Impartial Portfolio Manager",
            tools=[
                toolMap["fetchCompanyProfile"],
                toolMap["executePythonCalculation"],
                toolMap["confirmBoardroomDecision"]
            ],
            color=ANSI.MAGENTA
        ),
        llmClient=llmClient,
        dateStr=simulatedDate
    )    

    boardroom = BoardroomEngine({
        "macroAnalyst": macroAgent,
        "bullAnalyst": bullAgent,
        "bearAnalyst": bearAgent,
        "aggRiskAnalyst": aggRiskAnalystAgent,
        "consRiskAnalyst": consRiskAnalystAgent,
        "portManager": portfolioManager,
    }, allowParallel=llmClient.allowParallel)
    return boardroom

