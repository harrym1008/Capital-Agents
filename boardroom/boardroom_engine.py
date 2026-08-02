from datetime import datetime
from typing import Dict

import pandas as pd

from concurrent.futures import ThreadPoolExecutor

from cli.ansi import ANSI

from llm.client_duo import ClientDuo
from tools.registry_builder import ToolRegistry
from llm.agents.agent import FinancialAgent
from boardroom.boardroom_config import BoardroomConfig, SingleEquityRatingConfig, TimeHorizon, TIME_HORIZON_INFO

from ui.ui_hooks import getCurrentStage, setCurrentStage, emitEvent, SimulationStoppedException


class BoardroomEngine:
    def __init__(
            self, 
            agents: Dict[str, FinancialAgent], 
            timestamp: pd.Timestamp,
            toolRegistry: ToolRegistry
        ):

        self.clientDuo: ClientDuo = None
        self.allowParallel = False

        self.timestamp = timestamp
        self.toolRegistry = toolRegistry
        
        self.macroAnalyst = agents.get("macroAnalyst")
        self.bullAnalyst = agents.get("bullAnalyst")
        self.bearAnalyst = agents.get("bearAnalyst")
        self.aggRiskAnalyst = agents.get("aggRiskAnalyst")
        self.consRiskAnalyst = agents.get("consRiskAnalyst")
        self.portManager = agents.get("portManager")


    def assignClientDuo(self, clientDuo: ClientDuo):
        self.clientDuo = clientDuo
        for agent in [self.macroAnalyst, self.bullAnalyst, self.bearAnalyst, self.aggRiskAnalyst, self.consRiskAnalyst, self.portManager]:
            agent.setClientDuo(clientDuo)
        self.allowParallel = clientDuo.boardroomClient.allowParallel


    def _runAgentsConcurrently(self, *tasks):
        if not self.allowParallel:
            return [task() for task in tasks]
        
        currentStageNum = getCurrentStage()
        
        def wrappedTask(task):
            setCurrentStage(currentStageNum)
            return task()

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(wrappedTask, task) for task in tasks]
            results = []
            try:
                for future in futures:
                    results.append(future.result())
            except SimulationStoppedException:
                # Cancel any remaining futures and propagate the stop
                for f in futures:
                    f.cancel()
                raise
            return results


    def _newPhaseHeader(self, phaseNumber, phaseName):
        setCurrentStage(phaseNumber)
        
        if phaseNumber == 0:
            tempHeader = f"{phaseName}"
        else:
            tempHeader = f"Phase {phaseNumber}: {phaseName}"
        tempHeader = f"{'|'*5} {tempHeader} {'|'*5}"
        headerLength = len(tempHeader)
        print(f"\n{ANSI.BOLD}{'-'*headerLength}\n{tempHeader}\n{'-'*headerLength}{ANSI.RESET}\n")

        # Resolve active agents
        agents = []
        isFast = getattr(self, "fastMode", True)
        
        if phaseNumber == 1:
            agents = [{"role": "Macro Analyst", "color": self.macroAnalyst.color, "name": "Macro Analyst"}]
        elif phaseNumber == 2:
            agents = [
                {"role": "Bullish Value Analyst", "color": self.bullAnalyst.color, "name": "Bullish Analyst"},
                {"role": "Bearish Risk Analyst", "color": self.bearAnalyst.color, "name": "Bearish Analyst"}
            ]
        elif phaseNumber == 3:
            if isFast:
                agents = [{"role": "Impartial Portfolio Manager", "color": self.portManager.color, "name": "Portfolio Manager"}]
            else:
                agents = [
                    {"role": "Aggressive Risk Analyst", "color": self.aggRiskAnalyst.color, "name": "Aggressive Risk"},
                    {"role": "Conservative Risk Analyst", "color": self.consRiskAnalyst.color, "name": "Conservative Risk"}
                ]
        elif phaseNumber == 4:
            if isFast:
                agents = [{"role": "Impartial Portfolio Manager", "color": self.portManager.color, "name": "Portfolio Manager"}]
            else:
                agents = [
                    {"role": "Bullish Value Analyst", "color": self.bullAnalyst.color, "name": "Bullish Analyst"},
                    {"role": "Bearish Risk Analyst", "color": self.bearAnalyst.color, "name": "Bearish Analyst"}
                ]
        elif phaseNumber == 5:
            agents = [
                {"role": "Aggressive Risk Analyst", "color": self.aggRiskAnalyst.color, "name": "Aggressive Risk"},
                {"role": "Conservative Risk Analyst", "color": self.consRiskAnalyst.color, "name": "Conservative Risk"}
            ]
        elif phaseNumber == 6:
            agents = [{"role": "Impartial Portfolio Manager", "color": self.portManager.color, "name": "Portfolio Manager"}]
        elif phaseNumber == 7:
            agents = [{"role": "Impartial Portfolio Manager", "color": self.portManager.color, "name": "Portfolio Manager"}]

        emitEvent("stageStart", {
            "stageNum": phaseNumber,
            "stageName": phaseName,
            "agents": agents
        })


    def executeFastSingleEquityRating(self, config: SingleEquityRatingConfig):
        targetTicker, _, timeHorizon, _ = config.unpack()
        timeHorizonInfo = config.getTimeHorizonInfo()
        finalSubmitToolName = timeHorizonInfo["llmSubmitToolName"]

        startTime = datetime.now()
        dateStr = self.timestamp.strftime("%Y-%m-%d")
        print(f"\n{'='*70}\nStarting Fast Boardroom Evaluation for: {targetTicker}\n{'='*70}")        

        # Phase 1: Macro Environment Analysis
        self._newPhaseHeader(1, "Macro Environment Analysis")
        macroPrompt = (
            "Task: Conduct top-down macroeconomic analysis for the US financial markets.\n"
            "Use your macro-specific tools to retrieve economic indicators, headlines, and sentiment history. "
            "Present a narrative macro summary and explicitly output your overall market regime classification as BULLISH, BEARISH, or NEUTRAL."
        )
        macroRaw, macroUISummary = self.macroAnalyst.analyseAndReply(
            macroPrompt, self.toolRegistry, self.timestamp, subrole=None, requireInitialTools=True, promptArgs=timeHorizonInfo
        )
        
        # Phase 2: Specialist Research
        self._newPhaseHeader(2, f"Specialist Research on {targetTicker}")
        researchPrompt = (
            f"Macroeconomic Context:\n{macroRaw}\n\n"
            f"Task: Conduct single-stock research on ticker {targetTicker}.\n"
            f"Execute your data tools (valuation metrics, financial statements, stock price performance, company profile, etc.) to retrieve hard facts. "
            f"Present your thesis and state: explicit rating ({{permittedRatings}}), OVERWEIGHT/EQUAL-WEIGHT/UNDERWEIGHT weight, and preliminary {timeHorizonInfo['llmPriceTargets']}."
        )
        
        (bullThesisRaw, bullThesisUISummary), (bearThesisRaw, bearThesisUISummary) = self._runAgentsConcurrently(
            lambda: self.bullAnalyst.analyseAndReply(
                researchPrompt.format(permittedRatings="BUY/HOLD"), self.toolRegistry, self.timestamp, 
                subrole="research", requireInitialTools=True, promptArgs=timeHorizonInfo
            ),
            lambda: self.bearAnalyst.analyseAndReply(
                researchPrompt.format(permittedRatings="HOLD/SELL"), self.toolRegistry, self.timestamp, 
                subrole="research", requireInitialTools=True, promptArgs=timeHorizonInfo
            )
        )


        # Phase 3/6: Final Executive Decision
        self._newPhaseHeader(3, f"Final Executive Decision on {targetTicker}")
        managerPrompt = (
            f"Target Asset: {targetTicker}\n"
            f"Macro Conditions:\n{macroRaw}\n\n"
            f"Aggressive Allocation Case:\n{bullThesisRaw}\n\n"
            f"Conservative Allocation Case:\n{bearThesisRaw}\n\n"
            f"Task: Produce the final executive investment decision for {targetTicker}.\n"
            f"Weigh upside potential against solvency risks. You MUST verify your final price targets using the 'calculateDistFromCurrPrice' tool. "
            f"Include a definitive rating (BUY/HOLD/SELL), weighting (OVERWEIGHT/EQUAL-WEIGHT/UNDERWEIGHT), and {timeHorizonInfo['llmFinalLinePriceTargets']}."
        )
        self.portManager.removeTool(finalSubmitToolName)
        finalDecisionRaw, finalDecisionUISummary = self.portManager.analyseAndReply(
            managerPrompt, self.toolRegistry, self.timestamp, subrole="decision", requireInitialTools=False, promptArgs=timeHorizonInfo
        )


        # Phase 4/7: Decision Upload
        self._newPhaseHeader(4, "Decision Upload")
        uploadPrompt = (
            f"Target Asset: {targetTicker}\n"
            f"Final Decision Summary:\n{finalDecisionRaw}\n\n"
            f"Task: Upload and log the final boardroom verdict for {targetTicker}.\n"
            f"Execute the {finalSubmitToolName} tool with ticker='{targetTicker}', rating, weighting and the {timeHorizonInfo['llmFinalLinePriceTargets']} based on your final decision."
        )
        self.portManager.clearTools()
        self.portManager.addTool(finalSubmitToolName, self.toolRegistry)
        _, _ = self.portManager.analyseAndReply(
            uploadPrompt, self.toolRegistry, self.timestamp, subrole="upload", requireInitialTools=False, promptArgs=timeHorizonInfo, summarisationOverride=False
        )

        try:
            formattedExecutiveDecision = self.toolRegistry.getTool(finalSubmitToolName).toolLog[-1]
        except Exception:
            formattedExecutiveDecision = "Decision not found."        
                
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
        




    def executeCompleteSingleEquityRating(self, config: SingleEquityRatingConfig):
        targetTicker, _, timeHorizon, _ = config.unpack()
        timeHorizonInfo = config.getTimeHorizonInfo()
        finalSubmitToolName = timeHorizonInfo["llmSubmitToolName"]

        startTime = datetime.now()
        dateStr = self.timestamp.strftime("%Y-%m-%d")
        print(f"\n{'='*70}\nStarting Live Boardroom Evaluation for: {targetTicker}\n{'='*70}")        

        # Phase 1: Macro Environment Analysis
        self._newPhaseHeader(1, "Macro Environment Analysis")
        macroPrompt = (
            "Task: Conduct top-down macroeconomic analysis for the US financial markets.\n"
            "Use your macro-specific tools to retrieve economic indicators, headlines, and sentiment history. "
            "Present a narrative macro summary and explicitly output your overall market regime classification as BULLISH, BEARISH, or NEUTRAL."
        )
        macroRaw, macroUISummary = self.macroAnalyst.analyseAndReply(
            macroPrompt, self.toolRegistry, self.timestamp, subrole=None, requireInitialTools=True, promptArgs=timeHorizonInfo
        )
        
        # Phase 2: Specialist Research
        self._newPhaseHeader(2, f"Specialist Research on {targetTicker}")
        researchPrompt = (
            f"Macroeconomic Context:\n{macroRaw}\n\n"
            f"Task: Conduct single-stock research on ticker {targetTicker}.\n"
            f"Execute your data tools (valuation metrics, financial statements, stock price performance, company profile) to retrieve hard facts. "
            f"Present your thesis and state: explicit rating ({{permittedRatings}}), OVERWEIGHT/EQUAL-WEIGHT/UNDERWEIGHT weight, and preliminary {timeHorizonInfo['llmPriceTargets']}."
        )
        
        (bullThesisRaw, bullThesisUISummary), (bearThesisRaw, bearThesisUISummary) = self._runAgentsConcurrently(
            lambda: self.bullAnalyst.analyseAndReply(
                researchPrompt.format(permittedRatings="BUY/HOLD"), self.toolRegistry, self.timestamp, 
                subrole="research", requireInitialTools=True, promptArgs=timeHorizonInfo
            ),
            lambda: self.bearAnalyst.analyseAndReply(
                researchPrompt.format(permittedRatings="HOLD/SELL"), self.toolRegistry, self.timestamp, 
                subrole="research", requireInitialTools=True, promptArgs=timeHorizonInfo
            )
        )

        # Phase 3: Senior Risk Debate
        self._newPhaseHeader(3, f"Senior Risk Debate on {targetTicker}")
        aggDebatePrompt = (
            f"Macroeconomic Context:\n{macroRaw}\n\n"
            f"Bearish Analyst's Thesis on {targetTicker}:\n{bearThesisRaw}\n\n"
            f"Task: Challenge the Bearish Analyst's stance on {targetTicker}.\n"
            f"Formulate 2-3 quantitative questions challenging their downside assumptions."
        )
        consDebatePrompt = (
            f"Macroeconomic Context:\n{macroRaw}\n\n"
            f"Bullish Analyst's Thesis on {targetTicker}:\n{bullThesisRaw}\n\n"
            f"Task: Challenge the Bullish Analyst's stance on {targetTicker}.\n"
            f"Formulate 2-3 quantitative questions challenging their upside assumptions."
        )
        
        (aggQuestionsRaw, aggQuestionsUISummary), (consQuestionsRaw, consQuestionsUISummary) = self._runAgentsConcurrently(
            lambda: self.aggRiskAnalyst.analyseAndReply(
                aggDebatePrompt, self.toolRegistry, self.timestamp, "critique", requireInitialTools=False, promptArgs=timeHorizonInfo
            ),
            lambda: self.consRiskAnalyst.analyseAndReply(
                consDebatePrompt, self.toolRegistry, self.timestamp, "critique", requireInitialTools=False, promptArgs=timeHorizonInfo
            )
        )
        
        # Phase 4: Analyst Defense
        self._newPhaseHeader(4, f"Analyst Defense on {targetTicker}")
        bullDefensePrompt = (
            f"Questions Posed by Conservative Risk Analyst:\n{consQuestionsRaw}\n\n"
            f"Task: Defend your bullish thesis and price targets for {targetTicker}.\n"
            f"Answer each question quantitatively using your tools or Python models if needed. Revise your thesis, targets, or rating if substantiated deficiencies were highlighted."
        )
        bearDefensePrompt = (
            f"Questions Posed by Aggressive Risk Analyst:\n{aggQuestionsRaw}\n\n"
            f"Task: Defend your bearish risk analysis and price targets for {targetTicker}.\n"
            f"Answer each question quantitatively using your tools or Python models if needed. Revise your risk assessment, targets, or rating if substantiated upside catalysts were highlighted."
        )
        
        (bullDefenseRaw, bullDefenseUISummary), (bearDefenseRaw, bearDefenseUISummary) = self._runAgentsConcurrently(
            lambda: self.bullAnalyst.analyseAndReply(
                bullDefensePrompt, self.toolRegistry, self.timestamp, "defense", requireInitialTools=False, promptArgs=timeHorizonInfo
            ),
            lambda: self.bearAnalyst.analyseAndReply(
                bearDefensePrompt, self.toolRegistry, self.timestamp, "defense", requireInitialTools=False, promptArgs=timeHorizonInfo
            )
        )

        # Phase 5: Q&A Based Proposals
        self._newPhaseHeader(5, f"Q&A-Based Proposals on {targetTicker}")
        aggProposalPrompt = (
            f"Bearish Analyst's Defense:\n{bearDefenseRaw}\n\n"
            f"Task: Formulate your final aggressive allocation proposal for {targetTicker}.\n"
            f"Propose your {timeHorizonInfo['llmPriceTargets']} and position weight (OVERWEIGHT/EQUAL-WEIGHT/UNDERWEIGHT), justifying your high-upside growth assumptions."
        )
        consProposalPrompt = (
            f"Bullish Analyst's Defense:\n{bullDefenseRaw}\n\n"
            f"Task: Formulate your final conservative allocation proposal for {targetTicker}.\n"
            f"Propose your {timeHorizonInfo['llmPriceTargets']} and position weight (OVERWEIGHT/EQUAL-WEIGHT/UNDERWEIGHT), incorporating a robust margin of safety."
        )
        
        (aggProposalRaw, aggProposalUISummary), (consProposalRaw, consProposalUISummary) = self._runAgentsConcurrently(
            lambda: self.aggRiskAnalyst.analyseAndReply(
                aggProposalPrompt, self.toolRegistry, self.timestamp, subrole="proposal", requireInitialTools=False, promptArgs=timeHorizonInfo
            ),
            lambda: self.consRiskAnalyst.analyseAndReply(
                consProposalPrompt, self.toolRegistry, self.timestamp, subrole="proposal", requireInitialTools=False, promptArgs=timeHorizonInfo
            )
        )

        

        # Phase 6: Final Executive Decision
        self._newPhaseHeader(6, f"Final Executive Decision on {targetTicker}")
        managerPrompt = (
            f"Target Asset: {targetTicker}\n"
            f"Macro Conditions:\n{macroRaw}\n\n"
            f"Aggressive Allocation Case:\n{aggProposalRaw}\n\n"
            f"Conservative Allocation Case:\n{consProposalRaw}\n\n"
            f"Task: Produce the final executive investment decision for {targetTicker}.\n"
            f"Weigh upside potential against solvency risks. You MUST verify your final price targets using the 'calculateDistFromCurrPrice' tool. "
            f"Include a definitive rating (BUY/HOLD/SELL), weighting (OVERWEIGHT/EQUAL-WEIGHT/UNDERWEIGHT), and {timeHorizonInfo['llmFinalLinePriceTargets']}."
        )
        self.portManager.removeTool(finalSubmitToolName)
        finalDecisionRaw, finalDecisionUISummary = self.portManager.analyseAndReply(
            managerPrompt, self.toolRegistry, self.timestamp, subrole="decision", requireInitialTools=True, promptArgs=timeHorizonInfo
        )

        
        # Phase 7: Decision Upload
        self._newPhaseHeader(7, "Decision Upload")
        uploadPrompt = (
            f"Target Asset: {targetTicker}\n"
            f"Final Decision Summary:\n{finalDecisionRaw}\n\n"
            f"Task: Upload and log the final boardroom verdict for {targetTicker}.\n"
            f"Execute the {finalSubmitToolName} tool with ticker='{targetTicker}', rating, weighting and the {timeHorizonInfo['llmFinalLinePriceTargets']} based on your final decision."
        )
        self.portManager.clearTools()
        self.portManager.addTool(finalSubmitToolName, self.toolRegistry)
        _, _ = self.portManager.analyseAndReply(
            uploadPrompt, self.toolRegistry, self.timestamp, subrole="upload", requireInitialTools=True, promptArgs=timeHorizonInfo
        )

        try:
            formattedExecutiveDecision = self.toolRegistry.getTool(finalSubmitToolName).toolLog[-1]
        except Exception:
            formattedExecutiveDecision = "Decision not found."



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
        

    def executeSingleEquityRating(self, config: SingleEquityRatingConfig):
        if self.clientDuo is None:
            raise ValueError("ClientDuo is not assigned. Please assign a ClientDuo before executing the boardroom.")

        self.clientDuo.boardroomClient.newTask()
        if self.clientDuo.summaryClient is not self.clientDuo.boardroomClient:
            self.clientDuo.summaryClient.newTask()

        if config.fastMode:
            self.executeFastSingleEquityRating(config)
        else:
            self.executeCompleteSingleEquityRating(config)


    def execute(self, config: BoardroomConfig):
        if isinstance(config, SingleEquityRatingConfig):
            self.executeSingleEquityRating(config)
        else:
            raise NotImplementedError(f"BoardroomConfig type '{type(config).__name__}' is not supported yet.")




def generateBoardroom(toolRegistry: ToolRegistry, timestamp: pd.Timestamp) -> BoardroomEngine:
    timestampStr = timestamp.strftime("%Y-%m-%d")
    toolMap = toolRegistry.getToolMap()

    macroAgent = FinancialAgent(
        agentRole="Macro Analyst",
        tools=[
            toolMap["fetchMacroContext"],
            toolMap["fetchMacroNews"],
            toolMap["fetchMacroSentimentHistory"],
            toolMap["executePythonCalculation"]
        ],
        ansiColor=ANSI.CYAN,
        dateStr=timestampStr
    )

    bullAgent = FinancialAgent(
        agentRole="Bullish Value Analyst",
        tools=[
            toolMap["fetchCompanyProfile"],
            toolMap["fetchCompanyValuationMetrics"],
            toolMap["fetchIncomeStatement"],
            toolMap["fetchBalanceSheet"],
            toolMap["fetchCashFlowStatement"],
            # toolMap["fetchStatementOfEquity"],
            # toolMap["fetchComprehensiveIncomeStatement"],
            toolMap["fetchStockPricePerformance"],
            toolMap["fetchCompanyRecentNews"],
            toolMap["fetchTickerSentimentHistory"],
            toolMap["fetchSentimentDivergence"],
            toolMap["calculateDistFromCurrPrice"],
            toolMap["executePythonCalculation"]
        ],
        ansiColor=ANSI.GREEN,
        dateStr=timestampStr
    )

    bearAgent = FinancialAgent(
        agentRole="Bearish Risk Analyst",
        tools=[
            toolMap["fetchCompanyProfile"],
            toolMap["fetchCompanyValuationMetrics"],
            toolMap["fetchIncomeStatement"],
            toolMap["fetchBalanceSheet"],
            toolMap["fetchCashFlowStatement"],
            # toolMap["fetchStatementOfEquity"],
            # toolMap["fetchComprehensiveIncomeStatement"],
            toolMap["fetchStockPricePerformance"],
            toolMap["fetchCompanyRecentNews"],
            toolMap["fetchTickerSentimentHistory"],
            toolMap["fetchSentimentDivergence"],
            toolMap["calculateDistFromCurrPrice"],
            toolMap["executePythonCalculation"]
        ],
        ansiColor=ANSI.RED,
        dateStr=timestampStr
    )

    aggRiskAnalystAgent = FinancialAgent(
        agentRole="Aggressive Risk Analyst",
        tools=[
            toolMap["fetchCompanyProfile"],
            toolMap["fetchCompanyValuationMetrics"],
            toolMap["fetchIncomeStatement"],
            toolMap["fetchBalanceSheet"],
            toolMap["fetchCashFlowStatement"],
            # toolMap["fetchStatementOfEquity"],
            # toolMap["fetchComprehensiveIncomeStatement"],
            toolMap["fetchStockPricePerformance"],
            toolMap["fetchCompanyRecentNews"],
            toolMap["fetchTickerSentimentHistory"],
            toolMap["fetchSentimentDivergence"],
            toolMap["calculateDistFromCurrPrice"],
            toolMap["executePythonCalculation"]
        ],
        ansiColor=ANSI.YELLOW,
        dateStr=timestampStr
    )

    consRiskAnalystAgent = FinancialAgent(
        agentRole="Conservative Risk Analyst",
        tools=[
            toolMap["fetchCompanyProfile"],
            toolMap["fetchCompanyValuationMetrics"],
            toolMap["fetchIncomeStatement"],
            toolMap["fetchBalanceSheet"],
            toolMap["fetchCashFlowStatement"],
            # toolMap["fetchStatementOfEquity"],
            # toolMap["fetchComprehensiveIncomeStatement"],
            toolMap["fetchStockPricePerformance"],
            toolMap["fetchCompanyRecentNews"],
            toolMap["fetchTickerSentimentHistory"],
            toolMap["fetchSentimentDivergence"],
            toolMap["calculateDistFromCurrPrice"],
            toolMap["executePythonCalculation"]
        ],
        ansiColor=ANSI.BLUE,
        dateStr=timestampStr
    )

    portfolioManager = FinancialAgent(
        agentRole="Impartial Portfolio Manager",
        tools=[
            toolMap["fetchCompanyProfile"],
            toolMap["executePythonCalculation"],
            toolMap["calculateDistFromCurrPrice"],
            toolMap["confirmBoardroomDecisionLongTerm"]
        ],
        ansiColor=ANSI.MAGENTA,
        dateStr=timestampStr
    )    

    boardroom = BoardroomEngine(
        agents={
            "macroAnalyst": macroAgent,
            "bullAnalyst": bullAgent,
            "bearAnalyst": bearAgent,
            "aggRiskAnalyst": aggRiskAnalystAgent,
            "consRiskAnalyst": consRiskAnalystAgent,
            "portManager": portfolioManager,
        }, 
        timestamp=timestamp,
        toolRegistry=toolRegistry
    )

    return boardroom

