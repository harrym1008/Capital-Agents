import re
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from cli.ansi import ANSI
from llm.client_duo import ClientDuo
from llmtools.tool_registry import ToolRegistry, Tool
from llm.agents.agent import FinancialAgent
from llm.agents.agent_prompts import buildSpokespersonSysPrompt, buildSpecialistQnASysPrompt
from boardroom.boardroom_config import BoardroomConfig, SingleEquityRatingConfig, TimeHorizon, BoardroomPace
from ui.ui_hooks import getCurrentStage, isStopRequested, setCurrentStage, setCurrentAgent, setAgentPhase, emitEvent, SimulationStoppedException


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

        self.agentsList = list(agents.values())
        self.macroAnalyst = agents.get("macroAnalyst")
        self.bullAnalyst = agents.get("bullAnalyst")
        self.bearAnalyst = agents.get("bearAnalyst")
        self.aggRiskAnalyst = agents.get("aggRiskAnalyst")
        self.consRiskAnalyst = agents.get("consRiskAnalyst")
        self.portManager = agents.get("portManager")
        self.oneShotAnalyst = agents.get("oneShotAnalyst")
        self.spokesperson = agents.get("spokesperson")

        self.specialistMap = {
            "Macro Analyst": self.macroAnalyst,
            "Bullish Value Analyst": self.bullAnalyst,
            "Bearish Risk Analyst": self.bearAnalyst,
            "Aggressive Risk Analyst": self.aggRiskAnalyst,
            "Conservative Risk Analyst": self.consRiskAnalyst,
            "Impartial Portfolio Manager": self.portManager,
            "One-Shot Analyst": self.oneShotAnalyst
        }

        self.lastConfig: Optional[SingleEquityRatingConfig] = None
        self.fullConvSummary: str = ""
        self.qnaHistory: list = []
        self.qnaTurns: list = []

        self.configureTransferToolSchema(BoardroomPace.COMPLETE)


    def deleteQnATurn(self, turnIndex: int) -> bool:
        if 0 <= turnIndex < len(self.qnaTurns):
            targetTurn = self.qnaTurns[turnIndex]
            
            # Prune spokesperson message history
            if self.spokesperson:
                spokespersonTargetLen = targetTurn.get("spokespersonHistoryLen", 0)
                self.spokesperson.messageHistory = self.spokesperson.messageHistory[:spokespersonTargetLen]
            
            # Prune specialist message histories
            for role, targetLen in targetTurn.get("specialistHistoryLens", {}).items():
                agent = self.specialistMap.get(role)
                if agent is not None:
                    agent.messageHistory = agent.messageHistory[:targetLen]
            
            # Prune qnaTurns list
            self.qnaTurns = self.qnaTurns[:turnIndex]
            return True
        return False


    @staticmethod
    def getActiveSpecialistRoles(pace: BoardroomPace) -> List[str]:
        if pace == BoardroomPace.ONE_SHOT:
            return ["One-Shot Analyst"]
        elif pace == BoardroomPace.FAST:
            return [
                "Macro Analyst",
                "Bullish Value Analyst",
                "Bearish Risk Analyst",
                "Impartial Portfolio Manager"
            ]
        else:
            return [
                "Macro Analyst",
                "Bullish Value Analyst",
                "Bearish Risk Analyst",
                "Aggressive Risk Analyst",
                "Conservative Risk Analyst",
                "Impartial Portfolio Manager"
            ]


    def configureTransferToolSchema(self, pace: BoardroomPace = BoardroomPace.COMPLETE) -> None:
        activeRoles = self.getActiveSpecialistRoles(pace)
        transferTool = self.toolRegistry.getTool("transferToAgent") if self.toolRegistry else None
        if transferTool:
            transferTool.paramSchema = {
                "type": "object",
                "properties": {
                    "agentRole": {
                        "type": "string",
                        "enum": activeRoles,
                        "description": f"The exact name of the active specialist boardroom agent to transfer to ({', '.join(activeRoles)})."
                    },
                    "transferMessage": {
                        "type": "string",
                        "description": "A clear, concise instruction or summary of the question for the specialist agent to answer."
                    }
                },
                "required": ["agentRole", "transferMessage"]
            }


    def assignClientDuo(self, clientDuo: ClientDuo):
        self.clientDuo = clientDuo
        for agent in self.agentsList:
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
            except Exception:
                if isStopRequested():
                    for f in futures:
                        f.cancel()
                    raise SimulationStoppedException("Simulation stopped by user.")
                raise
            return results


    def _newPhaseHeader(self, phaseNumber, phaseName, pace: BoardroomPace = BoardroomPace.COMPLETE):
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

        if pace == BoardroomPace.ONE_SHOT:
            analystColor = self.oneShotAnalyst.color if self.oneShotAnalyst is not None else self.macroAnalyst.color
            agents = [{"role": "One-Shot Analyst", "color": analystColor, "name": "One-Shot Analyst"}]
            emitEvent("stageStart", {
                "stageNum": phaseNumber,
                "stageName": phaseName,
                "agents": agents
            })
            return

        isFast = pace == BoardroomPace.FAST
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


    def executeOneShotSingleEquityRating(self, config: SingleEquityRatingConfig):
        targetTicker = config.ticker
        pace = config.boardroomPace
        timeHorizonInfo = config.getTimeHorizonInfo()
        finalSubmitToolName = timeHorizonInfo["llmSubmitToolName"]

        startTime = datetime.now()
        dateStr = self.timestamp.strftime("%Y-%m-%d")
        print(f"\n{'='*70}\nStarting One-Shot Boardroom Evaluation for: {targetTicker}\n{'='*70}")        

        # Phase 1: One Shot Analysis
        self._newPhaseHeader(1, "One-Shot Analysis", pace)
        oneShotPrompt = (
            f"Task: Conduct your complete analysis of macro conditions, single-stock reserach, risk assessment and final " 
            f"executive decision in one go for the ticker: {targetTicker}.\n"
            f"Execute your data tools (macro, financials, valuation, statements, stock performance, news) to retrieve hard facts. "
            f"Present your final executive decision with explicit rating (BUY/HOLD/SELL), weighting (OVERWEIGHT/EQUAL-WEIGHT/UNDERWEIGHT), and {timeHorizonInfo['llmFinalLinePriceTargets']}." 
            
        )
        oneShotRaw, _ = self.oneShotAnalyst.analyseAndReply(
            oneShotPrompt, self.toolRegistry, self.timestamp, config, subrole="analysis", requireInitialTools=True
        )

        # Phase 2: Decision Upload
        self._newPhaseHeader(2, "Decision Upload", pace)
        uploadPrompt = (
            f"Target Asset: {targetTicker}\n"
            f"Decision:\n{oneShotRaw}\n\n"
            f"Task: Upload and log the final boardroom verdict for {targetTicker}.\n"
            f"Execute the {finalSubmitToolName} tool with ticker='{targetTicker}', rating, weighting and the {timeHorizonInfo['llmFinalLinePriceTargets']} based on your final decision."
        )
        origTools = list(self.oneShotAnalyst.tools)
        self.oneShotAnalyst.clearTools()
        self.oneShotAnalyst.addTool(finalSubmitToolName, self.toolRegistry)
        _, _ = self.oneShotAnalyst.analyseAndReply(
            uploadPrompt, self.toolRegistry, self.timestamp, config, subrole="upload", requireInitialTools=False, summarisationOverride=False
        )
        self.oneShotAnalyst.tools = origTools

        try:
            formattedExecutiveDecision = self.toolRegistry.getTool(finalSubmitToolName).toolLog[-1]
        except Exception:
            formattedExecutiveDecision = "Decision not found."        
                
        endTime = datetime.now()
        timeTaken = endTime - startTime

        print()
        self._newPhaseHeader(0, f"Final Boardroom Summary on {targetTicker}", pace)                    

        convSummary = (
            f"\n{ANSI.BOLD}{self.oneShotAnalyst.color}Final Executive Decision:\n{ANSI.RESET}{oneShotRaw}\n"
            f"\n{formattedExecutiveDecision}\n"
            f"Time taken for boardroom discussion: {timeTaken.seconds//60} mins {timeTaken.seconds%60} secs\n"
        )

        print(convSummary)

        with open(f"output\\{targetTicker}_oneshot_{startTime.strftime('%Y-%m-%d_%H-%M-%S')}.ans", "w", encoding="utf-8") as f:
            f.write(convSummary)

        self.lastConfig = config
        self.fullConvSummary = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', convSummary)





    def executeFastSingleEquityRating(self, config: SingleEquityRatingConfig):
        targetTicker = config.ticker
        pace = config.boardroomPace
        timeHorizonInfo = config.getTimeHorizonInfo()
        finalSubmitToolName = timeHorizonInfo["llmSubmitToolName"]

        startTime = datetime.now()
        dateStr = self.timestamp.strftime("%Y-%m-%d")
        print(f"\n{'='*70}\nStarting Fast Boardroom Evaluation for: {targetTicker}\n{'='*70}")        

        # Phase 1: Macro Environment Analysis
        self._newPhaseHeader(1, "Macro Environment Analysis", pace)
        macroPrompt = (
            "Task: Conduct top-down macroeconomic analysis for the US financial markets.\n"
            "Use your macro-specific tools to retrieve economic indicators, headlines, and sentiment history. "
            "Present a narrative macro summary and explicitly output your overall market regime classification as BULLISH, BEARISH, or NEUTRAL."
        )
        macroRaw, macroUISummary = self.macroAnalyst.analyseAndReply(
            macroPrompt, self.toolRegistry, self.timestamp, config, subrole=None, requireInitialTools=True
        )
        
        # Phase 2: Specialist Research
        self._newPhaseHeader(2, f"Specialist Research on {targetTicker}", pace)
        researchPrompt = (
            f"Macroeconomic Context:\n{macroRaw}\n\n"
            f"Task: Conduct single-stock research on ticker {targetTicker}.\n"
            f"Execute your data tools (valuation metrics, financial statements, stock price performance, company profile, etc.) to retrieve hard facts. "
            f"Present your thesis and state: explicit rating ({{permittedRatings}}), OVERWEIGHT/EQUAL-WEIGHT/UNDERWEIGHT weight, and preliminary {timeHorizonInfo['llmPriceTargets']}."
        )
        
        (bullThesisRaw, bullThesisUISummary), (bearThesisRaw, bearThesisUISummary) = self._runAgentsConcurrently(
            lambda: self.bullAnalyst.analyseAndReply(
                researchPrompt.format(permittedRatings="BUY/HOLD"), self.toolRegistry, self.timestamp, 
                config, subrole="research", requireInitialTools=True
            ),
            lambda: self.bearAnalyst.analyseAndReply(
                researchPrompt.format(permittedRatings="HOLD/SELL"), self.toolRegistry, self.timestamp, 
                config, subrole="research", requireInitialTools=True
            )
        )


        # Phase 3/6: Final Executive Decision
        self._newPhaseHeader(3, f"Final Executive Decision on {targetTicker}", pace)
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
            managerPrompt, self.toolRegistry, self.timestamp, config, subrole="decision", requireInitialTools=False
        )


        # Phase 4/7: Decision Upload
        self._newPhaseHeader(4, "Decision Upload", pace)
        uploadPrompt = (
            f"Target Asset: {targetTicker}\n"
            f"Final Decision Summary:\n{finalDecisionRaw}\n\n"
            f"Task: Upload and log the final boardroom verdict for {targetTicker}.\n"
            f"Execute the {finalSubmitToolName} tool with ticker='{targetTicker}', rating, weighting and the {timeHorizonInfo['llmFinalLinePriceTargets']} based on your final decision."
        )
        self.portManager.clearTools()
        self.portManager.addTool(finalSubmitToolName, self.toolRegistry)
        _, _ = self.portManager.analyseAndReply(
            uploadPrompt, self.toolRegistry, self.timestamp, config, subrole="upload", requireInitialTools=False, summarisationOverride=False
        )

        try:
            formattedExecutiveDecision = self.toolRegistry.getTool(finalSubmitToolName).toolLog[-1]
        except Exception:
            formattedExecutiveDecision = "Decision not found."        
                
        endTime = datetime.now()
        timeTaken = endTime - startTime

        print()
        self._newPhaseHeader(0, f"Final Boardroom Summary on {targetTicker}", pace)                    

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
        
        self.lastConfig = config
        self.fullConvSummary = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', fullConvSummary)




    def executeCompleteSingleEquityRating(self, config: SingleEquityRatingConfig):
        targetTicker = config.ticker
        pace = config.boardroomPace
        timeHorizonInfo = config.getTimeHorizonInfo()
        finalSubmitToolName = timeHorizonInfo["llmSubmitToolName"]

        startTime = datetime.now()
        dateStr = self.timestamp.strftime("%Y-%m-%d")
        print(f"\n{'='*70}\nStarting Live Boardroom Evaluation for: {targetTicker}\n{'='*70}")        

        # Phase 1: Macro Environment Analysis
        self._newPhaseHeader(1, "Macro Environment Analysis", pace)
        macroPrompt = (
            "Task: Conduct top-down macroeconomic analysis for the US financial markets.\n"
            "Use your macro-specific tools to retrieve economic indicators, headlines, and sentiment history. "
            "Present a narrative macro summary and explicitly output your overall market regime classification as BULLISH, BEARISH, or NEUTRAL."
        )
        macroRaw, macroUISummary = self.macroAnalyst.analyseAndReply(
            macroPrompt, self.toolRegistry, self.timestamp, config, subrole=None, requireInitialTools=True
        )
        
        # Phase 2: Specialist Research
        self._newPhaseHeader(2, f"Specialist Research on {targetTicker}", pace)
        researchPrompt = (
            f"Macroeconomic Context:\n{macroRaw}\n\n"
            f"Task: Conduct single-stock research on ticker {targetTicker}.\n"
            f"Execute your data tools (valuation metrics, financial statements, stock price performance, company profile) to retrieve hard facts. "
            f"Present your thesis and state: explicit rating ({{permittedRatings}}), OVERWEIGHT/EQUAL-WEIGHT/UNDERWEIGHT weight, and preliminary {timeHorizonInfo['llmPriceTargets']}."
        )
        
        (bullThesisRaw, bullThesisUISummary), (bearThesisRaw, bearThesisUISummary) = self._runAgentsConcurrently(
            lambda: self.bullAnalyst.analyseAndReply(
                researchPrompt.format(permittedRatings="BUY/HOLD"), self.toolRegistry, self.timestamp, 
                config, subrole="research", requireInitialTools=True
            ),
            lambda: self.bearAnalyst.analyseAndReply(
                researchPrompt.format(permittedRatings="HOLD/SELL"), self.toolRegistry, self.timestamp, 
                config, subrole="research", requireInitialTools=True
            )
        )

        # Phase 3: Senior Risk Debate
        self._newPhaseHeader(3, f"Senior Risk Debate on {targetTicker}", pace)
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
                aggDebatePrompt, self.toolRegistry, self.timestamp, config, "critique", requireInitialTools=False
            ),
            lambda: self.consRiskAnalyst.analyseAndReply(
                consDebatePrompt, self.toolRegistry, self.timestamp, config, "critique", requireInitialTools=False
            )
        )
        
        # Phase 4: Analyst Defense
        self._newPhaseHeader(4, f"Analyst Defense on {targetTicker}", pace)
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
                bullDefensePrompt, self.toolRegistry, self.timestamp, config, "defense", requireInitialTools=False
            ),
            lambda: self.bearAnalyst.analyseAndReply(
                bearDefensePrompt, self.toolRegistry, self.timestamp, config, "defense", requireInitialTools=False
            )
        )

        # Phase 5: Q&A Based Proposals
        self._newPhaseHeader(5, f"Q&A-Based Proposals on {targetTicker}", pace)
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
                aggProposalPrompt, self.toolRegistry, self.timestamp, config, subrole="proposal", requireInitialTools=False
            ),
            lambda: self.consRiskAnalyst.analyseAndReply(
                consProposalPrompt, self.toolRegistry, self.timestamp, config, subrole="proposal", requireInitialTools=False
            )
        )

        

        # Phase 6: Final Executive Decision
        self._newPhaseHeader(6, f"Final Executive Decision on {targetTicker}", pace)
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
            managerPrompt, self.toolRegistry, self.timestamp, config, subrole="decision", requireInitialTools=True
        )

        
        # Phase 7: Decision Upload
        self._newPhaseHeader(7, "Decision Upload", pace)
        uploadPrompt = (
            f"Target Asset: {targetTicker}\n"
            f"Final Decision Summary:\n{finalDecisionRaw}\n\n"
            f"Task: Upload and log the final boardroom verdict for {targetTicker}.\n"
            f"Execute the {finalSubmitToolName} tool with ticker='{targetTicker}', rating, weighting and the {timeHorizonInfo['llmFinalLinePriceTargets']} based on your final decision."
        )
        self.portManager.clearTools()
        self.portManager.addTool(finalSubmitToolName, self.toolRegistry)
        _, _ = self.portManager.analyseAndReply(
            uploadPrompt, self.toolRegistry, self.timestamp, config, subrole="upload", requireInitialTools=True, summarisationOverride=False
        )

        try:
            formattedExecutiveDecision = self.toolRegistry.getTool(finalSubmitToolName).toolLog[-1]
        except Exception:
            formattedExecutiveDecision = "Decision not found."



        endTime = datetime.now()
        timeTaken = endTime - startTime

        print()
        self._newPhaseHeader(0, f"Final Boardroom Summary on {targetTicker}", pace)   
            
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

        self.lastConfig = config
        self.fullConvSummary = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', fullConvSummary)
        

    def executeSingleEquityRating(self, config: SingleEquityRatingConfig):
        if self.clientDuo is None:
            raise ValueError("ClientDuo is not assigned. Please assign a ClientDuo before executing the boardroom.")

        self.configureTransferToolSchema(config.boardroomPace)

        self.clientDuo.boardroomClient.newTask()
        if self.clientDuo.summaryClient is not self.clientDuo.boardroomClient:
            self.clientDuo.summaryClient.newTask()

        if config.boardroomPace == BoardroomPace.ONE_SHOT:
            self.executeOneShotSingleEquityRating(config)
        elif config.boardroomPace == BoardroomPace.FAST:
            self.executeFastSingleEquityRating(config)
        else:
            self.executeCompleteSingleEquityRating(config)


    def execute(self, config: BoardroomConfig):
        if isinstance(config, SingleEquityRatingConfig):
            self.executeSingleEquityRating(config)
        else:
            raise NotImplementedError(f"BoardroomConfig type '{type(config).__name__}' is not supported yet.")


    def executeSpecialistTransfer(self, agentRole: str, transferMessage: str, config: SingleEquityRatingConfig) -> Dict[str, Any]:
        specialist = self.specialistMap.get(agentRole)
        if not specialist:
            return {"error": f"Specialist agent '{agentRole}' not found."}

        if self.clientDuo and not specialist.mainApiClient:
            specialist.setClientDuo(self.clientDuo)

        setCurrentStage("qa")

        dateStr = config.simulatedDateStr if config and config.simulatedDateStr else self.timestamp.strftime("%Y-%m-%d")
        sysPrompt = buildSpecialistQnASysPrompt(
            dateStr=dateStr,
            agentRole=agentRole,
            toolsStr=specialist.getSpecificToolsStr(),
            promptArgs=config.getPromptArgs() if config else None
        )

        incomingPrompt = (
            f"The Boardroom Spokesperson has transferred the following user question to you:\n\n"
            f"Transfer Request: {transferMessage}\n\n"
            f"Task: Answer the user's question directly from your role as {agentRole}. "
            f"Rely on your previous thinking steps, tool outputs, and message history from earlier stages."
        )

        try:
            rawResponse, _ = specialist.analyseAndReply(
                incomingMessage=incomingPrompt,
                toolRegistry=self.toolRegistry,
                timestamp=self.timestamp,
                config=config,
                subrole="qa",
                requireInitialTools=False,
                summarisationOverride=False,
                sysPromptOverride=sysPrompt
            )
            return {
                "status": "success",
                "specialist": agentRole,
                "response": rawResponse
            }
        except Exception as e:
            return {
                "status": "error",
                "specialist": agentRole,
                "error": str(e)
            }
        finally:
            emitEvent("agentRunEnd", {
                "agentRole": agentRole,
                "phase": "raw",
                "stageNum": "qa"
            })
            if self.spokesperson:
                setCurrentAgent("Boardroom Spokesperson", self.spokesperson.color)
            setCurrentStage("qa")
            setAgentPhase("raw")


    def processQnAQuery(self, query: str, config: Optional[SingleEquityRatingConfig] = None):
        if config is None:
            config = self.lastConfig

        if config is None:
            config = SingleEquityRatingConfig(
                ticker="UNKNOWN",
                simulatedDateStr=self.timestamp.strftime("%Y-%m-%d"),
                timeHorizon=TimeHorizon.LONG,
                boardroomPace=BoardroomPace.FAST
            )

        activeRoles = self.getActiveSpecialistRoles(config.boardroomPace)
        self.configureTransferToolSchema(config.boardroomPace)

        transferTool = self.toolRegistry.getTool("transferToAgent") if self.toolRegistry else None
        self.spokesperson.tools = [transferTool] if transferTool else []
        if transferTool:
            transferTool.toolLog.clear()

        setCurrentStage("qa")

        contextStr = self.fullConvSummary or "No context found!"
        dateStr = config.simulatedDateStr if config.simulatedDateStr else self.timestamp.strftime("%Y-%m-%d")
        sysPrompt = buildSpokespersonSysPrompt(
            dateStr=dateStr,
            toolsStr="transferToAgent",
            boardroomContextStr=contextStr,
            promptArgs=config.getPromptArgs(),
            activeRoles=activeRoles
        )

        try:
            turnRecord = {
                "turnIndex": len(self.qnaTurns),
                "userQuery": query,
                "spokespersonHistoryLen": len(self.spokesperson.messageHistory) if self.spokesperson else 0,
                "specialistHistoryLens": {
                    role: len(agent.messageHistory)
                    for role, agent in self.specialistMap.items()
                    if agent is not None
                }
            }
            self.qnaTurns.append(turnRecord)
            self.qnaHistory.append({"role": "user", "content": query})

            spokespersonPrompt = (
                f"User Question: {query}\n\n"
                f"Task: Review the question against the completed boardroom discussion context.\n"
                f"- If this is a simple or high-level inquiry, answer directly.\n"
                f"- If this requires specialist financial depth, quantitative valuation, macro analysis, or risk challenge, "
                f"delegate to the specialist(s) using the 'transferToAgent' tool at the end of your response."
            )

            rawSpokespersonResponse, _ = self.spokesperson.analyseAndReply(
                incomingMessage=spokespersonPrompt,
                toolRegistry=self.toolRegistry,
                timestamp=self.timestamp,
                config=config,
                subrole="qa",
                requireInitialTools=False,
                summarisationOverride=False,
                sysPromptOverride=sysPrompt
            )

            self.qnaHistory.append({"role": "assistant", "content": rawSpokespersonResponse})

            # Check if transfers were logged during this turn
            if transferTool and transferTool.toolLog:
                for transferCall in list(transferTool.toolLog):
                    agentRole = transferCall.get("agentRole")
                    transferMessage = transferCall.get("transferMessage")
                    if agentRole and transferMessage:
                        self.executeSpecialistTransfer(agentRole, transferMessage, config)

        except SimulationStoppedException:
            emitEvent("simStopped", {"message": "Q&A stopped by user."})
        except Exception as e:
            if isStopRequested():
                emitEvent("simStopped", {"message": "Q&A stopped by user."})
            else:
                traceback.print_exc()
                emitEvent("error", {"message": f"Q&A Error: {str(e)}"})
        finally:
            emitEvent("agentRunEnd", {
                "agentRole": "Boardroom Spokesperson",
                "phase": "raw",
                "stageNum": "qa"
            })
            emitEvent("qaComplete", {"stageNum": "qa"})




