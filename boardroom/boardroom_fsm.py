from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from ui.ui_hooks import isStopRequested, SimulationStoppedException


# Boardroom Context represents the state and data shared across different stages of the boardroom simulation
class BoardroomContext:
    def __init__(self, engine: Any, config: Any):
        self.engine = engine
        self.config = config
        self.data: Dict[str, Any] = {}
        self.startTime: datetime = datetime.now()

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def has(self, key: str) -> bool:
        return key in self.data

    @property
    def pace(self) -> Any:
        if hasattr(self.config, "boardroomPace"):
            return self.config.boardroomPace
        if isinstance(self.config, dict):
            return self.config.get("boardroomPace")
        return getattr(self.config, "boardroomPace", None)

    def executeMandatedTool(
        self,
        agent: Any,
        initialPrompt: str,
        mandatedToolName: str,
        subrole: Optional[str] = None,
        maxRetries: Union[Tuple[int, int], int] = (8, 4),
        summarisationOverride: Optional[bool] = None,
        requireInitialTools: bool = True,
        confirmationPrompt: Optional[str] = None,
        modeOverride: Optional[str] = None
    ) -> Tuple[str, str]:
        return self.engine.executeMandatedToolStage(
            agent=agent,
            initialPrompt=initialPrompt,
            mandatedToolName=mandatedToolName,
            config=self.config,
            subrole=subrole,
            maxRetries=maxRetries,
            summarisationOverride=summarisationOverride,
            requireInitialTools=requireInitialTools,
            confirmationPrompt=confirmationPrompt,
            modeOverride=modeOverride
        )


# Abstract base class for a stage in the boardroom FSM, which defines the interface for executing stage logic and determining the next stage
class BoardroomStage(ABC):
    def __init__(
        self, 
        stageId: str, 
        phaseNumber: int, 
        phaseName: str, 
        customAgents: Optional[List[Dict[str, str]]] = None
    ):
        self.stageId = stageId
        self.phaseNumber = phaseNumber
        self.phaseName = phaseName
        self.customAgents = customAgents

    @abstractmethod
    def execute(self, context: BoardroomContext) -> Optional[str]:
        # Execute stage logic using the provided context, or return the next stageId for dynamic branching
        pass


# Specialised BoardroomStage for stages requiring a mandatory tool execution with retries and confirmation
class MandatedToolStage(BoardroomStage, ABC):
    def __init__(
        self,
        stageId: str,
        phaseNumber: int,
        phaseName: str,
        agentName: str,
        mandatedToolName: str,
        subrole: Optional[str] = None,
        maxRetries: Union[Tuple[int, int], int] = (8, 4),
        requireInitialTools: bool = True,
        customAgents: Optional[List[Dict[str, str]]] = None
    ):
        super().__init__(
            stageId=stageId,
            phaseNumber=phaseNumber,
            phaseName=phaseName,
            customAgents=customAgents
        )
        self.agentName = agentName
        self.mandatedToolName = mandatedToolName
        self.subrole = subrole
        self.maxRetries = maxRetries
        self.requireInitialTools = requireInitialTools

    @abstractmethod
    def buildPrompt(self, context: BoardroomContext) -> str:
        pass

    def buildConfirmationPrompt(self, context: BoardroomContext) -> Optional[str]:
        return None

    def execute(self, context: BoardroomContext) -> Optional[str]:
        agent = getattr(context.engine, self.agentName, None)
        if agent is None and hasattr(context.engine, "agents"):
            agent = context.engine.agents.get(self.agentName)
        if not agent:
            raise ValueError(f"Agent '{self.agentName}' not found on engine for stage '{self.stageId}'.")

        prompt = self.buildPrompt(context)
        confPrompt = self.buildConfirmationPrompt(context)

        rawResult, uiSummary = context.executeMandatedTool(
            agent=agent,
            initialPrompt=prompt,
            mandatedToolName=self.mandatedToolName,
            subrole=self.subrole,
            maxRetries=self.maxRetries,
            requireInitialTools=self.requireInitialTools,
            confirmationPrompt=confPrompt
        )

        context.set(f"{self.stageId}Raw", rawResult)
        context.set(f"{self.stageId}UISummary", uiSummary)
        return None


# FSM which manages the flow of stages in the boardroom simulation... this allows for dynamic branching based on the context and results of each stage
class BoardroomFSM:
    def __init__(self, initialStageId: str, title: str):
        self.initialStageId: str = initialStageId
        self.title: str = title
        self.stages: Dict[str, BoardroomStage] = {}
        self.transitions: Dict[str, Union[str, Callable[[BoardroomContext], Optional[str]]]] = {}

    def addStage(self, stage: BoardroomStage, nextStage: Optional[Union[str, Callable[[BoardroomContext], Optional[str]]]] = None) -> "BoardroomFSM":
        # Add a stage to the FSM, optionally define the next stage or a function to determine the next stage based on the context
        self.stages[stage.stageId] = stage
        if nextStage is not None:
            self.transitions[stage.stageId] = nextStage
        return self

    def setTransition(self, fromStageId: str, toStage: Union[str, Callable[[BoardroomContext], Optional[str]]]) -> "BoardroomFSM":
        # Define a transition from one stage to another, or a function to determine the next stage based on the context
        self.transitions[fromStageId] = toStage
        return self

    def getNextStageId(self, currentStageId: str, context: BoardroomContext) -> Optional[str]:
        # Get the next stageId based on the current stage and context, either from a predefined transition or by calling a function
        transition = self.transitions.get(currentStageId)
        if callable(transition):
            return transition(context)
        return transition

    def run(self, context: BoardroomContext) -> BoardroomContext:
        # Run the FSM starting from the initial stage, executing each stage in sequence or branching based on the context, until there are no more stages to execute
        currentStageId: Optional[str] = self.initialStageId
        pace = context.pace

        startTime = datetime.now()
        print(f"\n{'='*70}\nStarting FSM for '{self.title}'\n{'='*70}")

        while currentStageId:
            if isStopRequested():
                raise SimulationStoppedException("Simulation stopped by user.")

            stage = self.stages.get(currentStageId)
            if not stage:
                raise ValueError(f"FSM Error: Stage '{currentStageId}' is not registered in FSM.")

            emitPhaseFn = getattr(context.engine, "emitNewPhase", None) or getattr(context.engine, "_newPhaseHeader", None)
            if emitPhaseFn:
                emitPhaseFn(
                    phaseNumber=stage.phaseNumber,
                    phaseName=stage.phaseName,
                    pace=pace,
                    customAgents=stage.customAgents
                )

            explicitNext = stage.execute(context)

            if explicitNext is not None:
                currentStageId = explicitNext
            else:
                currentStageId = self.getNextStageId(currentStageId, context)

        endTime = datetime.now()
        timeTaken = endTime - startTime
        timeStr = f"{timeTaken.seconds // 60} mins {timeTaken.seconds % 60} secs"
        print(f"\n{'='*70}\nFSM for '{self.title}' completed in {timeStr}\n{'='*70}")

        return context
