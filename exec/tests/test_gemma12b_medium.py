import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

import json
import time
from dotenv import load_dotenv

load_dotenv()

from boardroom.boardroom_config import TIME_HORIZON_INFO, SingleEquityRatingConfig, BoardroomPace, SingleEquityTimeHorizon
from boardroom.boardroom_mgr import executeBoardroomConfig

from llm.server_manager import serverManager


MODEL_IDENTIFIER = "Gemma 4 12B Q4_K_XL"
MODEL_NAME = "GEMMA_4_12B"
MODEL = MODEL_IDENTIFIER
SIM_TIME_STR = "2025-01-31"

TICKERS = ["MSFT", "KO", "MCD", "NFLX", "LLY", "NVDA", "MU", "JPM", "TSLA", "IBM"]

TIME_HORIZON = SingleEquityTimeHorizon.MEDIUM       # Want 3mo and 12mo targets
SUBMIT_TOOL = TIME_HORIZON_INFO[TIME_HORIZON]["llmSubmitToolName"]

RUNS_PER_TICKER = 3
MODE = BoardroomPace.COMPLETE

OUTPUT_DIR = os.path.join(ROOT, "exec", "tests", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gemma12b_medium_variance.json")


def runSingleEvaluation(model: str, ticker: str, runIndex: int, toolRegistry=None):
    try:
        config = SingleEquityRatingConfig(
            generateSummaries=False,
            maxIterations=4,
            temperature=0.5,
            ticker=ticker,
            simulatedDateStr=SIM_TIME_STR,
            timeHorizon=TIME_HORIZON,
            boardroomPace=MODE
        )
        activeRegistry = toolRegistry if toolRegistry is not None else serverManager.getToolRegistry()
        llmClient = serverManager.getClient()

        elapsedSeconds = executeBoardroomConfig(
            config=config,
            llmClient=llmClient,
            toolRegistry=activeRegistry
        )
        submitTool = activeRegistry.getTool(SUBMIT_TOOL)
        if submitTool and len(submitTool.toolLog) >= 2:
            decisionToolOutput = submitTool.toolLog[-2]
        elif submitTool and len(submitTool.toolLog) == 1:
            decisionToolOutput = submitTool.toolLog[-1]
        else:
            decisionToolOutput = None

        return {
            "model": model,
            "ticker": ticker,
            "run": runIndex,
            "simulatedDate": SIM_TIME_STR,
            "timeHorizon": TIME_HORIZON.value,
            "mode": MODE.value,
            "seconds": elapsedSeconds,
            "output": decisionToolOutput
        }
    except Exception as e:
        print(f"Error during evaluation for model={model}, ticker={ticker}, run={runIndex}: {e.__class__.__name__}: {str(e)}")
        return {
            "model": model,
            "ticker": ticker,
            "run": runIndex,
            "simulatedDate": SIM_TIME_STR,
            "timeHorizon": TIME_HORIZON.value,
            "mode": MODE.value,
            "error": f"{e.__class__.__name__}: {str(e)}"
        }


def exportResults(results: list):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)


def runAll():
    results = []
    sharedToolRegistry = serverManager.getToolRegistry()

    success, message = serverManager.startServer(
        provider="llamacpp",
        modelName=MODEL,
        allowParallel=True
    )
    if not success:
        print(f"Failed to start server for model {MODEL_NAME}: {message}")
    else:
        for ticker in TICKERS:
            for runIndex in range(1, RUNS_PER_TICKER + 1):
                print(f"Running evaluation for model={MODEL_NAME}, ticker={ticker}, run={runIndex}/{RUNS_PER_TICKER}")

                result = runSingleEvaluation(MODEL_NAME, ticker, runIndex, sharedToolRegistry)
                results.append(result)
                print(f"Completed evaluation for model={MODEL_NAME}, ticker={ticker}, run={runIndex}: {result}")
                exportResults(results)          # Save progress after each evaluation

    serverManager.stopServer()
    time.sleep(2)

    exportResults(results)    # ensure export at the end of all evaluations
    return results


def main():
    results = runAll()
    return results


if __name__ == "__main__":
    main()
