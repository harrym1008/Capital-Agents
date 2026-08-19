import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

import json
import time

from boardroom.boardroom_config import TIME_HORIZON_INFO, SingleEquityRatingConfig, BoardroomPace, TimeHorizon
from boardroom.boardroom_mgr import executeBoardroomConfig

from llm.server_manager import serverManager
from llm.llamacpp.llamacpp_args import LlamaCppModel


MODELS = {
    # "GEMMA_4_E2B": LlamaCppModel.GEMMA_4_E2B,
    # "GEMMA_4_E4B": LlamaCppModel.GEMMA_4_E4B,
    # "GEMMA_4_12B": LlamaCppModel.GEMMA_4_12B,
    "GEMMA_4_26B_A4B": LlamaCppModel.GEMMA_4_26B_A4B
}

SIM_TIME_STR = "2026-06-24"
TICKERS = ["MSFT", "KO", "MCD", "NFLX", "LLY"]

TIME_HORIZON = TimeHorizon.LONG
SUBMIT_TOOL = TIME_HORIZON_INFO[TIME_HORIZON]["llmSubmitToolName"]

MODES = {
    "oneshot": BoardroomPace.ONE_SHOT,
    "fast": BoardroomPace.FAST,
    "complete": BoardroomPace.COMPLETE
}

OUTPUT_DIR = os.path.join(ROOT, "exec", "tests", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "output_variance.json")


def runSingleEvaluation(model: str, ticker: str, mode: str):
    try:
        config = SingleEquityRatingConfig(
            generateSummaries=False,
            maxIterations=4,
            temperature=0.5,
            ticker=ticker,
            simulatedDateStr=SIM_TIME_STR,
            timeHorizon=TIME_HORIZON,
            boardroomPace=MODES[mode]
        )
        boardroomClient, summaryClient = serverManager.getClients()

        elapsedSeconds = executeBoardroomConfig(
            config=config,
            boardroomClient=boardroomClient,
            summaryClient=summaryClient,
            toolRegistry=serverManager.getToolRegistry()
        )
        decisionToolOutput = serverManager.getToolRegistry().getTool(SUBMIT_TOOL).toolLog[-2]

        return {
            "model": model,
            "ticker": ticker,
            "mode": mode,
            "seconds": elapsedSeconds,
            "output": decisionToolOutput
        }
    except Exception as e:
        print(f"Error during evaluation for model={model}, ticker={ticker}, mode={mode}: {e.__class__.__name__}: {str(e)}")
        return {
            "model": model,
            "ticker": ticker,
            "mode": mode,
            "error": f"{e.__class__.__name__}: {str(e)}"
        }


def exportResults(results: list):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    

def runAll():
    results = []
    for modelName in MODELS.keys():
        modelEnum = MODELS[modelName]

        success, message = serverManager.startServer(
            provider="llamacpp",
            modelName=modelEnum.name,
            allowParallel=True,
            wantSummaryServer=False
        )
        if not success:
            print(f"Failed to start server for model {modelName}: {message}")
            continue

        for ticker in TICKERS:
            for modeName in MODES.keys():
                print(f"Running evaluation for model={modelName}, ticker={ticker}, mode={modeName}")
                
                result = runSingleEvaluation(modelName, ticker, modeName)
                results.append(result)
                print(f"Completed evaluation for model={modelName}, ticker={ticker}, mode={modeName}: {result}")
                exportResults(results)          # Update the output file after each evaluation to ensure progress is saved

        serverManager.stopServer()
        time.sleep(2)

    exportResults(results)    # ensure export at the end of all evaluations 
    return results


def main():
    results = runAll()
    return results


if __name__ == "__main__":
    main()