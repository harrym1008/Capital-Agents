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


MODELS = {
    "GEMMA_4_26B_A4B": "Gemma 4 26B-A4B (no split-mode)",
    "GEMMA_4_12B": "Gemma 4 12B Q4_K_XL",
    "GEMMA_4_E4B": "Gemma 4 E4B",
    "GEMMA_4_E2B": "Gemma 4 E2B",
}

SIM_TIME_STR = "2026-06-24"
TICKERS = ["MSFT", "KO", "MCD", "NFLX", "LLY"]

TIME_HORIZON = SingleEquityTimeHorizon.LONG
SUBMIT_TOOL = TIME_HORIZON_INFO[TIME_HORIZON]["llmSubmitToolName"]

MODES = {
    "oneshot": BoardroomPace.ONE_SHOT,
    "fast": BoardroomPace.FAST,
    "complete": BoardroomPace.COMPLETE
}

OUTPUT_DIR = os.path.join(ROOT, "exec", "tests", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "output_variance.json")


def runSingleEvaluation(model: str, ticker: str, mode: str, toolRegistry=None):
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


def warmUpLruCache(sharedToolRegistry):
    warmupModel = "Gemma 4 E4B"
    print(f"\n{'=' * 80}")
    print(f"Starting initial warm-up run with {warmupModel} in fast mode to prime LRU cache...")
    print(f"{'=' * 80}\n")

    success, message = serverManager.startServer(
        provider="llamacpp",
        modelName=warmupModel,
        allowParallel=True
    )
    if not success:
        print(f"Failed to start server for warm-up model {warmupModel}: {message}")
        return

    for ticker in TICKERS:
        print(f"[WARM-UP] Running fast mode evaluation for ticker={ticker}...")
        runSingleEvaluation(warmupModel, ticker, "fast", sharedToolRegistry)
        cacheUsage = sharedToolRegistry.dataProviders.cache.getCacheUsagePrettyString()
        print(f"[WARM-UP] Finished {ticker}. LRU Cache usage: {cacheUsage}")

    serverManager.stopServer()
    time.sleep(2)
    print(f"\n{'=' * 80}")
    print(f"Initial LRU cache priming complete! Cache usage: {sharedToolRegistry.dataProviders.cache.getCacheUsagePrettyString()}")
    print(f"{'=' * 80}\n")


def runAll(warmUpFirst: bool = True):
    sharedToolRegistry = serverManager.getToolRegistry()

    if warmUpFirst:
        warmUpLruCache(sharedToolRegistry)

    results = []
    for modelName in MODELS.keys():
        modelIdentifier = MODELS[modelName]

        success, message = serverManager.startServer(
            provider="llamacpp",
            modelName=modelIdentifier,
            allowParallel=True
        )
        if not success:
            print(f"Failed to start server for model {modelName}: {message}")
            continue

        for ticker in TICKERS:
            for modeName in MODES.keys():
                print(f"Running evaluation for model={modelName}, ticker={ticker}, mode={modeName}")

                result = runSingleEvaluation(modelName, ticker, modeName, sharedToolRegistry)
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