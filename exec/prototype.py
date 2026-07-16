import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(errors='replace')
    sys.stderr.reconfigure(errors='replace')
except AttributeError:
    pass

import time
import math

from enum import Enum
from dotenv import load_dotenv
load_dotenv()

from llm.llm_client import BaseLLMClient

from llm.llamacpp.llamacpp_init import LlamaCppProcessInitiator, killExistingLlamaCppProcesses, rudimentaryVramClear
from llm.llamacpp.llamacpp_client import LlamaCppClient
from llm.llamacpp.llamacpp_args import LlamaCppModel

from llm.cloud.openrouter_client import OpenRouterClient
from llm.cloud.groq_client import GroqClient

from llm.boardroom_engine import boardroomGenerator


class LLMClient(Enum):
    LlamaCpp = 1
    OpenRouter = 2
    Groq = 3


def runBoardroom(llmClient: LLMClient, model: str | LlamaCppModel, tickerToEval: str, 
                 fastMode: bool = True, allowParallel: bool = True):
    llmClient: BaseLLMClient

    if llmClient == LLMClient.LlamaCpp:        
        killExistingLlamaCppProcesses()
        rudimentaryVramClear()
        argOverrides = []
        if not allowParallel:
            argOverrides += {"-np": "1", "--ctx-size": "65536"}
        serverProcess = LlamaCppProcessInitiator(model=model, killExistingProcesses=False, argOverrides=argOverrides)
        startThread = serverProcess.startOnAnotherThread()
        startThread.join()  

        llmClient = LlamaCppClient(serverProcess)
    elif llmClient == LLMClient.OpenRouter:
        apiKey = os.getenv("OPENROUTER_API_KEY")
        if not apiKey:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

        llmClient = OpenRouterClient(apiKey=apiKey, model=model)
    elif llmClient == LLMClient.Groq:
        apiKey = os.getenv("GROQ_API_KEY")
        if not apiKey:
            raise ValueError("GROQ_API_KEY environment variable is not set.")

        llmClient = GroqClient(apiKey=apiKey, model=model)

    simulatedDate = time.strftime("%Y-%m-%d", time.localtime())
    boardroom = boardroomGenerator(simulatedDate, llmClient)
    
    timeBefore = time.time()
    boardroom.executeSingleEquityRating(targetTicker=tickerToEval, fastMode=fastMode)
    timeAfter = time.time()

    seconds = timeAfter - timeBefore
    print(f"\nTotal time taken for boardroom evaluation: {math.floor(seconds/60)} mins {seconds%60:.1f} secs")

    if llmClient == LLMClient.LlamaCpp:
        time.sleep(2)
        serverProcess.stop()


if __name__ == "__main__":
    runBoardroom(llmClient=LLMClient.LlamaCpp, model=LlamaCppModel.GEMMA_4_12B, tickerToEval="NVDA", 
                 fastMode=False, allowParallel=True)
    


