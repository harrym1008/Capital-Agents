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
from llm.client_duo import ClientDuo

from llm.llamacpp.llamacpp_init import LlamaCppProcessInitiator, killExistingLlamaCppProcesses, rudimentaryVramClear
from llm.llamacpp.llamacpp_client import LlamaCppClient
from llm.llamacpp.llamacpp_args import LlamaCppModel

from llm.summarise.local_summary_server import LlamaCppSummaryClient
from llm.summarise.cloud_summary_server import OpenRouterSummaryClient

from llm.cloud.openrouter_client import OpenRouterClient
from llm.cloud.groq_client import GroqClient

from llm.boardroom_engine import boardroomGenerator


class LLMClientType(Enum):
    LlamaCpp = 1
    OpenRouter = 2
    Groq = 3


def runBoardroom(llmClientType: LLMClientType, model: str | LlamaCppModel, tickerToEval: str, 
                 fastMode: bool = True, allowParallel: bool = True):
    boardroomClient: BaseLLMClient
    summaryClient: BaseLLMClient

    if llmClientType == LLMClientType.LlamaCpp:        
        killExistingLlamaCppProcesses()
        rudimentaryVramClear()
        
        summaryClient = LlamaCppSummaryClient()

        argOverrides = []
        if not allowParallel:
            argOverrides += {"-np": "1", "--ctx-size": "65536"}

        serverProcess = LlamaCppProcessInitiator(
            serverName="boardroom", 
            model=model, 
            printLogsToTerminal=True,
            killExistingProcesses=False, 
            argOverrides=argOverrides
        )

        startThread = serverProcess.startOnAnotherThread()
        startThread.join()  

        boardroomClient = LlamaCppClient(serverProcess)

    elif llmClientType == LLMClientType.OpenRouter:
        apiKey = os.getenv("OPENROUTER_API_KEY")
        if not apiKey:  raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

        boardroomClient = OpenRouterClient(apiKey=apiKey, model=model)
        summaryClient = OpenRouterSummaryClient(apiKey=apiKey)

    elif llmClientType == LLMClientType.Groq:
        apiKey = os.getenv("GROQ_API_KEY")
        if not apiKey: raise ValueError("GROQ_API_KEY environment variable is not set.")

        boardroomClient = GroqClient(apiKey=apiKey, model=model)
        summaryClient = OpenRouterSummaryClient(apiKey=apiKey)

    clientDuo = ClientDuo(boardroomClient, summaryClient)

    simulatedDate = time.strftime("%Y-%m-%d", time.localtime())
    boardroom = boardroomGenerator(simulatedDate, clientDuo)
    
    timeBefore = time.time()
    boardroom.executeSingleEquityRating(targetTicker=tickerToEval, fastMode=fastMode)
    timeAfter = time.time()

    seconds = timeAfter - timeBefore
    print(f"\nTotal time taken for boardroom evaluation: {math.floor(seconds/60)} mins {seconds%60:.1f} secs")

    if llmClientType == LLMClientType.LlamaCpp:
        time.sleep(2)
        serverProcess.stop()
        summaryClient.stop()


if __name__ == "__main__":
    runBoardroom(llmClientType=LLMClientType.LlamaCpp, model=LlamaCppModel.GEMMA_4_12B, tickerToEval="NVDA", 
                 fastMode=False, allowParallel=True)
    


