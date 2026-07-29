import os
import sys
try:
    sys.stdout.reconfigure(errors='replace')
    sys.stderr.reconfigure(errors='replace')
except AttributeError:
    pass

import json
import asyncio
import threading
import time
from flask import Flask, render_template
import websockets

from dotenv import load_dotenv
load_dotenv()

UI_PORT = 9091
WS_PORT = 9092

# Add local path to sys.path so we can import local modules
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import urllib.request
import urllib.error

from exec.terminal_quick_run import runBoardroom, LLMClientType
from llm.llamacpp.llamacpp_args import LlamaCppModel, LLAMACPP_PORT
from ui.ui_hooks import setEventCallback, emitEvent
import logging

from ui.ws_api import registerApiRoutes

class MetricsFilter(logging.Filter):
    def filter(self, record):
        return "/api/metrics" not in record.getMessage()

logging.getLogger('werkzeug').addFilter(MetricsFilter())

app = Flask(__name__, template_folder=os.path.join(ROOT, "ui/templates"))
registerApiRoutes(app)

# Track active websocket connection and the event loop it runs on
activeWebsocket = None
eventLoop = None

def broadcastEvent(eventData):
    global activeWebsocket, eventLoop
    if activeWebsocket and eventLoop:
        message = json.dumps(eventData)
        # Thread-safe scheduling of the ws send coroutine on the asyncio event loop
        asyncio.run_coroutine_threadsafe(activeWebsocket.send(message), eventLoop)

@app.route("/")
def indexPage():
    return render_template("index.html")



def runSimulationThread(clientType, model, ticker, simulatedDate, fastMode, allowParallel, summaryHasOwnLocalServer):
    try:
        startTime = time.time()
        runBoardroom(
            llmClientType=clientType,
            model=model,
            tickerToEval=ticker,
            simulatedDate=simulatedDate,
            fastMode=fastMode,
            allowParallel=allowParallel,
            summaryHasOwnLocalServer=summaryHasOwnLocalServer
        )
        endTime = time.time()
        elapsed = endTime - startTime
        mins = int(elapsed // 60)
        secs = elapsed % 60
        totalTimeStr = f"{mins} mins {secs:.3f} secs"
        
        # Notify completion
        emitEvent("simComplete", {"totalTime": totalTimeStr})
    except Exception as e:
        import traceback
        traceback.print_exc()
        emitEvent("error", {"message": f"{e.__class__.__name__}: {str(e)}"})

async def websocketHandler(websocket):
    global activeWebsocket, eventLoop
    activeWebsocket = websocket
    eventLoop = asyncio.get_running_loop()
    
    print("Client connected to Boardroom WebSocket")
    try:
        async for message in websocket:
            data = json.loads(message)
            action = data.get("action")
            if action == "start":
                ticker = data.get("ticker", "NVDA")
                mode = data.get("mode", "fast")
                clientTypeStr = data.get("clientType", "OpenRouter")
                modelName = data.get("model", "")
                simulatedDate = data.get("simulatedDate") or None
                allowParallel = data.get("allowParallel", True)
                summaryHasOwnLocalServer = data.get("summaryHasOwnLocalServer", False)
                
                # Map client type
                if clientTypeStr == "LlamaCpp":
                    clientType = LLMClientType.LlamaCpp
                    # Look up enum member by its name from front-end select dropdown
                    try:
                        model = LlamaCppModel[modelName]
                    except KeyError:
                        model = LlamaCppModel.GEMMA_4_12B
                elif clientTypeStr == "OpenRouter":
                    clientType = LLMClientType.OpenRouter
                    model = modelName
                else:
                    clientType = LLMClientType.OpenRouter
                    model = modelName
                
                # Start simulation in background thread
                simThread = threading.Thread(
                    target=runSimulationThread,
                    args=(clientType, model, ticker, simulatedDate, mode == "fast", allowParallel, summaryHasOwnLocalServer),
                    daemon=True
                )
                simThread.start()
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected from Boardroom WebSocket")
    finally:
        if activeWebsocket == websocket:
            activeWebsocket = None

def startWebsocketServer():
    async def main():
        async with websockets.serve(websocketHandler, "127.0.0.1", WS_PORT) as server:
            # print(f"WebSocket Server running on ws://127.0.0.1:{WS_PORT}")
            await asyncio.Future()  # run forever

    asyncio.run(main())

if __name__ == "__main__":
    # Register global callback for agent simulation events
    setEventCallback(broadcastEvent)

    # Start WebSocket background server thread
    websocketThread = threading.Thread(target=startWebsocketServer, daemon=True)
    websocketThread.start()

    app.run(debug=False, threaded=True, host="127.0.0.1", port=UI_PORT)
