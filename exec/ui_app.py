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
from flask import Flask, jsonify, render_template, redirect, url_for
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

from llm.server_manager import serverManager, LoadedModelType
from ui.ui_hooks import setEventCallback, emitEvent, requestStop, resetStop, isStopRequested, SimulationStoppedException
import logging

from ui.ws_api import registerApiRoutes

class MetricsFilter(logging.Filter):
    def filter(self, record):
        return "/api/metrics" not in record.getMessage()

logging.getLogger('werkzeug').addFilter(MetricsFilter())

app = Flask(
    __name__, 
    template_folder=os.path.join(ROOT, "ui/templates"),
    static_folder=os.path.join(ROOT, "ui/static")
)
registerApiRoutes(app)

@app.route("/")
def landingPage():
    if serverManager.loadedModelType != LoadedModelType.NONE:
        return redirect("/single-equity-rating")
    return redirect("/server")

@app.route("/server")
def serverSetupPage():
    return render_template("server.html")

@app.route("/single-equity-rating")
def indexPage():
    return render_template("tickerrate.html")

# Track active websocket connection and the event loop it runs on
activeWebsocket = None
eventLoop = None

def broadcastEvent(eventData):
    global activeWebsocket, eventLoop
    if activeWebsocket and eventLoop:
        message = json.dumps(eventData)
        # Thread-safe scheduling of the ws send coroutine on the asyncio event loop
        asyncio.run_coroutine_threadsafe(activeWebsocket.send(message), eventLoop)



def runBoardroom(ticker: str, simulatedDate: str = None, fastMode: bool = True, allowParallel: bool = True):
    """Run boardroom simulation using active clients from serverManager."""
    try:
        from exec.boardroom_runner import executeBoardroomRating

        boardroomClient, summaryClient = serverManager.getClients()

        if not boardroomClient:
            emitEvent("error", {"message": "No active LLM server found. Please start a server from the manager setup page."})
            return

        executeBoardroomRating(
            boardroomClient=boardroomClient,
            summaryClient=summaryClient,
            tickerToEval=ticker,
            simulatedDate=simulatedDate,
            fastMode=fastMode
        )
    except SimulationStoppedException:
        print("Boardroom evaluation stopped by user.")
        emitEvent("simStopped", {"message": "Simulation stopped by user."})
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
                resetStop()
                ticker = data.get("ticker", "NVDA")
                mode = data.get("mode", "fast")
                simulatedDate = data.get("simulatedDate") or None
                allowParallel = data.get("allowParallel", True)

                simThread = threading.Thread(
                    target=runBoardroom,
                    args=(ticker, simulatedDate, mode == "fast", allowParallel),
                    daemon=True
                )
                simThread.start()
            elif action == "stop":
                requestStop()
                emitEvent("simStopped", {"message": "Simulation stopped by user."})
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected from Boardroom WebSocket")
    finally:
        if activeWebsocket == websocket:
            activeWebsocket = None

def startWebsocketServer():
    async def main():
        async with websockets.serve(websocketHandler, "127.0.0.1", WS_PORT) as server:
            await asyncio.Future()  # run forever

    asyncio.run(main())

if __name__ == "__main__":
    # Register global callback for agent simulation events
    setEventCallback(broadcastEvent)

    # Start WebSocket background server thread on the main process, not the reloader
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        websocketThread = threading.Thread(
            target=startWebsocketServer,
            daemon=True
        )
        websocketThread.start()

    app.run(debug=True, threaded=True, host="127.0.0.1", port=UI_PORT)
