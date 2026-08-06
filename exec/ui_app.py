import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

try:
    sys.stdout.reconfigure(errors='replace')
    sys.stderr.reconfigure(errors='replace')
except AttributeError:
    pass

import json
import asyncio
import threading
from flask import Flask, render_template, redirect
import websockets

from dotenv import load_dotenv
load_dotenv()

UI_PORT = 9091
WS_PORT = 9092

from llm.server_manager import serverManager, LoadedModelType

from ui.ui_hooks import setEventCallback, emitEvent, requestStop, resetStop, isStopRequested, SimulationStoppedException
import logging

from ui.ws_api import registerApiRoutes, registerWsAction, handleWsMessage
from simulation.simulation_api import registerSimulationWsRoutes

class MetricsFilter(logging.Filter):
    def filter(self, record):
        return "/api/metrics" not in record.getMessage()

logging.getLogger('werkzeug').addFilter(MetricsFilter())

app = Flask(
    __name__, 
    template_folder=os.path.join(ROOT, "ui/templates"),
    static_folder=os.path.join(ROOT, "ui/static")
)
app.secret_key = "capital_agents_sim_secret_key"
registerApiRoutes(app)
registerSimulationWsRoutes()

@app.route("/")
def landingPage():
    return render_template("landing.html")

@app.route("/server-config")
def serverSetupPage():
    return render_template("server.html")

@app.route("/single-equity-rating")
def indexPage():
    if serverManager.loadedModelType == LoadedModelType.NONE:
        return redirect("/")

    serverManager.getToolRegistry()           # Ensure the tool registry is initialized
    return render_template("tickerrate.html")

@app.route("/marketsim")
def marketSimPage():
    return render_template("marketsim.html")


# Track active websocket connection and the event loop it runs on
activeWebsocket = None
eventLoop = None

def broadcastEvent(eventData):
    global activeWebsocket, eventLoop
    if activeWebsocket and eventLoop:
        message = json.dumps(eventData)
        # Thread-safe scheduling of the ws send coroutine on the asyncio event loop
        asyncio.run_coroutine_threadsafe(activeWebsocket.send(message), eventLoop)



def runBoardroom(config):
    """Run boardroom simulation using active clients from serverManager."""
    try:
        from boardroom.boardroom_runner import executeBoardroomConfig

        boardroomClient, summaryClient = serverManager.getClients()

        if not boardroomClient:
            emitEvent("error", {"message": "No active LLM server found. Please start a server from the manager setup page."})
            return

        executeBoardroomConfig(
            config=config,
            boardroomClient=boardroomClient,
            summaryClient=summaryClient
        )
    except SimulationStoppedException:
        print("Boardroom evaluation stopped by user.")
        emitEvent("simStopped", {"message": "Simulation stopped by user."})
    except Exception as e:
        import traceback
        traceback.print_exc()
        emitEvent("error", {"message": f"{e.__class__.__name__}: {str(e)}"})
    finally:
        resetStop()


def handleBoardroomStart(data, websocket, eventLoop):
    resetStop()
    from boardroom.boardroom_config import SingleEquityRatingConfig
    config = SingleEquityRatingConfig.fromDict(data)

    simThread = threading.Thread(
        target=runBoardroom,
        args=(config,),
        daemon=True
    )
    simThread.start()
    return {"ok": True, "message": "Boardroom simulation started."}

def handleBoardroomStop(data, websocket, eventLoop):
    requestStop()
    return {"ok": True, "message": "Stop requested."}

registerWsAction("start", handleBoardroomStart)
registerWsAction("stop", handleBoardroomStop)


async def websocketHandler(websocket):
    global activeWebsocket, eventLoop
    activeWebsocket = websocket
    eventLoop = asyncio.get_running_loop()
    
    print("Client connected to Boardroom WebSocket")
    try:
        async for message in websocket:
            await handleWsMessage(websocket, message, eventLoop)
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


def startServer():
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


if __name__ == "__main__":
    startServer()