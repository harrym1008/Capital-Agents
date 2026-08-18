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
from flask import Flask, render_template, redirect, jsonify
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

logging.getLogger("werkzeug").addFilter(MetricsFilter())

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


# Track connected websockets and the asyncio event loop
connectedWebsockets = set()
connectedWebsocketsLock = threading.Lock()
eventLoop = None

activeBoardroomThread = None
isBoardroomRunning = False
boardroomStateLock = threading.Lock()

def isBoardroomActive() -> bool:
    global activeBoardroomThread, isBoardroomRunning
    with boardroomStateLock:
        return bool(isBoardroomRunning and activeBoardroomThread is not None and activeBoardroomThread.is_alive())

def stopBoardroomSync(timeout: float = 5.0) -> bool:
    global activeBoardroomThread
    requestStop()
    with boardroomStateLock:
        thread = activeBoardroomThread
    if thread and thread.is_alive():
        thread.join(timeout=timeout)
    return not isBoardroomActive()

def broadcastEvent(eventData):
    global connectedWebsockets, eventLoop
    if not eventLoop:
        return
    with connectedWebsocketsLock:
        sockets = list(connectedWebsockets)
    if sockets:
        message = json.dumps(eventData)
        for ws in sockets:
            try:
                asyncio.run_coroutine_threadsafe(ws.send(message), eventLoop)
            except Exception:
                pass


@app.route("/api/boardroom/status")
def apiBoardroomStatus():
    return jsonify({"isRunning": isBoardroomActive()})


@app.route("/api/boardroom/stop", methods=["POST"])
def apiBoardroomStop():
    stopped = stopBoardroomSync(timeout=5.0)
    return jsonify({"ok": True, "stopped": stopped, "isRunning": isBoardroomActive()})


def runBoardroom(config):
    """Run boardroom simulation using active clients from serverManager."""
    global isBoardroomRunning, activeBoardroomThread
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
        if isStopRequested():
            print("Boardroom evaluation stopped by user.")
            emitEvent("simStopped", {"message": "Simulation stopped by user."})
        else:
            import traceback
            traceback.print_exc()
            emitEvent("error", {"message": f"{e.__class__.__name__}: {str(e)}"})
    finally:
        with boardroomStateLock:
            isBoardroomRunning = False
            activeBoardroomThread = None
        resetStop()


def handleBoardroomStart(data, websocket, eventLoop):
    global activeBoardroomThread, isBoardroomRunning
    resetStop()
    from boardroom.boardroom_config import SingleEquityRatingConfig
    config = SingleEquityRatingConfig.fromDict(data)

    with boardroomStateLock:
        isBoardroomRunning = True
        simThread = threading.Thread(
            target=runBoardroom,
            args=(config,),
            daemon=True
        )
        activeBoardroomThread = simThread
        simThread.start()
    return {"ok": True, "message": "Boardroom simulation started."}

def handleBoardroomStop(data, websocket, eventLoop):
    stopped = stopBoardroomSync(timeout=5.0)
    return {"ok": True, "stopped": stopped, "isRunning": isBoardroomActive(), "message": "Stop requested."}

def handleBoardroomStatus(data, websocket, eventLoop):
    return {"ok": True, "action": "status", "isRunning": isBoardroomActive()}

registerWsAction("start", handleBoardroomStart)
registerWsAction("stop", handleBoardroomStop)
registerWsAction("status", handleBoardroomStatus)


async def websocketHandler(websocket):
    global connectedWebsockets, eventLoop
    eventLoop = asyncio.get_running_loop()
    with connectedWebsocketsLock:
        connectedWebsockets.add(websocket)
    
    print(f"Client connected to Boardroom WebSocket (total: {len(connectedWebsockets)})")
    try:
        async for message in websocket:
            await handleWsMessage(websocket, message, eventLoop)
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        with connectedWebsocketsLock:
            connectedWebsockets.discard(websocket)
        print(f"Client disconnected from Boardroom WebSocket (remaining: {len(connectedWebsockets)})")

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