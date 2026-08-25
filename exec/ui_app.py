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
    return render_template("landing_page.html")

@app.route("/server-config")
def serverSetupPage():
    return render_template("server_config.html")

@app.route("/llamacpp-setup")
def llamacppSetupPage():
    return render_template("llamacpp_setup.html")

@app.route("/market-sim")
def marketSimPage():
    return render_template("market_sim.html")

@app.route("/single-equity-rating")
def singleEquityRatingPage():
    if serverManager.loadedModelType == LoadedModelType.NONE:
        return redirect("/")

    # serverManager.getToolRegistry()       
    return render_template("ticker_rate.html")


# Track connected websockets and the asyncio event loop
connectedWebsockets = set()
connectedWebsocketsLock = threading.Lock()
eventLoop = None

from boardroom.boardroom_config import SingleEquityRatingConfig
from boardroom.boardroom_mgr import boardroomManager

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
    return jsonify({"isRunning": boardroomManager.isBoardroomActive()})


@app.route("/api/boardroom/stop", methods=["POST"])
def apiBoardroomStop():
    stopped = boardroomManager.stopBoardroom(timeout=5.0)
    return jsonify({"ok": True, "stopped": stopped, "isRunning": boardroomManager.isBoardroomActive()})


def handleBoardroomStart(data, websocket, eventLoop):
    config = SingleEquityRatingConfig.fromDict(data)
    ok, msg = boardroomManager.startBoardroom(config)
    return {"ok": ok, "message": msg}

def handleBoardroomStop(data, websocket, eventLoop):
    stopped = boardroomManager.stopBoardroom(timeout=5.0)
    return {"ok": True, "stopped": stopped, "isRunning": boardroomManager.isBoardroomActive(), "message": "Stop requested."}

def handleBoardroomStatus(data, websocket, eventLoop):
    return {"ok": True, "action": "status", "isRunning": boardroomManager.isBoardroomActive()}

def handleQaQuery(data, websocket, eventLoop):
    query = data.get("query", "").strip()
    if not query:
        return {"ok": False, "error": "Query cannot be empty."}

    if not boardroomManager.activeBoardroom:
        return {"ok": False, "error": "No boardroom evaluation found. Please run a boardroom analysis first."}

    qaThread = threading.Thread(
        target=boardroomManager.processQnAQuery,
        args=(query,),
        daemon=True
    )
    qaThread.start()
    return {"ok": True, "message": "Q&A query processing started."}


def handleQaDeleteTurn(data, websocket, eventLoop):
    turnIndex = data.get("turnIndex")
    if turnIndex is not None:
        ok = boardroomManager.deleteQnATurn(int(turnIndex))
        return {"ok": ok, "turnIndex": turnIndex}
    return {"ok": False, "error": "turnIndex is required"}


registerWsAction("start", handleBoardroomStart)
registerWsAction("stop", handleBoardroomStop)
registerWsAction("status", handleBoardroomStatus)
registerWsAction("qa_query", handleQaQuery)
registerWsAction("qa_delete_turn", handleQaDeleteTurn)


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