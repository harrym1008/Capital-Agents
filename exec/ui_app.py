import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

try:
    sys.stdout.reconfigure(errors='replace')
    sys.stderr.reconfigure(errors='replace')
except AttributeError:
    pass

from colorama import init as coloramaInitialise
coloramaInitialise(autoreset=False)

import json
import logging
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
from ui.ws_api import registerApiRoutes, registerWsAction, handleWsMessage
from simulation.simulation_api import registerSimulationWsRoutes

from boardroom.boardroom_config import SingleEquityRatingConfig, PortfolioCreationConfig, PortfolioRebalancingConfig, AgentPortfolioSimulationConfig
from boardroom.boardroom_mgr import boardroomManager


# Suppress super noisy polling requests in Werkzeug server logs
class MetricsFilter(logging.Filter):
    def filter(self, record):
        return "/api/metrics" not in record.getMessage()

logging.getLogger("werkzeug").addFilter(MetricsFilter())

app = Flask(
    __name__, 
    template_folder=os.path.join(ROOT, "ui/templates"),
    static_folder=os.path.join(ROOT, "ui/static")
)
app.secret_key = "capital_agents_secret_key"
registerApiRoutes(app)
registerSimulationWsRoutes()

# HTML page routes
@app.route("/")
def landingPage():
    return render_template("landing_page.html")

@app.route("/server-config")
def serverSetupPage():
    return render_template("server_config.html")

@app.route("/llamacpp-setup")
def llamacppSetupPage():
    return render_template("llamacpp_setup.html")

@app.route("/llamacpp-logs")
def llamacppLogsPage():
    return render_template("llamacpp_logs.html")

@app.route("/user-portfolio-sim")
def userMarketSimPage():
    return render_template("market_sim.html")


def routeOnlyIfModelLoaded(location):
    # Require loaded LLM backend before accessing simulation views
    if serverManager.loadedModelType == LoadedModelType.NONE:
        return redirect("/")
    return render_template(location) 

@app.route("/agent-portfolio-sim")
def agentPortfolioSimPage():
    return routeOnlyIfModelLoaded("agent_portfolio_sim.html")

@app.route("/single-equity-rating")
def singleEquityRatingPage():
    return routeOnlyIfModelLoaded("single_equity_rating.html")

@app.route("/portfolio-creation")
def portfolioCreationPage():
    return routeOnlyIfModelLoaded("portfolio_creation.html")

@app.route("/portfolio-rebalancing")
def portfolioRebalancingPage():
    return routeOnlyIfModelLoaded("portfolio_rebalancing.html")



# Connected WebSocket clients and event loop state
connectedWebsockets = set()
connectedWebsocketsLock = threading.Lock()
eventLoop = None


def broadcastEvent(eventData):
    # Broadcast event payload across active WebSocket connections
    global connectedWebsockets, eventLoop
    if not eventLoop:
        return
    with connectedWebsocketsLock:
        sockets = list(connectedWebsockets)
    if sockets:
        try:
            message = json.dumps(eventData)
        except Exception as e:
            evtType = eventData.get("type") if isinstance(eventData, dict) else "unknown"
            print(f"Error serialising UI event '{evtType}': {e}")
            return

        for ws in sockets:
            try:
                asyncio.run_coroutine_threadsafe(ws.send(message), eventLoop)
            except Exception:
                pass


@app.route("/api/boardroom/status")
def apiBoardroomStatus():
    # Return active state of boardroom manager
    return jsonify({"isRunning": boardroomManager.isBoardroomActive()})


@app.route("/api/boardroom/stop", methods=["POST"])
def apiBoardroomStop():
    # Request stop for running boardroom session
    stopped = boardroomManager.stopBoardroom(timeout=5.0)
    return jsonify({"ok": True, "stopped": stopped, "isRunning": boardroomManager.isBoardroomActive()})


def handleBoardroomStart(data, websocket, eventLoop):
    # Initialise and dispatch boardroom workflow based on configuration type
    engineType = data.get("engineType")
    if engineType == "portfolio_rebalancing":
        config = PortfolioRebalancingConfig.fromDict(data)
    elif engineType == "portfolio_creation":
        config = PortfolioCreationConfig.fromDict(data)
    elif engineType == "agent_portfolio_simulation":
        config = AgentPortfolioSimulationConfig.fromDict(data)
    else:
        config = SingleEquityRatingConfig.fromDict(data)
    ok, msg = boardroomManager.startBoardroom(config)
    return {"ok": ok, "message": msg}


def handleBoardroomStop(data, websocket, eventLoop):
    # Signal boardroom manager to halt active agent execution
    stopped = boardroomManager.stopBoardroom(timeout=5.0)
    return {"ok": True, "stopped": stopped, "isRunning": boardroomManager.isBoardroomActive(), "message": "Stop requested."}

def handleBoardroomStatus(data, websocket, eventLoop):
    # Return current status flag for boardroom
    return {"ok": True, "action": "status", "isRunning": boardroomManager.isBoardroomActive()}

def handleQaQuery(data, websocket, eventLoop):
    # Dispatch Q&A query to boardroom agent in background worker thread
    query = data.get("query", "").strip()
    if not query:
        return {"ok": False, "error": "Query cannot be empty."}

    if not boardroomManager.activeBoardroom:
        return {"ok": False, "error": "No boardroom evaluation found. Please run a boardroom analysis first."}

    targetAgent = data.get("targetAgent", "auto")

    qaThread = threading.Thread(
        target=boardroomManager.processQnAQuery,
        args=(query, targetAgent),
        daemon=True
    )
    qaThread.start()
    return {"ok": True, "message": "Q&A query processing started."}


def handleQaDeleteTurn(data, websocket, eventLoop):
    # Delete specific turn from active Q&A transcript
    turnIndex = data.get("turnIndex")
    if turnIndex is not None:
        ok = boardroomManager.deleteQnATurn(int(turnIndex))
        return {"ok": ok, "turnIndex": turnIndex}
    return {"ok": False, "error": "turnIndex is required"}


def handleGetSources(data, websocket, eventLoop):
    # Fetch list of referenced citation sources from active boardroom session
    toolReg = boardroomManager.getToolRegistry()
    if hasattr(toolReg, "sourcesManager") and toolReg.sourcesManager:
        return {"ok": True, "sources": toolReg.sourcesManager.getSources()}
    return {"ok": True, "sources": []}


# Register WebSocket action handlers
registerWsAction("start", handleBoardroomStart)
registerWsAction("stop", handleBoardroomStop)
registerWsAction("status", handleBoardroomStatus)
registerWsAction("qa_query", handleQaQuery)
registerWsAction("qa_delete_turn", handleQaDeleteTurn)
registerWsAction("get_sources", handleGetSources)



async def websocketHandler(websocket):
    # Handle incoming WebSocket client connections and message routing
    global connectedWebsockets, eventLoop
    eventLoop = asyncio.get_running_loop()
    with connectedWebsocketsLock:
        connectedWebsockets.add(websocket)
    
    print(f"Client connected to Boardroom WebSocket (total: {len(connectedWebsockets)})")
    try:
        initialStatusMsg = json.dumps({
            "type": "serverStatus",
            **serverManager.getStatus()
        })
        await websocket.send(initialStatusMsg)
    except Exception as e:
        print(f"Error sending initial server status on WS connect: {e}")

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
    # Run async WebSocket server on dedicated port
    async def main():
        async with websockets.serve(websocketHandler, "127.0.0.1", WS_PORT) as server:
            await asyncio.Future()      # Run server loop forever
    asyncio.run(main())


def startServer(debug: bool = True, useReloader: bool = False):
    # Start background WebSocket listener and launch Flask application
    setEventCallback(broadcastEvent)

    if not useReloader or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        websocketThread = threading.Thread(
            target=startWebsocketServer,
            daemon=True
        )
        websocketThread.start()

    app.run(debug=debug, threaded=True, host="127.0.0.1", port=UI_PORT, use_reloader=useReloader)


if __name__ == "__main__":
    startServer()