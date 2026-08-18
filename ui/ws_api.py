import json
import time
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
from flask import request, jsonify

from collectors.constants import NEW_YORK
from llm.llamacpp.llamacpp_args import LlamaCppModel, LLAMACPP_PORT
from llm.server_manager import serverManager
from simulation.simulation_api import simulationManager



openRouterModelsCache = None
openRouterCacheTime = 0


def registerApiRoutes(app):
    @app.route("/api/llamacpp-models")
    def getLlamaCppModels():
        models = [
            member.name
            for member in LlamaCppModel
            if member.name != "SUMMARY_MODEL"
        ]
        return jsonify({"models": models})

    @app.route("/api/openrouter-models")
    def getOpenRouterModels():
        global openRouterModelsCache, openRouterCacheTime
        currentTime = time.time()
        if openRouterModelsCache is not None and (currentTime - openRouterCacheTime) < 3600:
            return jsonify({"models": openRouterModelsCache})

        try:
            url = "https://openrouter.ai/api/v1/models"
            req = urllib.request.Request(url, headers={"User-Agent": "CapitalAgents"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                modelsData = data.get("data", [])
                openRouterModelsCache = modelsData
                openRouterCacheTime = currentTime
                return jsonify({"models": modelsData})
        except Exception as e:
            if openRouterModelsCache is not None:
                return jsonify({"models": openRouterModelsCache})
            return jsonify({"models": [], "error": str(e)}), 500

    @app.route("/api/openrouter-endpoints")
    def getOpenRouterEndpoints():
        modelId = request.args.get("model", "").strip()
        if not modelId:
            return jsonify({"providers": [], "endpoints": []})

        try:
            url = f"https://openrouter.ai/api/v1/models/{modelId}/endpoints"
            req = urllib.request.Request(url, headers={"User-Agent": "CapitalAgents"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                endpointsData = data.get("data", {}).get("endpoints", [])
                providers = []
                seen = set()
                for ep in endpointsData:
                    pName = ep.get("provider_name")
                    if pName and pName not in seen:
                        seen.add(pName)
                        providers.append(pName)
                return jsonify({"providers": providers, "endpoints": endpointsData})
        except Exception as e:
            return jsonify({"providers": [], "endpoints": [], "error": str(e)})


    @app.route("/api/server/start", methods=["POST"])
    @app.route("/api/llamacpp/start", methods=["POST"])
    @app.route("/api/openrouter/start", methods=["POST"])
    def apiStartServer():
        data = request.get_json(silent=True) or {}
        provider = data.get("provider")
        if not provider:
            provider = "openrouter" if request.path.endswith("/openrouter/start") else "llamacpp"

        modelName = data.get("model", "GEMMA_4_12B").strip()
        providerRouter = data.get("providerRouter") or data.get("router")
        allowParallel = data.get("allowParallel", True)
        wantSummaryServer = data.get("wantSummaryServer", False)
        preclearVram = data.get("preclearVram", False)

        success, message = serverManager.startServer(
            provider=provider,
            modelName=modelName,
            providerRouter=providerRouter,
            allowParallel=allowParallel,
            wantSummaryServer=wantSummaryServer,
            preclearVram=preclearVram
        )
        return jsonify({"ok": success, "message": message})


    @app.route("/api/server/stop", methods=["POST"])
    @app.route("/api/llamacpp/stop", methods=["POST"])
    @app.route("/api/openrouter/stop", methods=["POST"])
    @app.route("/api/llamacpp/summary/stop", methods=["POST"])
    def apiStopServer():
        message = serverManager.stopServer()
        return jsonify({"ok": True, "message": message})

    @app.route("/api/server/status")
    def apiServerStatus():
        return jsonify(serverManager.getStatus())

    @app.route("/api/ohlcv")
    def getOhlcvChart():
        ticker = request.args.get("ticker", "NVDA").strip()
        simDateStr = request.args.get("simDate")
        horizon = request.args.get("horizon", "long").strip().lower()

        if not simDateStr:
            simDateTs = pd.Timestamp.now(tz=NEW_YORK).normalize()
        else:
            try:
                simDateTs = pd.Timestamp(simDateStr, tz=NEW_YORK)
            except Exception:
                simDateTs = pd.Timestamp.now(tz=NEW_YORK).normalize()

        targetsRaw = request.args.get("targets", "")
        targetsList = []
        if targetsRaw:
            for item in targetsRaw.split(","):
                if ":" in item:
                    try:
                        monthsOffset, price = item.split(":")
                        targetsList.append((float(monthsOffset), float(price)))
                    except ValueError:
                        pass

        try:
            chartData = simulationManager.generateOhlcvChartData(ticker, simDateTs, targets=targetsList, horizon=horizon)
            return jsonify(chartData)
        except Exception as e:
            return jsonify({"error": str(e)}), 500


wsActionHandlers = {}

def registerWsAction(actionName, handlerFunc):
    wsActionHandlers[actionName] = handlerFunc

def sendWsResponse(websocket, payload, eventLoop=None):
    if websocket:
        try:
            msg = json.dumps(payload)
            if eventLoop:
                import asyncio
                asyncio.run_coroutine_threadsafe(websocket.send(msg), eventLoop)
        except Exception as e:
            print(f"Error sending WS response: {e}")

async def handleWsMessage(websocket, messageStr, eventLoop=None):
    try:
        data = json.loads(messageStr)
    except Exception as e:
        await websocket.send(json.dumps({"ok": False, "error": f"Invalid JSON payload: {str(e)}"}))
        return

    action = data.get("action")
    requestId = data.get("requestId")

    if not action:
        await websocket.send(json.dumps({"requestId": requestId, "ok": False, "error": "Missing 'action' in request."}))
        return

    handler = wsActionHandlers.get(action)
    if not handler:
        await websocket.send(json.dumps({"requestId": requestId, "action": action, "ok": False, "error": f"Unknown WebSocket action: '{action}'."}))
        return

    try:
        import inspect
        if inspect.iscoroutinefunction(handler):
            result = await handler(data, websocket, eventLoop)
        else:
            result = handler(data, websocket, eventLoop)

        if result is not None:
            if isinstance(result, dict):
                if "requestId" not in result and requestId is not None:
                    result["requestId"] = requestId
                if "action" not in result:
                    result["action"] = action
                await websocket.send(json.dumps(result))
    except Exception as e:
        import traceback
        traceback.print_exc()
        await websocket.send(json.dumps({"requestId": requestId, "action": action, "ok": False, "error": str(e)}))

