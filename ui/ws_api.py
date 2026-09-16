import json
import time
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
from flask import request, jsonify

from collectors.constants import NEW_YORK
from llm.llamacpp.llamacpp_args import LLAMACPP_PORT, loadConfig, saveConfig, validateGGUFPath, getLlamaCppModelsList, openNativeGgufFileDialog, openNativeExecutableFileDialog
from llm.server_manager import serverManager



openRouterModelsCache = None
openRouterCacheTime = 0


def registerApiRoutes(app):
    @app.route("/api/llamacpp-models")
    def getLlamaCppModels():
        models = getLlamaCppModelsList()
        return jsonify({"models": models})

    @app.route("/api/llamacpp/config", methods=["GET", "POST"])
    def apiLlamaCppConfig():
        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            if not isinstance(data, dict):
                return jsonify({"ok": False, "error": "Invalid configuration data format"}), 400
            saved = saveConfig(data)
            return jsonify({"ok": saved, "config": loadConfig()})
        else:
            return jsonify({"ok": True, "config": loadConfig()})

    @app.route("/api/llamacpp/validate-gguf", methods=["POST"])
    def apiValidateGguf():
        data = request.get_json(silent=True) or {}
        filePath = data.get("filePath", "")
        valid, result = validateGGUFPath(filePath)
        if valid:
            return jsonify({"ok": True, "valid": True, "info": result})
        else:
            return jsonify({"ok": False, "valid": False, "error": result})

    @app.route("/api/llamacpp/browse-gguf", methods=["POST"])
    def apiBrowseGguf():
        selectedPath = openNativeGgufFileDialog()
        if not selectedPath:
            return jsonify({"ok": True, "cancelled": True})
        
        valid, result = validateGGUFPath(selectedPath)
        if not valid:
            return jsonify({"ok": False, "cancelled": False, "error": result})
        
        fileName = result.get("fileName", "")
        defaultAlias = fileName[:-5] if fileName.lower().endswith(".gguf") else fileName
        return jsonify({
            "ok": True,
            "cancelled": False,
            "filePath": selectedPath,
            "fileName": fileName,
            "defaultAlias": defaultAlias
        })

    @app.route("/api/llamacpp/browse-executable", methods=["POST"])
    def apiBrowseExecutable():
        selectedPath = openNativeExecutableFileDialog()
        if not selectedPath:
            return jsonify({"ok": True, "cancelled": True})
        return jsonify({
            "ok": True,
            "cancelled": False,
            "filePath": selectedPath
        })




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
    @app.route("/api/openai/start", methods=["POST"])
    def apiStartServer():
        data = request.get_json(silent=True) or {}
        provider = data.get("provider")
        if not provider:
            if request.path.endswith("/openrouter/start"):
                provider = "openrouter"
            elif request.path.endswith("/openai/start"):
                provider = "openaicompatible"
            else:
                provider = "llamacpp"

        modelName = data.get("model", "GEMMA_4_12B").strip()
        baseUrl = data.get("baseUrl") or data.get("base_url")
        apiKey = data.get("apiKey") or data.get("api_key")
        providerRouter = data.get("providerRouter") or data.get("router")
        allowParallel = data.get("allowParallel", True)
        preclearVram = data.get("preclearVram", False)

        success, message = serverManager.startServer(
            provider=provider,
            modelName=modelName,
            baseUrl=baseUrl,
            apiKey=apiKey,
            providerRouter=providerRouter,
            allowParallel=allowParallel,
            preclearVram=preclearVram
        )
        return jsonify({"ok": success, "message": message})


    @app.route("/api/server/stop", methods=["POST"])
    @app.route("/api/llamacpp/stop", methods=["POST"])
    @app.route("/api/openrouter/stop", methods=["POST"])
    @app.route("/api/openai/stop", methods=["POST"])
    def apiStopServer():
        message = serverManager.stopServer()
        return jsonify({"ok": True, "message": message})

    @app.route("/api/llamacpp/logs")
    @app.route("/api/server/logs")
    def apiLlamaCppLogs():
        currentLogs = serverManager.getLogs()
        serverRunning = serverManager.llamacppRunning
        modelName = serverManager.loadedModelName
        return jsonify({
            "ok": True,
            "running": serverRunning,
            "modelName": modelName,
            "logs": currentLogs
        })

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
            from simulation.simulation_api import simulationManager
            chartData = simulationManager.generateOhlcvChartData(ticker, simDateTs, targets=targetsList, horizon=horizon)
            return jsonify(chartData)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/portfolio/backtest", methods=["POST"])
    def getPortfolioBacktest():
        data = request.get_json(silent=True) or {}
        simDate = data.get("simDate") or data.get("simulatedDate")
        if not simDate:
            return jsonify({"ok": False, "error": "Missing 'simDate' in request."}), 400

        initialCapital = data.get("initialCapital", 100_000.0)
        positions = data.get("positions", [])
        cashPosition = data.get("cashPosition", {})

        try:
            from simulation.simulation_api import simulationManager
            backtestData = simulationManager.generatePortfolioBacktestData(
                simDate=simDate,
                initialCapital=initialCapital,
                positions=positions,
                cashPosition=cashPosition
            )
            return jsonify(backtestData)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/boardroom/sources")
    def apiBoardroomSources():
        try:
            from boardroom.boardroom_mgr import boardroomManager
            toolReg = boardroomManager.getToolRegistry()
            if hasattr(toolReg, "sourcesManager") and toolReg.sourcesManager:
                return jsonify({"ok": True, "sources": toolReg.sourcesManager.getSources()})
            return jsonify({"ok": True, "sources": []})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e), "sources": []}), 500



wsActionHandlers = {}

def registerWsAction(actionName, handlerFunc):
    wsActionHandlers[actionName] = handlerFunc

def handleGetServerStatus(data, websocket, eventLoop):
    return {
        "type": "serverStatus",
        **serverManager.getStatus()
    }

registerWsAction("get_server_status", handleGetServerStatus)

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

