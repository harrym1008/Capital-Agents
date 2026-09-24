import os
import threading
import time
import socket
import urllib.request
import collections
from enum import Enum
from typing import Tuple, Optional, List, Any

from llm.llamacpp.llamacpp_args import LLAMACPP_PORT
from llm.llamacpp.llamacpp_init import LlamaCppProcessInitiator, rudimentaryVramClear

from llm.llamacpp.llamacpp_client import LlamaCppClient
from llm.cloud.openrouter_client import OpenRouterClient
from llm.cloud.openai_compatible_client import OpenAICompatibleClient
from llm.server_config_store import setSectionValues

from llm.llm_client import BaseLLMClient
from llm.token_cost_tracker import TokenCostTracker
from llmtools.tool_registry import ToolRegistry

from finbert.finbert_engines import getSentimentEngine, unloadSentimentEngine
from ui.ui_hooks import emitEvent


def isPortReachable(host: str, port: int, timeout: float = 0.5) -> bool:
    # Check if TCP port accepts inbound connections
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except (ConnectionRefusedError, TimeoutError, OSError):
            return False


def parsePrometheusMetrics(text: str) -> dict:
    # Parse Prometheus metrics endpoint text into key-value dictionary (requires --metrics be enabled in params)
    metrics = {}
    if not text:
        return metrics
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        spaceIdx = line.rfind(' ')
        if spaceIdx == -1:
            continue
        key = line[:spaceIdx].strip()
        try:
            val = float(line[spaceIdx + 1:].strip())
        except ValueError:
            continue
        braceIdx = key.find('{')
        if braceIdx != -1:
            key = key[:braceIdx]
        metrics[key] = val
        metrics[key.replace(':', '_')] = val
    return metrics


def extractTokSpeeds(text: str) -> Tuple[float, float]:
    # Extract prompt prefill and generation token speeds from Prometheus metrics
    metrics = parsePrometheusMetrics(text)

    prefillSpeed = metrics.get('llamacpp:prompt_tokens_seconds') or metrics.get('llamacpp_prompt_tokens_seconds') or metrics.get('llamacpp:prompt_tokens_per_second') or metrics.get('llamacpp_prompt_tokens_per_second')
    genSpeed = metrics.get('llamacpp:predicted_tokens_seconds') or metrics.get('llamacpp_predicted_tokens_seconds') or metrics.get('llamacpp:tokens_predicted_per_second') or metrics.get('llamacpp_tokens_predicted_per_second') or metrics.get('llamacpp:tokens_predicted_seconds') or metrics.get('llamacpp:tokens_predicted_seconds')

    if prefillSpeed is None or genSpeed is None:
        predTokens = metrics.get('llamacpp:tokens_predicted_total') or metrics.get('llamacpp_tokens_predicted_total') or metrics.get('llamacpp:predicted_tokens_total') or metrics.get('llamacpp_predicted_tokens_total') or 0.0
        predSecs = metrics.get('llamacpp:tokens_predicted_seconds_total') or metrics.get('llamacpp_tokens_predicted_seconds_total') or metrics.get('llamacpp:predicted_seconds_total') or metrics.get('llamacpp_predicted_seconds_total') or metrics.get('llamacpp:generation_seconds_total') or 0.0

        promptTokens = metrics.get('llamacpp:prompt_tokens_total') or metrics.get('llamacpp_prompt_tokens_total') or metrics.get('llamacpp:tokens_evaluated_total') or metrics.get('llamacpp_tokens_evaluated_total') or 0.0
        promptSecs = metrics.get('llamacpp:prompt_seconds_total') or metrics.get('llamacpp_prompt_seconds_total') or metrics.get('llamacpp:prompt_processing_seconds_total') or metrics.get('llamacpp_prompt_processing_seconds_total') or metrics.get('llamacpp:prompt_evaluation_seconds_total') or 0.0

        genSpeed = (predTokens / predSecs) if (predTokens > 0 and predSecs > 0) else 0.0
        prefillSpeed = (promptTokens / promptSecs) if (promptTokens > 0 and promptSecs > 0) else 0.0

    return float(prefillSpeed or 0.0), float(genSpeed or 0.0)


def testLlmClient(client: BaseLLMClient, modelName: str) -> Tuple[bool, str]:
    # Execute non-streaming health verification query against model backend
    try:
        serverManager.recordLog("Sending test query...")
        response = client.openaiClient.chat.completions.create(
            model=modelName,
            messages=[{"role": "user", "content": "This is a test. Exit <think> immediately. Reply solely with the word 'OK'."}],
            max_completion_tokens=20,        # Limit response length to a few tokens
            temperature=0.0,
            stream=False
        )
        replyText = ""
        if response.choices and len(response.choices) > 0:
            replyText = response.choices[0].message.content or ""

        replyClean = replyText.strip()
        serverManager.recordLog(f"Test response received: '{replyClean}'")
        return True, replyClean
    
    except Exception as e:
        errorMsg = f"LLM test query failed: {e.__class__.__name__}: {str(e)}"
        emitEvent("error", {"message": errorMsg})
        return False, errorMsg


# 3 server backends supported: Llama.cpp, OpenRouter, OpenAI Compatible
class LoadedModelType(Enum):
    NONE = "none"
    OPENROUTER = "openrouter"
    LLAMACPP = "llamacpp"
    OPENAI_COMPATIBLE = "openaicompatible"


# Central coordinator managing local llama-server processes, cloud backends, logs, and token tracking
class ServerManager:
    def __init__(self):
        self.loadedModelType = LoadedModelType.NONE
        self.loadedModelName = ""

        self.boardroomProcess = None
        self.llmClient: Optional[BaseLLMClient] = None
        self.sharedToolRegistry: Optional[Any] = None
        self.costTracker = TokenCostTracker()

        self.startupLogs: List[str] = []
        self.serverLogs = collections.deque(maxlen=10000)
        self.metricsThread: Optional[threading.Thread] = None
        self.serverLock = threading.Lock()
        self.sentimentPreloadThread: Optional[threading.Thread] = None


    @property
    def boardroomClient(self) -> Optional[BaseLLMClient]:
        return self.llmClient

    @boardroomClient.setter
    def boardroomClient(self, client: Optional[BaseLLMClient]):
        self.llmClient = client

    @property
    def llamacppRunning(self) -> bool:
        return self.loadedModelType == LoadedModelType.LLAMACPP

    @property
    def openrouterRunning(self) -> bool:
        return self.loadedModelType == LoadedModelType.OPENROUTER

    @property
    def openaiCompatibleRunning(self) -> bool:
        return self.loadedModelType == LoadedModelType.OPENAI_COMPATIBLE


    def startSentimentEnginePreload(self) -> threading.Thread:
        # Pre-warm FinBERT tokenizer and inference engine in background thread
        self.recordLog("Pre-loading AutoTokenizer and FinBERT sentiment engine...")
        thread = threading.Thread(target=getSentimentEngine, daemon=True, name="SentimentModelPreloader")
        self.sentimentPreloadThread = thread
        thread.start()
        return thread

    def ensureSentimentEngineReady(self, timeout: float = 40.0) -> bool:
        # Block until FinBERT engine preloading finishes
        if self.sentimentPreloadThread and self.sentimentPreloadThread.is_alive():
            self.recordLog("Waiting for FinBERT sentiment engine to complete initialisation...")
            self.sentimentPreloadThread.join(timeout=timeout)

        engine = getSentimentEngine()
        if engine is not None:
            self.recordLog(f"Sentiment engine ready and verified: {type(engine).__name__} (Batch Size: {engine.optimalBatchSize})")
            return True
        else:
            self.recordLog("Warning: Sentiment engine could not be loaded.")
            return False


    def recordLog(self, logLine: str):
        # Append message to runtime logs and emit log event to UI
        self.startupLogs.append(logLine)
        self.serverLogs.append(logLine)
        emitEvent("llamaCppLog", {"log": logLine})

    def getLogs(self) -> List[str]:
        with self.serverLock:
            return list(self.serverLogs)


    def getToolRegistry(self) -> ToolRegistry:
        # Lazy initialise shared tool registry instance
        with self.serverLock:
            if self.sharedToolRegistry is None:
                from llmtools.registry_builder import buildToolRegistry
                self.sharedToolRegistry = buildToolRegistry()
            return self.sharedToolRegistry

    def getClient(self) -> Optional[BaseLLMClient]:
        # Return active LLM client instance
        with self.serverLock:
            return self.llmClient

    def getClients(self) -> Tuple[Optional[BaseLLMClient], Optional[BaseLLMClient]]:
        # Return pair of active LLM client instances
        with self.serverLock:
            return self.llmClient, self.llmClient


    def startMetricsPolling(self):
        # Spawn loop on a separate thread querying token throughput metrics from llama-server
        def metricsLoop():
            while self.loadedModelType == LoadedModelType.LLAMACPP:
                try:
                    metricsUrl = f"http://127.0.0.1:{LLAMACPP_PORT}/metrics"
                    req = urllib.request.Request(metricsUrl)
                    with urllib.request.urlopen(req, timeout=1.5) as resp:
                        content = resp.read().decode('utf-8')
                        prefillSpeed, genSpeed = extractTokSpeeds(content)
                        emitEvent("metricsUpdate", {
                            "prefillSpeed": round(prefillSpeed, 1),
                            "genSpeed": round(genSpeed, 1)
                        })
                except Exception:
                    pass
                time.sleep(2.0)

        self.metricsThread = threading.Thread(target=metricsLoop, daemon=True)
        self.metricsThread.start()

    def startServer(self, provider: str, modelName: str, baseUrl: Optional[str] = None, apiKey: Optional[str] = None, providerRouter: Optional[str] = None, allowParallel: bool = True, preclearVram: bool = False):
        # Start and verify designated LLM provider backend
        with self.serverLock:
            if self.loadedModelType != LoadedModelType.NONE:
                self.stopServerInternal()

            self.startupLogs.clear()
            self.serverLogs.clear()
            self.costTracker.reset()
            providerClean = provider.strip().lower().replace("_", "").replace("-", "")

            # Start the server based on providerClean
            match providerClean:
                case "openaicompatible":
                    try:
                        cleanBaseUrl = (baseUrl or "").strip()
                        if not cleanBaseUrl:
                            return False, "Base URL is required for OpenAI Compatible provider."

                        cleanApiKey = (apiKey or "").strip()
                        cleanModel = modelName.strip() if modelName else "default"
                        self.recordLog(f"Initialising OpenAI Compatible test for model '{cleanModel}' at '{cleanBaseUrl}'...")

                        client = OpenAICompatibleClient(baseUrl=cleanBaseUrl, apiKey=cleanApiKey, model=cleanModel)
                        success, result = testLlmClient(client, cleanModel)

                        if not success:
                            return False, f"OpenAI Compatible test failed: {result}"

                        self.llmClient = client
                        self.loadedModelType = LoadedModelType.OPENAI_COMPATIBLE
                        self.loadedModelName = cleanModel

                        # Remember OpenAI Compatible configuration fields
                        try:
                            setSectionValues("openaicompatible", {
                                "baseUrl": cleanBaseUrl,
                                "apiKey": cleanApiKey,
                                "modelName": cleanModel
                            })
                        except Exception:
                            pass

                        # Preload FinBERT sentiment engine concurrently
                        self.startSentimentEnginePreload()
                        self.broadcastStatus()
                        return True, f"OpenAI Compatible server active and verified (Response: '{result}')."

                    except Exception as e:
                        errorMsg = f"OpenAI Compatible setup failed: {e.__class__.__name__}: {str(e)}"
                        emitEvent("error", {"message": errorMsg})
                        self.broadcastStatus()
                        return False, errorMsg

                case "openrouter":
                    try:
                        apiKey = os.getenv("OPENROUTER_API_KEY", "")
                        routerMsg = f" ({providerRouter})" if providerRouter else ""
                        self.recordLog(f"Initialising OpenRouter test for model '{modelName}'{routerMsg}...")

                        client = OpenRouterClient(apiKey=apiKey, model=modelName, providerRouter=providerRouter)
                        success, result = testLlmClient(client, modelName)

                        if not success:
                            self.broadcastStatus()
                            return False, f"OpenRouter test failed: {result}"

                        self.llmClient = client
                        self.loadedModelType = LoadedModelType.OPENROUTER
                        self.loadedModelName = modelName

                        # Remember selected OpenRouter model and routing preferences
                        try:
                            routerValues = {"lastUsedModelId": modelName}
                            if providerRouter:
                                routerValues["providerRouter"] = providerRouter
                            setSectionValues("openrouter", routerValues)
                        except Exception:
                            pass

                        # Preload FinBERT sentiment engine concurrently
                        self.startSentimentEnginePreload()
                        self.broadcastStatus()
                        return True, f"OpenRouter server active and verified (Response: '{result}')."

                    except Exception as e:
                        errorMsg = f"OpenRouter setup failed: {e.__class__.__name__}: {str(e)}"
                        emitEvent("error", {"message": errorMsg})
                        self.broadcastStatus()
                        return False, errorMsg

                case "llamacpp":
                    try:
                        cleanModelName = (modelName or "").strip()
                        if not cleanModelName:
                            return False, "Model name or alias is required for Llama.cpp."

                        if preclearVram:
                            self.recordLog("Clearing VRAM...")
                            freedVram = rudimentaryVramClear()
                            time.sleep(1)
                            self.recordLog(f"VRAM cleared. Freed {freedVram:.2f} GB of VRAM.")

                        # Start loading FinBERT concurrently with llama.cpp startup
                        self.startSentimentEnginePreload()

                        # Initiate llama.cpp server process and wait for it to become ready
                        self.recordLog(f"Starting Llama.cpp boardroom server with model '{cleanModelName}'...")
                        serverProcess = LlamaCppProcessInitiator(
                            serverName="boardroom",
                            model=cleanModelName,
                            printLogsToTerminal=True,
                            killExistingProcesses=True,
                            allowParallel=allowParallel
                        )

                        serverErrorHolder = []

                        def runServerStart():
                            try:
                                self.recordLog(f"Command:\n{' '.join(serverProcess.getCommandLine())}\n\n")
                                serverProcess.start(readyTimeout=120)
                            except Exception as e:
                                serverErrorHolder.append(str(e))

                        startThread = threading.Thread(target=runServerStart, daemon=True)
                        startThread.start()
                        startThread.join(timeout=120)

                        if serverErrorHolder:
                            return False, f"Failed to start Llama.cpp server: {serverErrorHolder[0]}"

                        if not serverProcess.isReady() and not isPortReachable("127.0.0.1", LLAMACPP_PORT):
                            return False, f"Llama.cpp boardroom server failed to start on port {LLAMACPP_PORT}"

                        boardroomClient = LlamaCppClient(processInitiator=serverProcess, allowParallel=allowParallel)
                        success, result = testLlmClient(boardroomClient, modelName="model")

                        if not success:
                            try:
                                serverProcess.stop()
                            except Exception:
                                pass
                            return False, f"Llama.cpp verification test failed: {result}"

                        self.llmClient = boardroomClient
                        self.loadedModelType = LoadedModelType.LLAMACPP
                        self.loadedModelName = modelName
                        self.boardroomProcess = serverProcess

                        # Update last used model ID in configuration
                        try:
                            from llm.llamacpp.llamacpp_args import loadConfig, saveConfig
                            currCfg = loadConfig()
                            if getattr(serverProcess, "modelConfig", None) and serverProcess.modelConfig.get("id"):
                                currCfg["lastUsedModelId"] = serverProcess.modelConfig.get("id")
                                saveConfig(currCfg)
                        except Exception:
                            pass

                        # Start background metrics polling loop for llama.cpp
                        self.startMetricsPolling()

                        self.broadcastStatus()
                        return True, "Llama.cpp boardroom server started and verified."

                    except Exception as e:
                        errorMsg = f"Llama.cpp startup failed: {e.__class__.__name__}: {str(e)}"
                        emitEvent("error", {"message": errorMsg})
                        self.broadcastStatus()
                        return False, errorMsg

                case _:
                    # Should not reach here but handle gracefully otherwise
                    self.broadcastStatus()
                    return False, f"Unsupported provider: {provider}"

    def broadcastStatus(self):
        # Broadcast current server state to UI clients
        try:
            emitEvent("serverStatus", self.getStatus())
        except Exception as e:
            print(f"Error broadcasting server status: {e}")

    def stopServer(self):
        # Thread-safe server shutdown wrapper
        with self.serverLock:
            return self.stopServerInternal()

    def stopServerInternal(self):
        # Stop active subprocesses, release client handles, and unload models from VRAM
        match self.loadedModelType:
            case LoadedModelType.LLAMACPP:
                if self.boardroomProcess:
                    try:
                        self.boardroomProcess.stop()
                    except Exception:
                        pass
                self.boardroomProcess = None
            case LoadedModelType.OPENROUTER | LoadedModelType.OPENAI_COMPATIBLE:
                pass
            case _:
                pass

        if self.sentimentPreloadThread and self.sentimentPreloadThread.is_alive():
            self.sentimentPreloadThread.join(timeout=3.0)
        self.sentimentPreloadThread = None

        self.llmClient = None
        prevType = self.loadedModelType.value
        self.loadedModelType = LoadedModelType.NONE
        self.loadedModelName = ""
        self.costTracker.reset()

        # Unload sentiment engine
        try:
            self.recordLog("Unloading FinBERT sentiment engine and clearing VRAM...")
            unloadSentimentEngine()
        except Exception as e:
            print(f"Error unloading sentiment engine: {e}")

        self.broadcastStatus()
        return f"Server ({prevType}) stopped"

    def getStatus(self):
        # Return state dictionary describing active model and provider
        return {
            "running": self.loadedModelType != LoadedModelType.NONE,
            "provider": self.loadedModelType.value,
            "modelName": self.loadedModelName,
            "llamacppRunning": self.loadedModelType == LoadedModelType.LLAMACPP,
            "openrouterRunning": self.loadedModelType == LoadedModelType.OPENROUTER,
            "openaiCompatibleRunning": self.loadedModelType == LoadedModelType.OPENAI_COMPATIBLE,
            "openrouterModel": self.loadedModelName if self.loadedModelType == LoadedModelType.OPENROUTER else "",
            "startupLogs": self.startupLogs,
            "costData": self.costTracker.getPayload(),
        }


# Global singleton instance
serverManager = ServerManager()