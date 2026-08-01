import os
import threading
import time
import socket
import urllib.request
from enum import Enum
from typing import Tuple, Optional, List, Any

from llm.llamacpp.llamacpp_args import LlamaCppModel, LLAMACPP_PORT, LLAMACPP_SUMMARY_PORT
from llm.llamacpp.llamacpp_init import LlamaCppProcessInitiator, rudimentaryVramClear
from llm.summarise.local_summary import LlamaCppSummaryClient

from llm.agents.agent import THINKING_BUDGET

from llm.llamacpp.llamacpp_client import LlamaCppClient
from llm.cloud.openrouter_client import OpenRouterClient

from llm.llm_client import BaseLLMClient
from llm.token_cost_tracker import TokenCostTracker
from ui.ui_hooks import emitEvent


def isPortReachable(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except (ConnectionRefusedError, TimeoutError, OSError):
            return False


def parsePrometheusMetrics(text: str) -> dict:
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
    metrics = parsePrometheusMetrics(text)

    prefillSpeed = metrics.get('llamacpp:prompt_tokens_seconds') or metrics.get('llamacpp_prompt_tokens_seconds') or metrics.get('llamacpp:prompt_tokens_per_second') or metrics.get('llamacpp_prompt_tokens_per_second')
    genSpeed = metrics.get('llamacpp:predicted_tokens_seconds') or metrics.get('llamacpp_predicted_tokens_seconds') or metrics.get('llamacpp:tokens_predicted_per_second') or metrics.get('llamacpp_tokens_predicted_per_second') or metrics.get('llamacpp:tokens_predicted_seconds') or metrics.get('llamacpp_tokens_predicted_seconds')

    if prefillSpeed is None or genSpeed is None:
        predTokens = metrics.get('llamacpp:tokens_predicted_total') or metrics.get('llamacpp_tokens_predicted_total') or metrics.get('llamacpp:predicted_tokens_total') or metrics.get('llamacpp_predicted_tokens_total') or 0.0
        predSecs = metrics.get('llamacpp:tokens_predicted_seconds_total') or metrics.get('llamacpp_tokens_predicted_seconds_total') or metrics.get('llamacpp:predicted_seconds_total') or metrics.get('llamacpp_predicted_seconds_total') or metrics.get('llamacpp_generation_seconds_total') or 0.0

        promptTokens = metrics.get('llamacpp:prompt_tokens_total') or metrics.get('llamacpp_prompt_tokens_total') or metrics.get('llamacpp:tokens_evaluated_total') or metrics.get('llamacpp_tokens_evaluated_total') or 0.0
        promptSecs = metrics.get('llamacpp:prompt_seconds_total') or metrics.get('llamacpp_prompt_seconds_total') or metrics.get('llamacpp:prompt_processing_seconds_total') or metrics.get('llamacpp_prompt_processing_seconds_total') or metrics.get('llamacpp_prompt_evaluation_seconds_total') or 0.0

        genSpeed = (predTokens / predSecs) if (predTokens > 0 and predSecs > 0) else 0.0
        prefillSpeed = (promptTokens / promptSecs) if (promptTokens > 0 and promptSecs > 0) else 0.0

    return float(prefillSpeed or 0.0), float(genSpeed or 0.0)


def testLlmClient(client: BaseLLMClient, modelName: str) -> Tuple[bool, str]:
    try:
        serverManager.recordLog("Sending test query...")
        response = client.openaiClient.chat.completions.create(
            model=modelName,
            messages=[{"role": "user", "content": "This is a test. Exit <think> immediately. Reply solely with the word 'OK'."}],
            max_tokens=5,
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


class LoadedModelType(Enum):
    NONE = "none"
    OPENROUTER = "openrouter"
    LLAMACPP = "llamacpp"


class ServerManager:
    """Manages LLM server lifecycles using unified startServer and stopServer methods."""

    def __init__(self):
        self.loadedModelType = LoadedModelType.NONE
        self.loadedModelName = ""
        self.summaryServerRunning = False
        self.boardroomProcess = None
        self.boardroomClient: Optional[BaseLLMClient] = None
        self.summaryClient: Optional[BaseLLMClient] = None
        self.startupLogs: List[str] = []
        self.metricsThread: Optional[threading.Thread] = None
        self.sharedToolRegistry: Optional[Any] = None
        self.costTracker = TokenCostTracker()
        self.serverLock = threading.Lock()

    @property
    def llamacppRunning(self) -> bool:
        return self.loadedModelType == LoadedModelType.LLAMACPP

    @property
    def openrouterRunning(self) -> bool:
        return self.loadedModelType == LoadedModelType.OPENROUTER

    def recordLog(self, logLine: str):
        """Append a log line to persistent startup logs list and emit WS event."""
        self.startupLogs.append(logLine)
        emitEvent("llamaCppStartupLog", {"log": logLine})

    def getToolRegistry(self):
        """Get or initialize the shared persistent ToolRegistry instance."""
        with self.serverLock:
            if self.sharedToolRegistry is None:
                from tools.registry_builder import buildToolRegistry
                self.sharedToolRegistry = buildToolRegistry()
            return self.sharedToolRegistry

    def getClients(self) -> Tuple[Optional[BaseLLMClient], Optional[BaseLLMClient]]:
        with self.serverLock:
            if not self.boardroomClient:
                return None, None
            if self.summaryClient is not None:
                return self.boardroomClient, self.summaryClient
            return self.boardroomClient, self.boardroomClient

    def _startMetricsPolling(self):
        """Poll llama-server Prometheus metrics every 2s while llamacpp is active, extract speeds, and broadcast over WS."""
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

    def startServer(self, provider: str, modelName: str, providerRouter: Optional[str] = None, allowParallel: bool = True, wantSummaryServer: bool = False):
        with self.serverLock:
            if self.loadedModelType != LoadedModelType.NONE:
                self.stopServerInternal()

            self.startupLogs.clear()
            self.costTracker.reset()
            providerClean = provider.strip().lower()

            match providerClean:
                case "openrouter":
                    try:
                        apiKey = os.getenv("OPENROUTER_API_KEY", "")
                        routerMsg = f" ({providerRouter})" if providerRouter else ""
                        self.recordLog(f"Initializing OpenRouter test for model '{modelName}'{routerMsg}...")

                        client = OpenRouterClient(apiKey=apiKey, model=modelName, providerRouter=providerRouter)
                        success, result = testLlmClient(client, modelName)

                        if not success:
                            return False, f"OpenRouter test failed: {result}"

                        self.boardroomClient = client
                        self.summaryClient = None
                        self.loadedModelType = LoadedModelType.OPENROUTER
                        self.loadedModelName = modelName
                        return True, f"OpenRouter server active and verified (Response: '{result}')."


                    except Exception as e:
                        errorMsg = f"OpenRouter setup failed: {e.__class__.__name__}: {str(e)}"
                        emitEvent("error", {"message": errorMsg})
                        return False, errorMsg

                case "llamacpp":
                    try:
                        try:
                            modelEnum = LlamaCppModel[modelName]
                        except KeyError:
                            return False, f"Unknown Llama.cpp model: {modelName}"

                        argOverrides = {}
                        if not allowParallel:
                            argOverrides["-np"] = "1"
                            argOverrides["--ctx-size"] = "65536"

                        self.recordLog(f"Clearing VRAM...")
                        rudimentaryVramClear()
                        self.recordLog("VRAM cleared, starting Llama.cpp boardroom server...")
                        serverProcess = LlamaCppProcessInitiator(
                            serverName="boardroom",
                            model=modelEnum,
                            printLogsToTerminal=False,
                            killExistingProcesses=True,
                            argOverrides=argOverrides,
                        )

                        serverErrorHolder = []

                        def runServerStart():
                            try:
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

                        if wantSummaryServer:
                            self.startSummServerInternal()
                        else:
                            self.summaryClient = None

                        self.boardroomClient = boardroomClient
                        self.loadedModelType = LoadedModelType.LLAMACPP
                        self.loadedModelName = modelName
                        self.boardroomProcess = serverProcess

                        # Start background metrics polling loop for Llama.cpp
                        self._startMetricsPolling()

                        return True, "Llama.cpp boardroom server started and verified."

                    except Exception as e:
                        errorMsg = f"Llama.cpp startup failed: {e.__class__.__name__}: {str(e)}"
                        emitEvent("error", {"message": errorMsg})
                        return False, errorMsg

                case _:
                    return False, f"Unsupported provider: {provider}"

    def stopServer(self):
        with self.serverLock:
            return self.stopServerInternal()

    def stopServerInternal(self):
        match self.loadedModelType:
            case LoadedModelType.LLAMACPP:
                if self.boardroomProcess:
                    try:
                        self.boardroomProcess.stop()
                    except Exception:
                        pass
                self.boardroomProcess = None
            case LoadedModelType.OPENROUTER:
                pass
            case _:
                pass

        if self.summaryClient:
            try:
                self.summaryClient.stop()
            except Exception:
                pass

        self.boardroomClient = None
        self.summaryClient = None
        self.summaryServerRunning = False
        prevType = self.loadedModelType.value
        self.loadedModelType = LoadedModelType.NONE
        self.loadedModelName = ""
        self.costTracker.reset()
        return f"Server ({prevType}) stopped"

    def startSummServerInternal(self):
        if self.summaryServerRunning:
            return
        try:
            client = LlamaCppSummaryClient(thinkingBudget=THINKING_BUDGET)
            ready = False
            for _ in range(120):
                if isPortReachable("127.0.0.1", LLAMACPP_SUMMARY_PORT):
                    ready = True
                    break
                time.sleep(0.5)
            if ready:
                self.summaryServerRunning = True
                self.summaryClient = client
        except Exception:
            pass

    def getStatus(self):
        """Return dict of current server running status and startup logs."""
        return {
            "running": self.loadedModelType != LoadedModelType.NONE,
            "provider": self.loadedModelType.value,
            "modelName": self.loadedModelName,
            "summaryServerRunning": self.summaryServerRunning,
            "llamacppRunning": self.loadedModelType == LoadedModelType.LLAMACPP,
            "openrouterRunning": self.loadedModelType == LoadedModelType.OPENROUTER,
            "openrouterModel": self.loadedModelName if self.loadedModelType == LoadedModelType.OPENROUTER else "",
            "startupLogs": self.startupLogs,
            "costData": self.costTracker.getPayload(),
        }


# Global singleton instance
serverManager = ServerManager()