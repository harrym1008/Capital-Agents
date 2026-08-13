import subprocess
import threading
import urllib.error
import urllib.request
import collections
from enum import Enum

from pathlib import Path
import shutil

import psutil
import time


from llm.llamacpp.llamacpp_args import LLAMACPP_EXECUTABLE, LLAMACPP_PORT, LLAMACPP_SUMMARY_PORT, \
    LlamaCppModel, EMPTY_ARG, LLAMACPP_MODEL_TO_ARGS, EXECUTABLE_ARG_OVERRIDE
from ui.ui_hooks import emitEvent


class ServerState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"


def killRemainingLlamaCppProcesses(executablePath=LLAMACPP_EXECUTABLE):
    for connection in psutil.net_connections(kind='inet'):      # Find llamacpp processes listening to the ports used by the server
        if connection.status == psutil.CONN_LISTEN and connection.laddr.port == LLAMACPP_PORT or \
            connection.laddr.port == LLAMACPP_SUMMARY_PORT and connection.pid is not None:
            try:
                process = psutil.Process(connection.pid)
                if process.name().lower() == executablePath.lower():
                    print(f"Killing remaining PID {process.pid}")
                    process.kill()
                    process.wait(timeout=5)
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass


def rudimentaryVramClear():
    try:
        import sys, gc, time, psutil
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
        
        skip = False
        freedVram = 0.0

        # Reset any existing sentiment engine singleton before clearing VRAM
        try:
            import llmtools.functions.sentiment as sentimentModule
            sentimentModule.sentimentEngine = None
            sentimentModule.engineLoadAttempted = False
        except Exception:
            pass

        gpuMem = []
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            memInfo = nvmlDeviceGetMemoryInfo(handle)

            totalVram = memInfo.total / (1024 ** 3)
            totalSysRam = psutil.virtual_memory().total / (1024 ** 3)
            targetAlloc = totalVram + min(totalSysRam * 0.25, 4)    # 25% of system RAM or 4GB overspill

            totalAlloc = 0
            usedVramBefore = round(memInfo.used / (1024 ** 3), 2)

            if usedVramBefore < 1.5:
                print(f"VRAM usage is already low: {usedVramBefore:.2f}/{totalVram:.2f} GB. Skipping VRAM clearing.")
                skip = True
                freedVram = 0.0
            
            else:
                print(f"Clearing VRAM: Before: {usedVramBefore:.2f}/{totalVram:.2f} GB...", end="\r", flush=True)
                import torch

                while True:
                    try:
                        emptyTensor = torch.empty((512, 512, 256), device="cuda")
                        gpuMem.append(emptyTensor)
                        totalAlloc += (512 * 512 * 256 * 4) / 1024**3
                        print(f"Clearing VRAM | Before: {usedVramBefore:.2f}/{totalVram:.2f} GB | Allocated {totalAlloc:.2f} GB", end="\r", flush=True)
                        if totalAlloc >= targetAlloc:
                            break
                        time.sleep(0.02)
                    except RuntimeError:
                        break

                time.sleep(3)

        finally:
            gpuMem.clear()
            del gpuMem

            if "torch" in sys.modules:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

            # for module in ["torch", "psutil"]:
            #     sys.modules.pop(module, None)

            gc.collect()
            time.sleep(1)

            if not skip:
                memInfo = nvmlDeviceGetMemoryInfo(handle)
                usedVramAfter = round(memInfo.used / (1024 ** 3), 2)
                freedVram = round(usedVramBefore - usedVramAfter, 2)
                print(f"\n  -> After: {usedVramAfter:.2f} GB | Freed: {freedVram:.2f} GB")

            # sys.modules.pop("pynvml", None)
            gc.collect()
            time.sleep(1)

    except Exception as e:
        print(f"Error during rudimentary VRAM clearing: {e}. Continuing without clearing VRAM.")
        freedVram = 0.0

    return freedVram


def checkExecutableExists(executablePath=LLAMACPP_EXECUTABLE):
    path = Path(executablePath)
    return path.is_file() or shutil.which(executablePath) is not None


class LlamaCppProcessInitiator:
    def __init__(self, 
                serverName: str, 
                model: LlamaCppModel, 
                printLogsToTerminal: bool = True,
                killExistingProcesses: bool = True, 
                argOverrides=None
        ):
        self.serverName = serverName
        
        self.args = LLAMACPP_MODEL_TO_ARGS.get(model, {}).copy()
        self.args.update(argOverrides or {})

        self.llamacppExecutable = self.args.get(EXECUTABLE_ARG_OVERRIDE, LLAMACPP_EXECUTABLE)

        if not checkExecutableExists(self.llamacppExecutable):
            raise FileNotFoundError(f"Llama.cpp executable not found at '{self.llamacppExecutable}' and not in PATH. Please ensure it is built and available.")
        
        if killExistingProcesses:
            if self.llamacppExecutable != LLAMACPP_EXECUTABLE:
                killRemainingLlamaCppProcesses(LLAMACPP_EXECUTABLE)
            killRemainingLlamaCppProcesses(self.llamacppExecutable)

        self.baseUrl = f"http://{self.args.get('--host', '127.0.0.1')}:{self.args.get('--port', LLAMACPP_PORT)}"
        self.healthUrl = f"{self.baseUrl}/health"
        self.apiUrl = f"{self.baseUrl}/v1"

        self.process = None
        self.state = ServerState.STOPPED

        self.stateLock = threading.Lock()
        self.stdoutThread = None
        self.stderrThread = None
        self.logs = collections.deque(maxlen=1000)
        self.printLogsToTerminal = printLogsToTerminal


    def setState(self, newState):
        with self.stateLock:
            self.state = newState


    def getState(self):
        with self.stateLock:
            return self.state


    def printToTerminal(self, *args, **kwargs):
        if self.printLogsToTerminal:
            print(*args, **kwargs)

    
    def readStream(self, stream, tag):
        try:
            for line in iter(stream.readline, ''):
                if self.getState() == ServerState.STARTING:
                    self.printToTerminal(line.rstrip())
                    try:
                        from llm.server_manager import serverManager
                        serverManager.recordLog(line.rstrip())
                    except Exception:
                        emitEvent("llamaCppStartupLog", {"log": line.rstrip()})
                self.logs.append(f"[{tag}] {line}")
        finally:
            stream.close()

    def isProcessAlive(self):
        return self.process is not None and self.process.poll() is None

    def isReady(self):
        try:
            with urllib.request.urlopen(self.healthUrl, timeout=1) as response:
                return response.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError):
            return False
        

    def startOnAnotherThread(self, readyTimeout=90):
        thread = threading.Thread(target=self.start, args=(readyTimeout,), daemon=True)
        thread.start()
        return thread

    def start(self, readyTimeout=90):
        with self.stateLock:
            if self.state in (ServerState.STARTING, ServerState.RUNNING):
                return
            
        command = [self.llamacppExecutable]
        for key, value in self.args.items():
            if key == EXECUTABLE_ARG_OVERRIDE:
                continue            
            command.append(key)
            if value != EMPTY_ARG:
                command.append(value)

        print(f"[{self.serverName}] Starting Llama.cpp process with command:\n'{' '.join(command)}'")
                
        self.setState(ServerState.STARTING)
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, bufsize=1)
        
        print(f"[{self.serverName}] Started Llama.cpp process PID={self.process.pid}")

        if self.process.stdout:
            self.stdoutThread = threading.Thread(target=self.readStream, args=(self.process.stdout, "STDOUT"), daemon=True)
            self.stdoutThread.start()
        if self.process.stderr:
            self.stderrThread = threading.Thread(target=self.readStream, args=(self.process.stderr, "STDERR"), daemon=True)
            self.stderrThread.start()

        waited = 0.0
        step = 0.25
        while waited < readyTimeout:
            if not self.isProcessAlive():
                self.setState(ServerState.STOPPED)
                exitCode = self.process.poll() if self.process else -1
                errorMsg = f"[{self.serverName}] Llama.cpp process terminated unexpectedly (exit code {exitCode})"
                print(errorMsg)
                emitEvent("error", {"message": errorMsg})
                raise RuntimeError(errorMsg)
            
            if self.isReady():
                self.setState(ServerState.RUNNING)
                print(f"[{self.serverName}] Llama.cpp server has loaded and is ready to take requests")
                return
            
            threading.Event().wait(step)
            waited += step

        self.stop()
        raise TimeoutError(f"[{self.serverName}] Llama.cpp process did not become ready within {readyTimeout} seconds")


    def stop(self, gracefulTimeout=10):
        with self.stateLock:
            if self.state in (ServerState.STOPPED, ServerState.STOPPING):
                return            
            self.state = ServerState.STOPPING

        if self.process is None:
            self.setState(ServerState.STOPPED)
            return
        
        if self.isProcessAlive():
            self.process.terminate()
            try:
                self.process.wait(timeout=gracefulTimeout)
                print(f"[{self.serverName}] Llama.cpp process has been stopped gracefully")
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
                print(f"[{self.serverName}]  Llama.cpp process was forcefully killed")

        self.process = None
        self.setState(ServerState.STOPPED)

        
