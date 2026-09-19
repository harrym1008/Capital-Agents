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


from typing import Optional, Any
from llm.llamacpp.llamacpp_args import LLAMACPP_EXECUTABLE, LLAMACPP_PORT, buildLlamaCppCommandLine
from ui.ui_hooks import emitEvent


class ServerState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"


def killRemainingLlamaCppProcesses(executablePath=LLAMACPP_EXECUTABLE):
    for connection in psutil.net_connections(kind='inet'):      # Find llamacpp processes listening to the ports used by the server
        if connection.status == psutil.CONN_LISTEN and connection.laddr.port == LLAMACPP_PORT and connection.pid is not None:
            try:
                process = psutil.Process(connection.pid)
                if process.name().lower() == executablePath.lower():
                    print(f"Killing remaining PID {process.pid}")
                    process.kill()
                    process.wait(timeout=5)
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass


def rudimentaryVramClear() -> float:
    freedVram = 0.0
    nvmlInitialised = False

    # Force unload sentiment engine and clear its references first
    try:
        from finbert.finbert_engines import unloadSentimentEngine
        unloadSentimentEngine()
    except Exception as e:
        print(f"[VRAM Clear] Note: Sentiment unload failed or not loaded: {e}")

    try:
        import sys, gc, time, psutil
        import torch
        from pynvml import (
            nvmlInit,
            nvmlShutdown,
            nvmlDeviceGetCount,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetName,
            nvmlDeviceGetMemoryInfo,
        )

        if not torch.cuda.is_available():
            print("[VRAM Clear] CUDA is not available. Skipping VRAM clearance.")
            gc.collect()
            return 0.0

        nvmlInit()
        nvmlInitialised = True
        deviceCount = nvmlDeviceGetCount()

        if deviceCount == 0:
            print("[VRAM Clear] No NVIDIA GPU devices detected.")
            return 0.0

        totalSysRam = psutil.virtual_memory().total / (1024 ** 3)
        gpuHandles = []
        gpuNames = []
        perGpuUsedBefore = []
        perGpuTotal = []
        perGpuFill = []

        for i in range(deviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)
            gpuHandles.append(handle)
            name = nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            gpuNames.append(name)

            memInfo = nvmlDeviceGetMemoryInfo(handle)
            usedGb = memInfo.used / (1024 ** 3)
            totalGb = memInfo.total / (1024 ** 3)
            perGpuUsedBefore.append(usedGb)
            perGpuTotal.append(totalGb)
            perGpuFill.append(0.0)

        totalVram = sum(perGpuTotal)
        totalUsedBefore = sum(perGpuUsedBefore)
        targetAlloc = totalVram + min(totalSysRam * 0.08, 4.0)

        print(f"\n[VRAM Clear] Detected {deviceCount} NVIDIA GPU(s) | Total VRAM: {totalVram:.2f} GB | Host RAM: {totalSysRam:.2f} GB")
        print(f"[VRAM Clear] Active VRAM before clear: {totalUsedBefore:.2f} GB | Target allocation: {targetAlloc:.2f} GB")

        allocatedTensors = []
        chunkElements = (512, 512, 256)
        chunkBytes = 512 * 512 * 256 * 4
        chunkGb = chunkBytes / (1024 ** 3)
        totalAllocatedGb = 0.0

        gpuActive = [True] * deviceCount

        # Phase 1: Dedicated VRAM Saturation across all GPUs
        while any(gpuActive) and totalAllocatedGb < totalVram:
            allocatedInLoop = False
            for i in range(deviceCount):
                if not gpuActive[i]:
                    continue
                try:
                    tensor = torch.empty(chunkElements, dtype=torch.float32, device=f"cuda:{i}")
                    allocatedTensors.append(tensor)
                    perGpuFill[i] += chunkGb
                    totalAllocatedGb += chunkGb
                    allocatedInLoop = True
                except (RuntimeError, torch.cuda.OutOfMemoryError):
                    gpuActive[i] = False

                if totalAllocatedGb >= totalVram:
                    break

            if not allocatedInLoop:
                break
            time.sleep(0.01)

        # Phase 2: Host System RAM Spillover
        while totalAllocatedGb < targetAlloc:
            try:
                cpuTensor = torch.empty(chunkElements, dtype=torch.float32, device="cpu")
                allocatedTensors.append(cpuTensor)
                totalAllocatedGb += chunkGb
                time.sleep(0.01)
            except (RuntimeError, MemoryError):
                break

        print(f"[VRAM Clear] Allocation complete ({totalAllocatedGb:.2f} GB allocated). Releasing memory buffers...")
        time.sleep(1.5)

        # Phase 3: Cleanup and Synchronised Cache Flushing
        allocatedTensors.clear()
        del allocatedTensors
        gc.collect()

        for i in range(deviceCount):
            try:
                with torch.cuda.device(i):
                    torch.cuda.synchronize(i)
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass

        gc.collect()
        time.sleep(1.5)

        # Phase 4: Per-GPU Measurement & Verification
        perGpuFreed = []
        for i in range(deviceCount):
            memInfo = nvmlDeviceGetMemoryInfo(gpuHandles[i])
            usedAfterGb = memInfo.used / (1024 ** 3)
            freed = max(0.0, perGpuUsedBefore[i] - usedAfterGb)
            perGpuFreed.append(freed)
            print(f"  -> GPU {i} ({gpuNames[i]}): {perGpuUsedBefore[i]:.2f} GB --> {usedAfterGb:.2f} GB (Freed: {freed:.2f} GB)")

        freedVram = round(sum(perGpuFreed), 2)
        print(f"[VRAM Clear] Total VRAM freed across all GPUs: {freedVram:.2f} GB\n")

    except Exception as e:
        print(f"[VRAM Clear] Error during VRAM clearing: {e}. Continuing gracefully.")
        freedVram = 0.0

    finally:
        if nvmlInitialised:
            try:
                nvmlShutdown()
            except Exception:
                pass

    return freedVram


def checkExecutableExists(executablePath=LLAMACPP_EXECUTABLE):
    path = Path(executablePath)
    return path.is_file() or shutil.which(executablePath) is not None


class LlamaCppProcessInitiator:
    def __init__(self, 
                serverName: str, 
                model: str, 
                printLogsToTerminal: bool = True,
                killExistingProcesses: bool = True, 
                allowParallel: bool = True,
                argOverrides=None
        ):
        self.serverName = serverName
        modelIdentifier = str(model).strip()
        
        self.llamacppExecutable, self.command, self.modelConfig = buildLlamaCppCommandLine(
            modelIdentifier=modelIdentifier,
            allowParallel=allowParallel,
            argOverrides=argOverrides
        )

        if not checkExecutableExists(self.llamacppExecutable):
            raise FileNotFoundError(f"Llama.cpp executable not found at '{self.llamacppExecutable}' and not in PATH. Please ensure it is built and available.")
        
        if killExistingProcesses:
            if self.llamacppExecutable != LLAMACPP_EXECUTABLE:
                killRemainingLlamaCppProcesses(LLAMACPP_EXECUTABLE)
            killRemainingLlamaCppProcesses(self.llamacppExecutable)

        self.baseUrl = f"http://127.0.0.1:{LLAMACPP_PORT}"
        self.healthUrl = f"{self.baseUrl}/health"
        self.apiUrl = f"{self.baseUrl}/v1"

        self.process = None
        self.state = ServerState.STOPPED

        self.stateLock = threading.Lock()
        self.stdoutThread = None
        self.stderrThread = None
        self.logs = collections.deque(maxlen=1000)
        self.printLogsToTerminal = printLogsToTerminal

    def getCommandLine(self):
        return self.command

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
                cleanLine = line.rstrip()
                if self.getState() == ServerState.STARTING:
                    self.printToTerminal(cleanLine)
                try:
                    from llm.server_manager import serverManager
                    serverManager.recordLog(cleanLine)
                except Exception:
                    emitEvent("llamaCppLog", {"log": cleanLine})
                self.logs.append(cleanLine)
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

    def start(self, readyTimeout=120):
        with self.stateLock:
            if self.state in (ServerState.STARTING, ServerState.RUNNING):
                return
            
        print(f"[{self.serverName}] Starting Llama.cpp process with command:\n'{' '.join(self.command)}'")
                
        self.setState(ServerState.STARTING)
        self.process = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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

        
