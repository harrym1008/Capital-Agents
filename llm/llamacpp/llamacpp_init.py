import subprocess
import threading
import urllib.error
import urllib.request
import collections
from enum import Enum

import psutil
import time


from llm.llamacpp.llamacpp_args import LLAMACPP_EXECUTABLE, LLAMACPP_PORT, LlamaCppModel, EMPTY_ARG, LLAMACPP_MODEL_TO_ARGS


class ServerState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"


def killExistingLlamaCppProcesses():
    killed = False
    for process in psutil.process_iter(["pid", "name"]):
        try:
            if process.info["name"].lower() == LLAMACPP_EXECUTABLE.lower():
                print(f"Killing PID {process.pid}")
                process.kill()
                killed = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if killed:
        time.sleep(2)


def rudimentaryVramClear():
    try:
        import gc, sys, time, psutil, torch
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
        
        skip = False

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
            
            else:
                print(f"Clearing VRAM: Before: {usedVramBefore:.2f}/{totalVram:.2f} GB...", end="\r", flush=True)

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

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            for module in ["torch", "psutil"]:
                sys.modules.pop(module, None)

            gc.collect()
            time.sleep(1)

            if not skip:
                memInfo = nvmlDeviceGetMemoryInfo(handle)
                usedVramAfter = round(memInfo.used / (1024 ** 3), 2)
                print(f"\n  ↳ After: {usedVramAfter:.2f} GB | Freed: {usedVramBefore - usedVramAfter:.2f} GB")

            sys.modules.pop("pynvml", None)
            gc.collect()
            time.sleep(1)

    except Exception as e:
        print(f"Error during rudimentary VRAM clearing: {e}. Continuing without clearing VRAM.")



class LlamaCppProcessInitiator:
    def __init__(self, model: LlamaCppModel, killExistingProcesses: bool = True, argOverrides=None):
        if killExistingProcesses:
            killExistingLlamaCppProcesses()

        self.args = LLAMACPP_MODEL_TO_ARGS.get(model, {}).copy()
        self.args.update(argOverrides or {})

        self.baseUrl = f"http://{self.args.get('--host', '127.0.0.1')}:{self.args.get('--port', LLAMACPP_PORT)}"
        self.healthUrl = f"{self.baseUrl}/health"
        self.apiUrl = f"{self.baseUrl}/v1"

        self.process = None
        self.state = ServerState.STOPPED

        self.stateLock = threading.Lock()
        self.stdoutThread = None
        self.stderrThread = None
        self.logs = collections.deque(maxlen=1000)


    def setState(self, newState):
        with self.stateLock:
            self.state = newState


    def getState(self):
        with self.stateLock:
            return self.state


    
    def readStream(self, stream, tag):
        try:
            for line in iter(stream.readline, ''):
                if self.getState() == ServerState.STARTING:
                    print(line.rstrip())
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
            
        command = [LLAMACPP_EXECUTABLE]
        for key, value in self.args.items():
            command.append(key)
            if value != EMPTY_ARG:
                command.append(value)
                
        self.setState(ServerState.STARTING)
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, bufsize=1)
        print(f"Started Llama.cpp process PID={self.process.pid}")

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
                raise RuntimeError("Llama.cpp process terminated unexpectedly")
            
            if self.isReady():
                self.setState(ServerState.RUNNING)
                print("Llama.cpp server has loaded and is ready to take requests")
                return
            
            threading.Event().wait(step)
            waited += step

        self.stop()
        raise TimeoutError(f"Llama.cpp process did not become ready within {readyTimeout} seconds")


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
                print("Llama.cpp process has been stopped gracefully")
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
                print("Llama.cpp process was forcefully killed")

        self.process = None
        self.setState(ServerState.STOPPED)

        
