import subprocess
import threading
import urllib.error
import urllib.request
import collections
from enum import Enum

import psutil
import time

from llm.llamacpp_args import LLAMACPP_EXECUTABLE, LLAMACPP_PORT, LlamaCppModel, EMPTY_ARG, LLAMACPP_MODEL_TO_ARGS


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
        time.sleep(3)



class LlamaCppProcessInitiator:
    def __init__(self, model: LlamaCppModel, argOverrides=None):
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

        
