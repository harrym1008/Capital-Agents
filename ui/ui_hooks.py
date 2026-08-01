import threading

# Thread-local storage to track the active agent role, color, and stage number in multi-threaded runs
_local = threading.local()

# Global callback function registered by the web UI server
eventCallback = None

stopRequestedEvent = threading.Event()

class SimulationStoppedException(Exception):
    pass

def setEventCallback(callback):
    global eventCallback
    eventCallback = callback

def requestStop():
    stopRequestedEvent.set()

def resetStop():
    stopRequestedEvent.clear()

def isStopRequested():
    return stopRequestedEvent.is_set()

def setCurrentAgent(agentRole, color):
    _local.agentRole = agentRole
    _local.color = color

def getCurrentAgent():
    return {
        "role": getattr(_local, "agentRole", None),
        "color": getattr(_local, "color", None)
    }

def setAgentPhase(phase):
    _local.phase = phase

def getAgentPhase():
    return getattr(_local, "phase", "raw")

def setCurrentStage(stageNum):
    _local.stageNum = stageNum

def getCurrentStage():
    return getattr(_local, "stageNum", 0)

def emitEvent(eventType, data=None):
    global eventCallback
    # Allow cleanup/finalisation events to fire even when stop is requested,
    # so that the UI can properly close streaming states and display the stop message.
    stopSafeEvents = {
        "simStopped", "simComplete", "error",
        "toolCallEnd", "agentRunEnd",
        "contentEnd", "reasoningEnd"
    }
    if eventType not in stopSafeEvents and isStopRequested():
        raise SimulationStoppedException("Simulation stopped by user.")
    if eventCallback:
        agentInfo = getCurrentAgent()
        payload = {
            "type": eventType,
            "agentRole": agentInfo["role"],
            "agentColor": agentInfo["color"],
            "phase": getAgentPhase(),
            "stageNum": getCurrentStage(),
            "threadId": threading.get_ident()
        }
        if data:
            payload.update(data)
        try:
            eventCallback(payload)
        except Exception as e:
            # Prevent crashing if callback fails
            print(f"Error in UI event callback: {e}")

