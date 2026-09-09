import threading

# Thread-local storage to track the active agent role, color, and stage number in multi-threaded runs
_local = threading.local()

# Global callback function registered by the web UI server
eventCallback = None

stopRequestedEvent = threading.Event()
_stopCallbacks = []

class SimulationStoppedException(Exception):
    pass

def registerStopCallback(callback):
    if callback not in _stopCallbacks:
        _stopCallbacks.append(callback)

def unregisterStopCallback(callback):
    if callback in _stopCallbacks:
        _stopCallbacks.remove(callback)

def setEventCallback(callback):
    global eventCallback
    eventCallback = callback

def requestStop():
    stopRequestedEvent.set()
    for cb in list(_stopCallbacks):
        try:
            cb()
        except Exception as e:
            print(f"Error in stop callback: {e}")

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

def setCurrentCallId(callId):
    _local.currentCallId = callId

def getCurrentCallId():
    return getattr(_local, "currentCallId", None)

def emitEvent(eventType, data=None):
    global eventCallback
    # Allow cleanup/finalisation events to fire even when stop is requested,
    # so that the UI can properly close streaming states and display the stop message.
    stopSafeEvents = {
        "simStopped", "simComplete", "error",
        "toolCallEnd", "agentRunEnd",
        "contentEnd", "reasoningEnd",
        "rateLimit"
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
        currentCallId = getCurrentCallId()
        if currentCallId:
            payload["callId"] = currentCallId
        if data:
            payload.update(data)
        
        # Format progression string if event is toolCallProgress
        if eventType == "toolCallProgress":
            rawProgress = payload.get("progress")
            if isinstance(rawProgress, (int, float)):
                clampedProgress = max(0.0, min(100.0, float(rawProgress)))
                formattedProgress = f"{int(clampedProgress)}%" if clampedProgress.is_integer() else f"{clampedProgress:.1f}%"
                payload["progress"] = formattedProgress
                payload["numericProgress"] = clampedProgress

        try:
            eventCallback(payload)
        except Exception as e:
            # Prevent crashing if callback fails
            print(f"Error in UI event callback: {e}")

