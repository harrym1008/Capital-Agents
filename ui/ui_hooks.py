import threading
import re

# Thread-local storage tracking active agent role, colour, and stage in multi-threaded runs
_local = threading.local()

# Global event dispatch callback registered by UI server
eventCallback = None

stopRequestedEvent = threading.Event()
stopCallbacks = []


class SimulationStoppedException(Exception):
    # Exception raised when simulation execution is halted by user
    pass


def registerStopCallback(callback):
    # Register handler to invoke when stop is requested
    if callback not in stopCallbacks:
        stopCallbacks.append(callback)

def unregisterStopCallback(callback):
    # Remove registered stop handler
    if callback in stopCallbacks:
        stopCallbacks.remove(callback)

def setEventCallback(callback):
    # Set global UI event listener callback
    global eventCallback
    eventCallback = callback

def requestStop():
    # Signal stop event and trigger all registered cancellation callbacks
    stopRequestedEvent.set()
    for cb in list(stopCallbacks):
        try:
            cb()
        except Exception as e:
            print(f"Error in stop callback: {e}")

def resetStop():
    # Clear stop requested state
    stopRequestedEvent.clear()

def isStopRequested():
    # Check if cancellation has been requested
    return stopRequestedEvent.is_set()

def setCurrentAgent(agentRole, color):
    # Store active agent role and display colour in thread-local context
    _local.agentRole = agentRole
    _local.color = color

def getCurrentAgent():
    # Retrieve current agent identity dictionary from thread-local storage
    return {
        "role": getattr(_local, "agentRole", None),
        "color": getattr(_local, "color", None)
    }

def setAgentPhase(phase):
    # Set execution phase (e.g. reasoning, content) for current thread
    _local.phase = phase

def getAgentPhase():
    # Get active thread execution phase
    return getattr(_local, "phase", "raw")

def setCurrentStage(stageNum):
    # Update pipeline stage number in thread-local storage
    _local.stageNum = stageNum

def getCurrentStage():
    # Get active pipeline stage number
    return getattr(_local, "stageNum", 0)

_globalMilestoneId = None

def setCurrentMilestoneId(milestoneId):
    # Set active milestone ID in thread-local and global scope
    global _globalMilestoneId
    _local.milestoneId = milestoneId
    _globalMilestoneId = milestoneId

def getCurrentMilestoneId():
    # Retrieve current milestone ID
    return getattr(_local, "milestoneId", None) or _globalMilestoneId

def setCurrentCallId(callId):
    # Set current tool call ID in thread-local context
    _local.currentCallId = callId

def getCurrentCallId():
    # Get current tool call ID
    return getattr(_local, "currentCallId", None)

def emitEvent(eventType, data=None):
    # Dispatch UI event payload to registered callback with thread metadata
    global eventCallback
    # Allow cleanup and termination events through even if stop requested
    stopSafeEvents = {
        "simStopped", "simComplete", "error",
        "toolCallEnd", "agentRunEnd",
        "contentEnd", "reasoningEnd",
        "agentErrorMsg"
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
        currentMilestoneId = getCurrentMilestoneId()
        if currentMilestoneId:
            payload["milestoneId"] = currentMilestoneId
        currentCallId = getCurrentCallId()
        if currentCallId:
            payload["callId"] = currentCallId
        if data:
            payload.update(data)
        
        # Format percentage string and numeric values for tool progress
        if eventType == "toolCallProgress":
            rawProgress = payload.get("progress")
            if isinstance(rawProgress, (int, float)):
                clampedProgress = max(0.0, min(100.0, float(rawProgress)))
                formattedProgress = f"{int(clampedProgress)}%" if clampedProgress.is_integer() else f"{clampedProgress:.1f}%"
                payload["progress"] = formattedProgress
                payload["numericProgress"] = clampedProgress
            elif isinstance(rawProgress, str):
                payload["progress"] = rawProgress
                match = re.search(r"(\d+(?:\.\d+)?)%", rawProgress)
                if match:
                    try:
                        payload["numericProgress"] = float(match.group(1))
                    except ValueError:
                        pass

        try:
            eventCallback(payload)
        except Exception as e:
            print(f"Error in UI event callback: {e}")
