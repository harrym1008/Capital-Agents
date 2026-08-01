import threading
from ui.ui_hooks import emitEvent


class TokenCostTracker:
    def __init__(self):
        self.lock = threading.Lock()

        # Task level usage (per single run of the boardroom)
        self.taskInputToks = 0
        self.taskOutputToks = 0
        self.taskCacheToks = 0
        self.taskCost = 0.0

        # Session level usage (across all runs of that specific server)
        self.sessionInputToks = 0
        self.sessionOutputToks = 0
        self.sessionCacheToks = 0
        self.sessionCost = 0.0


    def recordUsage(self, usage):
        if usage is None:
            return None

        usageDict = usage.model_dump() if hasattr(usage, "model_dump") else usage

        promptTokens = usageDict.get("prompt_tokens", 0)
        completionTokens = usageDict.get("completion_tokens", 0)
        cost = usageDict.get("cost", 0.0)

        # OpenRouter exposes the monetary cost on the usage object (usage.cost).
        # It may not survive model_dump() if the SDK drops unknown fields, so
        # fall back to direct attribute access when the dict has no cost.
        if not cost and hasattr(usage, "cost"):
            cost = getattr(usage, "cost", 0.0) or 0.0

        cachedTokens = 0
        promptDetails = usageDict.get("prompt_tokens_details")
        if promptDetails is not None:
            cachedTokens = promptDetails.get("cached_tokens", 0)

        with self.lock:
            self.taskInputToks += promptTokens
            self.taskOutputToks += completionTokens
            self.taskCacheToks += cachedTokens
            self.taskCost += cost

            self.sessionInputToks += promptTokens
            self.sessionOutputToks += completionTokens
            self.sessionCacheToks += cachedTokens
            self.sessionCost += cost

            payload = self.buildPayload(promptTokens, completionTokens, cachedTokens, cost)

        emitEvent("costUpdate", payload)
        return payload


    def newTask(self):
        with self.lock:
            self.taskInputToks = 0
            self.taskOutputToks = 0
            self.taskCacheToks = 0
            self.taskCost = 0.0

        payload = self.buildPayload(0, 0, 0, 0.0)
        emitEvent("costUpdate", payload)
        return payload


    def calculateCacheHit(self, cachedTokens, promptTokens):
        if promptTokens == 0:
            return 0.0
        return round(cachedTokens / promptTokens * 100, 1)


    def buildPayload(self, promptTokens, completionTokens, cachedTokens, cost):
        return {
            "stream": {
                "input": promptTokens,
                "output": completionTokens,
                "cached": cachedTokens,
                "cacheHitRate": self.calculateCacheHit(cachedTokens, promptTokens),
                "cost": round(cost, 6)
            },
            "task": {
                "input": self.taskInputToks,
                "output": self.taskOutputToks,
                "cached": self.taskCacheToks,
                "cacheHitRate": self.calculateCacheHit(self.taskCacheToks, self.taskInputToks),
                "cost": round(self.taskCost, 6)
            },
            "session": {
                "input": self.sessionInputToks,
                "output": self.sessionOutputToks,
                "cached": self.sessionCacheToks,
                "cacheHitRate": self.calculateCacheHit(self.sessionCacheToks, self.sessionInputToks),
                "cost": round(self.sessionCost, 6)
            }
        }


    def getPayload(self):
        with self.lock:
            return self.buildPayload(0, 0, 0, 0.0)


    def reset(self):
        with self.lock:
            self.taskInputToks = 0
            self.taskOutputToks = 0
            self.taskCacheToks = 0
            self.taskCost = 0.0

            self.sessionInputToks = 0
            self.sessionOutputToks = 0
            self.sessionCacheToks = 0
            self.sessionCost = 0.0

            payload = self.buildPayload(0, 0, 0, 0.0)
        emitEvent("costUpdate", payload)
        return payload
