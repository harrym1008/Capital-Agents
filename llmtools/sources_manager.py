import threading
import time
import json
import copy
from typing import Dict, Any, List, Optional
from ui.ui_hooks import emitEvent


class SourcesManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.sources: List[Dict[str, Any]] = []
        self.hashToCitation: Dict[int, int] = {}

    def _computeHash(self, result: Any) -> int:
        if isinstance(result, dict):
            resultToHash = {k: v for k, v in result.items() if k not in ("citationNumber", "toolCitationNumber")}
        else:
            resultToHash = result
        try:
            serialised = json.dumps(resultToHash, sort_keys=True, default=str)
        except Exception:
            serialised = str(resultToHash)
        return hash(serialised)

    def addSource(self, toolName: str, args: Dict[str, Any], result: Dict[str, Any]) -> Optional[int]:
        with self.lock:
            if isinstance(result, dict) and "error" in result:
                return None

            resultHash = self._computeHash(result)
            if resultHash in self.hashToCitation:
                return self.hashToCitation[resultHash]

            citationNumber = len(self.sources) + 1
            sourceRecord = {
                "citationNumber": citationNumber,
                "toolCitationNumber": citationNumber,
                "toolName": toolName,
                "args": copy.deepcopy(args) if args else {},
                "result": copy.deepcopy(result),
                "timestamp": time.time(),
                "hash": resultHash
            }
            self.sources.append(sourceRecord)
            self.hashToCitation[resultHash] = citationNumber

            emitEvent("newSource", {
                "source": {
                    "citationNumber": citationNumber,
                    "toolCitationNumber": citationNumber,
                    "toolName": toolName,
                    "args": sourceRecord["args"],
                    "result": sourceRecord["result"],
                    "timeStr": time.strftime("%H:%M:%S", time.localtime(sourceRecord["timestamp"]))
                }
            })

            return citationNumber

    def reset(self) -> None:
        with self.lock:
            self.sources.clear()
            self.hashToCitation.clear()
            emitEvent("resetSources", {})

    def getSources(self) -> List[Dict[str, Any]]:
        with self.lock:
            return copy.deepcopy(self.sources)

    def getSourceByCitation(self, citationNumber: int) -> Optional[Dict[str, Any]]:
        with self.lock:
            if 1 <= citationNumber <= len(self.sources):
                return copy.deepcopy(self.sources[citationNumber - 1])
            return None
