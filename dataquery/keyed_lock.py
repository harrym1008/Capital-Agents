import threading
from contextlib import contextmanager
from typing import Dict


class KeyedLockManager:
    def __init__(self):
        self.managerLock = threading.Lock()
        self.keyLocks: Dict[str, threading.RLock] = {}
        self.refCounts: Dict[str, int] = {}

    @contextmanager
    def lockKey(self, key: str):
        with self.managerLock:
            if key not in self.keyLocks:
                self.keyLocks[key] = threading.RLock()
                self.refCounts[key] = 0
            self.refCounts[key] += 1
            targetLock = self.keyLocks[key]

        targetLock.acquire()
        try:
            yield
        finally:
            targetLock.release()
            with self.managerLock:
                self.refCounts[key] -= 1
                if self.refCounts[key] <= 0:
                    self.keyLocks.pop(key, None)
                    self.refCounts.pop(key, None)

    def getActiveKeyCount(self) -> int:
        with self.managerLock:
            return len(self.keyLocks)
