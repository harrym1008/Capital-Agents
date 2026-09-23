import threading
from contextlib import contextmanager
from typing import Dict


# Thread-safe lock manager providing fine-grained re-entrant locks per key
class KeyedLockManager:
    def __init__(self):
        self.managerLock = threading.Lock()
        self.keyLocks: Dict[str, threading.RLock] = {}
        self.refCounts: Dict[str, int] = {}

    @contextmanager
    def lockKey(self, key: str):
        with self.managerLock:
            if key not in self.keyLocks:
                # Create a new re-entrant lock for this key if it doesn't exist
                self.keyLocks[key] = threading.RLock()
                self.refCounts[key] = 0
            # Increment reference count for this key
            self.refCounts[key] += 1
            targetLock = self.keyLocks[key]

        # Acquire the lock for the specific key and yield control to the caller
        targetLock.acquire()
        try:
            yield
        finally:
            targetLock.release() 
            # Release and decrement reference count, prune lock if no longer in use
            with self.managerLock:
                self.refCounts[key] -= 1
                if self.refCounts[key] <= 0:
                    self.keyLocks.pop(key, None)
                    self.refCounts.pop(key, None)

    def getActiveKeyCount(self) -> int:
        with self.managerLock:
            return len(self.keyLocks)
