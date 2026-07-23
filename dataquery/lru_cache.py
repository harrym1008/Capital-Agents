import sys
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd


def getSize(obj):
    if obj is None:
        return 0

    # Pandas Dataframes and Series
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return int(obj.memory_usage(deep=True, index=True).sum())

    # Numpy arrays
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)

    # Basic bytestreams
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return len(obj)

    # Estimate via pympler
    try:
        from pympler import asizeof
        return asizeof.asizeof(obj)
    except ImportError:
        # Final resort - may heavily underestimate, not much choice
        return sys.getsizeof(obj)
    


@dataclass
class CacheEntry:
    value: any
    size: int
    lastAccess: float


# Least recently used cache
class LRUCache:
    def __init__(self, maxSizeBytes):
        self.maxSizeBytes = maxSizeBytes
        self.currentSizeBytes = 0
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.Lock()


    def get(self, key):
        with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                return None
            entry.lastAccess = time.time()
            self.entries.move_to_end(key)
            return entry.value
    
    
    def put(self, key, value):
        if value is None:
            # Store None values without calculating memory usage
            with self.lock:
                if key in self.entries:
                    old = self.entries.pop(key)
                    self.currentSizeBytes -= old.size
                self.entries[key] = CacheEntry(value, 0, time.time())
                return
        
        sizeBytes = getSize(value)
        now = time.time()

        with self.lock:
            if sizeBytes > self.maxSizeBytes:
                return      # Entry is too large to fit in the cache

            if key in self.entries:
                old = self.entries.pop(key)
                self.currentSizeBytes -= old.size

            while self.entries and self.currentSizeBytes + sizeBytes > self.maxSizeBytes:
                _, evicted = self.entries.popitem(last=False)
                self.currentSizeBytes -= evicted.size

            self.entries[key] = CacheEntry(value=value, size=sizeBytes, lastAccess=now)
            self.currentSizeBytes += sizeBytes


    def getCacheUsage(self):
        with self.lock:
            return self.currentSizeBytes, self.maxSizeBytes
        
    def getCacheUsagePercent(self):
        usage, max = self.getCacheUsage()
        return (usage / max) * 100 if max > 0 else 0