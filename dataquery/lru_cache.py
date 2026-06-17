import time
import threading
from collections import OrderedDict
from dataclasses import dataclass

import pandas as pd


@dataclass
class CacheEntry:
    df: pd.DataFrame
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
            return entry.df
    
    
    def put(self, key, df):
        sizeBytes = int(df.memory_usage(deep=True, index=True).sum())
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

            self.entries[key] = CacheEntry(df=df, size=sizeBytes, lastAccess=now)
            self.currentSizeBytes += sizeBytes