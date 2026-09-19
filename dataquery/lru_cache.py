import os
import sys
import time
import pickle
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
        mem = obj.memory_usage(deep=True, index=True)
        if isinstance(mem, pd.Series):
            return int(mem.sum())
        return int(mem)

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
        # Final resort... may heavily underestimate, not much choice
        return sys.getsizeof(obj)
    


@dataclass
class CacheEntry:
    value: any
    size: int
    lastAccess: float


PERSIST_FILE_PATH = "_cache/persist_lru_cache.bin"
PERSIST_DEBOUNCE = 60   # seconds



# Least recently used cache
class LRUCache:
    def __init__(self, maxSizeBytes, main=False):
        self.maxSizeBytes = maxSizeBytes
        self.currentSizeBytes = 0
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.Lock()

        self.main = main
        self.persistThread = None

        if self.main:
            self.loadFromDisk()

            self.addEvent = threading.Event()
            self.stopEvent = threading.Event()
            self.lastAddTime = 0.0

            self.persistThread = threading.Thread(target=self.persistLoop, daemon=True)
            self.persistThread.start()


    def persistLoop(self):
        while not self.stopEvent.is_set():
            self.addEvent.wait()
            if self.stopEvent.is_set():
                break
            self.addEvent.clear()

            while not self.stopEvent.is_set():
                remaining = PERSIST_DEBOUNCE - (time.time() - self.lastAddTime)
                if remaining <= 0:
                    break
                if self.stopEvent.wait(remaining):
                    return  # stopped while waiting

            if self.stopEvent.is_set():
                break

            self.saveToDisk()


    def loadFromDisk(self):
        if not os.path.exists(PERSIST_FILE_PATH):
            print(f"LRUCache: No persisted cache file found at {PERSIST_FILE_PATH}. Starting with empty cache...")
            return      # No persisted cache file exists, nothing to load

        try:
            with open(PERSIST_FILE_PATH, "rb") as f:
                loadedEntries = pd.read_pickle(f)

            with self.lock:
                self.entries = loadedEntries
                self.currentSizeBytes = sum(entry.size for entry in self.entries.values())
        except Exception as e:
            print(f"LRUCache: Error loading cache from disk: {e}", file=sys.stderr)


    def saveToDisk(self):
        with self.lock:
            snapshot = OrderedDict()
            for key, entry in self.entries.items():
                try:
                    pickle.dumps(entry.value)
                except Exception:
                    # Value cannot be pickled
                    continue
                snapshot[key] = entry

        try:
            os.makedirs(os.path.dirname(PERSIST_FILE_PATH), exist_ok=True)

            tmpPath = PERSIST_FILE_PATH + ".tmp"
            with open(tmpPath, "wb") as f:
                pickle.dump(snapshot, f)
            os.replace(tmpPath, PERSIST_FILE_PATH)
            print(f"LRUCache: Cache persisted to disk at {PERSIST_FILE_PATH}")

        except Exception as e:
            print(f"LRUCache: Error saving cache to disk: {e}")



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

            if self.currentSizeBytes + sizeBytes > self.maxSizeBytes:
                targetSizeBytes = int(self.maxSizeBytes * 0.90)     # Evict entries until we are below 90% of max size to avoid frequent evictions
                while self.entries and self.currentSizeBytes + sizeBytes > targetSizeBytes:
                    _, evicted = self.entries.popitem(last=False)
                    self.currentSizeBytes -= evicted.size

            self.entries[key] = CacheEntry(value=value, size=sizeBytes, lastAccess=now)
            self.currentSizeBytes += sizeBytes

        if self.main:
            self.lastAddTime = now
            self.addEvent.set()


    def getCacheUsage(self):
        with self.lock:
            return self.currentSizeBytes, self.maxSizeBytes
        
    def getCacheUsagePercent(self):
        usage, max = self.getCacheUsage()
        return (usage / max) * 100 if max > 0 else 0

    def getCacheUsagePrettyString(self):
        def formatBytes(size):
            for unit in ["B", "KB", "MB"]:
                if size < 1024:
                    return f"{size:.2f} {unit}"
                size /= 1024
            return f"{size:.2f} GB"
        
        usage, max = self.getCacheUsage()
        usagePct = self.getCacheUsagePercent()
        usageStr = formatBytes(usage)
        maxStr = formatBytes(max)
        return f"{usageStr} / {maxStr} ({usagePct:.3f}%)"


    def __del__(self):
        if self.main and self.persistThread is not None:
            # Kill the thread, dont bother with what was waiting to be persisted
            self.stopEvent.set()
            self.addEvent.set()