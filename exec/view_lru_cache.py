import os
import sys
import json
import pickle
import tkinter as tk
from tkinter import ttk, scrolledtext

import numpy as np
import pandas as pd

CACHE_FILENAME = "_cache/persist_lru_cache.bin"


def findCacheFile():
    dirpath = os.path.abspath(os.getcwd())
    while True:
        candidate = os.path.join(dirpath, CACHE_FILENAME)
        if os.path.exists(candidate):
            return candidate, os.path.dirname(os.path.dirname(candidate))
        parent = os.path.dirname(dirpath)
        if parent == dirpath:
            return None, None
        dirpath = parent


def isJsonable(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


def safeStr(obj):
    """Best-effort string conversion that never raises (some __repr__ hit the network)."""
    for fn in (str, repr):
        try:
            return fn(obj)
        except Exception:
            continue
    return f"<{type(obj).__name__} (unprintable)>"


def convert(obj):
    """Best-effort textual conversion for non-table values."""
    if isinstance(obj, np.ndarray):
        return np.array2string(obj, max_line_width=200)
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return repr(obj)
    if isJsonable(obj):
        return json.dumps(obj, indent=2)
    return safeStr(obj)


def formatBytes(size):
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def loadEntries(path):
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        data = pickle.load(f)
    entries = []
    for key, value in data.items():
        size = 0
        if hasattr(value, "value"):
            key = key if isinstance(key, str) else repr(key)
            size = getattr(value, "size", 0)
            value = value.value
        entries.append((key, value, size))
    return entries


class Viewer(tk.Tk):
    def __init__(self, entries):
        super().__init__()
        self.title("LRU Cache Viewer")
        self.entries = entries
        self.values = [value for _, value, _ in entries]
        self.sizes = [size for _, _, size in entries]

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Draggable split between left and right panes
        self.panes = ttk.PanedWindow(self, orient="horizontal")
        self.panes.grid(row=0, column=0, sticky="nsew")

        # ---- Left: entry list ----
        left = ttk.Frame(self.panes, padding=5)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        ttk.Label(left, text="Cache Entries").grid(row=0, column=0, sticky="w")

        self.listbox = tk.Listbox(left, width=40, activestyle="dotbox")
        self.listbox.grid(row=1, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.listbox.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=scroll.set)

        for i, (key, _, _) in enumerate(entries):
            self.listbox.insert(tk.END, f"[{i}] {key}")

        # ---- Right: value display ----
        right = ttk.Frame(self.panes, padding=5)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self.valueLabel = ttk.Label(right, text="Value")
        self.valueLabel.grid(row=0, column=0, sticky="w")

        self.displayFrame = ttk.Frame(right)
        self.displayFrame.grid(row=1, column=0, sticky="nsew")
        self.displayFrame.columnconfigure(0, weight=1)
        self.displayFrame.rowconfigure(0, weight=1)

        self.text = scrolledtext.ScrolledText(self.displayFrame, width=80)
        self.text.grid(row=0, column=0, sticky="nsew")

        self.tree = None  # created lazily for dataframes

        self.panes.add(left, weight=1)
        self.panes.add(right, weight=3)

        self.listbox.bind("<<ListboxSelect>>", self.onSelect)

        if entries:
            self.listbox.selection_set(0)
            self.updateDisplay(0)

    def _clearDisplay(self):
        if self.tree is not None:
            self.tree.destroy()
            self.tree = None
        self.text.grid(row=0, column=0, sticky="nsew")
        self.text.delete("1.0", tk.END)

    def _showDataFrame(self, df):
        self.text.grid_remove()
        if self.tree is None:
            self.tree = ttk.Treeview(self.displayFrame, show="headings")
            self.tree.grid(row=0, column=0, sticky="nsew")
            vsb = ttk.Scrollbar(self.displayFrame, orient="vertical", command=self.tree.yview)
            vsb.grid(row=0, column=1, sticky="ns")
            hsb = ttk.Scrollbar(self.displayFrame, orient="horizontal", command=self.tree.xview)
            hsb.grid(row=1, column=0, sticky="ew")
            self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(df.columns)

        for col in df.columns:
            self.tree.heading(col, text=str(col))
            self.tree.column(col, width=120, anchor="w", stretch=True)

        for _, row in df.iterrows():
            self.tree.insert("", tk.END, values=[str(v) for v in row.tolist()])

    def onSelect(self, event=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        self.updateDisplay(sel[0])

    def updateDisplay(self, index):
        value = self.values[index]
        size = self.sizes[index]
        self.valueLabel.config(text=f"Value  ({type(value).__name__})  |  {formatBytes(size)}")

        if isinstance(value, pd.DataFrame):
            self._showDataFrame(value)
        elif isinstance(value, pd.Series):
            self._clearDisplay()
            self.text.insert(tk.END, value.to_string())
        else:
            self._clearDisplay()
            self.text.insert(tk.END, convert(value))


def main():
    cachePath, cacheRoot = findCacheFile()
    if cachePath is None:
        print(f"No persisted cache file found at {CACHE_FILENAME}.")
        return
    if cacheRoot and cacheRoot not in sys.path:
        sys.path.insert(0, cacheRoot)

    entries = loadEntries(cachePath)
    if not entries:
        print(f"No persisted cache file found at {cachePath}.")
        return
    print(f"Loaded {len(entries)} cache entries from {cachePath}.")
    Viewer(entries).mainloop()


if __name__ == "__main__":
    main()
