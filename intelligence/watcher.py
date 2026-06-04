"""
ARGUS V4 — Folder Watcher
Auto-detects new chart screenshots and triggers analysis.
Drop a PNG into the watch folder → ARGUS analyzes it automatically.
"""

import os, time, threading
from typing import Callable
from datetime import datetime


class FolderWatcher:
    def __init__(self, folder: str, callback: Callable[[str], None], poll_sec: int = 2):
        self._folder   = folder
        self._callback = callback
        self._poll     = poll_sec
        self._seen     = set()
        self._running  = False
        self._thread   = None

    def start(self):
        if self._running:
            return
        os.makedirs(self._folder, exist_ok=True)
        # Mark existing files as already seen so we don't re-analyze on startup
        self._seen = set(os.listdir(self._folder))
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def set_folder(self, folder: str):
        self._folder = folder
        os.makedirs(folder, exist_ok=True)
        self._seen = set(os.listdir(folder))

    def _loop(self):
        while self._running:
            try:
                current = set(os.listdir(self._folder))
                new     = current - self._seen
                for fname in sorted(new):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        full_path = os.path.join(self._folder, fname)
                        # Wait briefly so file finishes writing
                        time.sleep(0.5)
                        self._callback(full_path)
                self._seen = current
            except Exception:
                pass
            time.sleep(self._poll)
