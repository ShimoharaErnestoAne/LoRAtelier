"""Tiny pub/sub for server-sent events.  Image updates are coalesced so a burst of analysis
results produces a few small UI refreshes instead of hundreds of redraws."""
from __future__ import annotations

import json
import queue
import threading
import time


class Bus:
    def __init__(self):
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._pending_images: dict[str, set[int]] = {}
        threading.Thread(target=self._flush_loop, daemon=True, name="event-flush").start()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            self._subs.append(q)
        return q

    def listeners(self) -> int:
        with self._lock:
            return len(self._subs)

    def unsubscribe(self, q):
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, type_: str, **data):
        msg = json.dumps({"type": type_, **data}, ensure_ascii=False)
        with self._lock:
            for q in list(self._subs):
                try:
                    q.put_nowait(msg)
                except queue.Full:  # slow client: drop it rather than block workers
                    self._subs.remove(q)
                    q.dropped = True

    def image_changed(self, project_id: str, image_id: int):
        with self._lock:
            self._pending_images.setdefault(project_id, set()).add(image_id)

    def _flush_loop(self):
        while True:
            time.sleep(0.4)
            with self._lock:
                pending, self._pending_images = self._pending_images, {}
            for pid, ids in pending.items():
                self.publish("images", project=pid, ids=sorted(ids))


bus = Bus()
