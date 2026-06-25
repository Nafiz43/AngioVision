"""
Global concurrency gate for the expensive SSE endpoints (/api/query and
/api/image-query).

Both endpoints share ONE Ollama server (and likely one GPU) plus the image
embedding model, so running many at once degrades or crashes them. This module
limits how many heavy jobs run simultaneously and queues the overflow FIFO,
streaming "queued"/"queue_update"/"slot_acquired" SSE events to the client while
it waits so the request proceeds automatically when a slot frees.

NOTE ON SCALING: this is an *in-process* gate built on threading primitives,
which is correct because run_server.py uses app.run(threaded=True) (one process,
many request threads). If the server is ever scaled to multiple processes
(e.g. gunicorn workers), this gate will NOT span processes and would need a
shared backend such as Redis or a cross-process file lock (like the one used in
the ml-rag ingestion sync).
"""

import time
import threading
from typing import Any, Callable, Dict, Iterator, Optional

from .state import state


class QueueFull(Exception):
    """Raised when the waiting line is already at MAX_QUEUE (backpressure)."""


class ConcurrencyGate:
    """Thread-safe FIFO ticketing gate over a bounded semaphore.

    Each request takes a monotonically increasing ticket id under a lock. Only
    the ticket at the front of the line (position 1) attempts to grab the
    semaphore, which guarantees arrival-order (FIFO) service. Waiters poll on a
    heartbeat interval; when the running job releases its slot, the current head
    of the line acquires it on its next poll.
    """

    def __init__(self, max_concurrency: int, max_queue: int) -> None:
        self.max_concurrency = max(1, int(max_concurrency))
        self.max_queue = max(0, int(max_queue))
        self._sem = threading.BoundedSemaphore(self.max_concurrency)
        self._lock = threading.Lock()
        self._next_id = 0
        self._waiting: list[int] = []  # ticket ids not yet started, arrival order

    def acquire_ticket(self) -> int:
        """Register a new waiting ticket, or raise QueueFull for backpressure."""
        with self._lock:
            if len(self._waiting) >= self.max_queue:
                raise QueueFull()
            tid = self._next_id
            self._next_id += 1
            self._waiting.append(tid)
            return tid

    def _position(self, tid: int) -> int:
        """1-based position in line (count of not-yet-started tickets with a
        smaller id, plus one). 0 if the ticket is no longer waiting."""
        with self._lock:
            try:
                return self._waiting.index(tid) + 1
            except ValueError:
                return 0

    def _total_waiting(self) -> int:
        with self._lock:
            return len(self._waiting)

    def _start(self, tid: int) -> None:
        with self._lock:
            if tid in self._waiting:
                self._waiting.remove(tid)

    def release(self, tid: int, acquired: bool) -> None:
        """Always call in a finally: drop the ticket from the line, and release
        the semaphore slot if this ticket had acquired one."""
        with self._lock:
            if tid in self._waiting:
                self._waiting.remove(tid)
        if acquired:
            try:
                self._sem.release()
            except ValueError:
                pass  # already released; defensive against double-release

    def wait_for_slot(
        self,
        tid: int,
        emit: Callable[[Dict[str, Any]], str],
        heartbeat: float = 2.0,
    ) -> Iterator[str]:
        """Generator that yields SSE strings while the ticket waits its turn,
        and returns (StopIteration) once it has acquired a slot.

        Emits "queued" on the first wait and "queue_update" on every subsequent
        poll (which doubles as a ~heartbeat keepalive). If a slot is free
        immediately, it returns without yielding anything (caller may still send
        slot_acquired)."""
        first = True
        while True:
            pos = self._position(tid)
            # Only the head of the line competes for a slot -> FIFO ordering.
            if pos == 1 and self._sem.acquire(blocking=False):
                self._start(tid)
                return
            payload = {
                "position":      pos,
                "ahead":         max(0, pos - 1),
                "total_waiting": self._total_waiting(),
            }
            if first:
                first = False
                yield emit({"event": "queued", **payload})
            else:
                yield emit({"event": "queue_update", **payload})
            time.sleep(heartbeat)


# ── Lazily-built process-wide singleton ──────────────────────────────────────
_gate: Optional[ConcurrencyGate] = None
_gate_lock = threading.Lock()


def get_gate() -> ConcurrencyGate:
    """Return the shared gate, built once from state on first use."""
    global _gate
    with _gate_lock:
        if _gate is None:
            _gate = ConcurrencyGate(
                getattr(state, "max_concurrency", 1),
                getattr(state, "max_queue", 20),
            )
        return _gate


def gated(
    emit: Callable[[Dict[str, Any]], str],
    work: Callable[[], Iterator[str]],
    heartbeat: float = 2.0,
) -> Iterator[str]:
    """Wrap a heavy SSE pipeline with the global concurrency gate.

    Yields "queued"/"queue_update" events while waiting in line, a
    "slot_acquired" event when a slot is obtained, then delegates to ``work()``
    (the existing, unchanged pipeline). On backpressure it yields a single
    "busy" event instead of queueing. The semaphore slot and queue ticket are
    ALWAYS released on exit — normal completion, error, OR client disconnect
    (the generator's GeneratorExit) — so an abandoned request frees its slot and
    everyone behind it advances on the next heartbeat poll.

    Args:
        emit:      formats an event dict into an SSE ``data: ...`` string.
        work:      zero-arg callable returning the heavy pipeline's SSE iterator.
        heartbeat: seconds between queue_update keepalives while waiting.
    """
    gate = get_gate()
    try:
        ticket = gate.acquire_ticket()
    except QueueFull:
        yield emit({
            "event":   "busy",
            "message": "Server is at capacity, please retry shortly.",
        })
        return

    acquired = False
    try:
        for sse in gate.wait_for_slot(ticket, emit, heartbeat=heartbeat):
            yield sse
        acquired = True
        yield emit({"event": "slot_acquired"})
        yield from work()
    finally:
        gate.release(ticket, acquired)
