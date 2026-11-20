"""
Layer 3 Adapter: Scalable Asynchronous Background Task Queue.
Implements ITaskQueue using a thread-pool worker queue with bounded buffer.
Decouples long-running batch matrix predictions from synchronous HTTP request threads.
"""

from concurrent.futures import ThreadPoolExecutor
import logging
from queue import Queue, Full
import threading
from typing import Any, Callable, Dict, Optional

from src.application.ports.queue import ITaskQueue

logger = logging.getLogger("hscm.infrastructure.queue")


class ConcurrentTaskQueue(ITaskQueue):
    """
    Production-grade concurrent task queue.
    Executes asynchronous inference jobs on isolated background threads,
    preventing ASGI event-loop starvation and worker thread timeouts.
    """

    _instance: Optional["ConcurrentTaskQueue"] = None
    _lock = threading.Lock()

    def __init__(self, max_workers: int = 4, max_queue_size: int = 500):
        self._max_workers = max_workers
        self._max_queue_size = max_queue_size
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="hscm-worker-",
        )
        self._pending_count = 0
        self._count_lock = threading.Lock()

    @classmethod
    def get_instance(cls, max_workers: int = 4) -> "ConcurrentTaskQueue":
        with cls._lock:
            if cls._instance is None:
                cls._instance = ConcurrentTaskQueue(max_workers=max_workers)
            return cls._instance

    def enqueue(
        self,
        task_id: str,
        handler: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        with self._count_lock:
            if self._pending_count >= self._max_queue_size:
                logger.warning("Task queue full. Rejecting task %s (Backpressure applied)", task_id)
                return False
            self._pending_count += 1

        def _worker_wrapper():
            try:
                logger.info("Starting background task: %s", task_id)
                handler(*args, **kwargs)
                logger.info("Completed background task: %s", task_id)
            except Exception as e:
                logger.error("Background task %s failed with exception: %s", task_id, e, exc_info=True)
            finally:
                with self._count_lock:
                    self._pending_count -= 1

        self._executor.submit(_worker_wrapper)
        return True

    def get_queue_size(self) -> int:
        with self._count_lock:
            return self._pending_count

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)
