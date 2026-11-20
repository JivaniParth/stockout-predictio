"""
Application Layer Port: Task Queue Interface.
Decouples asynchronous background worker execution from queue implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional


class ITaskQueue(ABC):
    """Abstract contract for asynchronous task submission and execution."""

    @abstractmethod
    def enqueue(
        self,
        task_id: str,
        handler: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Enqueues a task for asynchronous background processing."""
        pass

    @abstractmethod
    def get_queue_size(self) -> int:
        """Returns the number of tasks pending execution in the queue."""
        pass
