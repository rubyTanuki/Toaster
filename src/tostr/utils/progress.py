import threading
from typing import Dict
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, SpinnerColumn

class ProgressTracker:
    def __init__(self, rich_progress: Progress, include_resolve: bool = True, include_describe: bool = True, include_embed: bool = True):
        self._progress = rich_progress
        self._tasks: Dict[str, int] = {}
        self._lock = threading.Lock()

        # Initialize the static bars
        if include_resolve:
            self._tasks['resolve'] = self._progress.add_task("[cyan]Resolving Dependencies:", total=0)
        if include_describe:
            self._tasks['describe'] = self._progress.add_task("[magenta]Describing: ", total=0)
        if include_embed:
            self._tasks['embed'] = self._progress.add_task("[green]Embedding:  ", total=0)

    def enqueue(self, task_type: str, amount: int = 1):
        """Increments the total for a specific task type."""
        with self._lock:
            if task_type in self._tasks:
                task_id = self._tasks[task_type]
                current_total = self._progress.tasks[task_id].total or 0
                self._progress.update(task_id, total=current_total + amount)

    def advance(self, task_type: str, amount: int = 1):
        """Advances the completion counter for a specific task type."""
        with self._lock:
            if task_type in self._tasks:
                self._progress.update(self._tasks[task_type], advance=amount)

    def finish(self):
        """Forces all bars to complete."""
        with self._lock:
            for task_id in self._tasks.values():
                task = self._progress.tasks[task_id]
                remaining = (task.total or 0) - task.completed
                if remaining > 0:
                    self._progress.update(task_id, advance=remaining)
