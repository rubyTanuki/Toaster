import threading
from typing import Dict, Optional, Set
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

class ProgressTracker:
    _DESCRIPTIONS = {
        'resolve': "[cyan]Resolving Dependencies:",
        'describe': "[magenta]Describing: ",
        'embed': "[green]Embedding:  ",
    }

    _GROUPS = {
        'resolve': 'resolve',
        'describe': 'describe_embed',
        'embed': 'describe_embed',
    }

    _PHASE_MESSAGES = {
        ("ast", "start"): "parsing AST...",
        ("ast", "end"): "✅ Finished parsing AST",
        ("resolve", "end"): "✅ Finished resolving dependencies",
    }

    def __init__(self, console: Console, include_resolve: bool = True, include_describe: bool = True, include_embed: bool = True):
        self._console = console
        self._lock = threading.Lock()

        enabled = {
            'resolve': include_resolve,
            'describe': include_describe,
            'embed': include_embed,
        }
        self._enabled = {t for t, on in enabled.items() if on}

        # One rich.Progress (and its Live) per phase group, created lazily on first advance().
        self._group_progress: Dict[str, Progress] = {}
        self._closed_groups: Set[str] = set()
        # task_type -> task id within its group's Progress.
        self._tasks: Dict[str, int] = {}
        # Totals accumulated via enqueue() before a task exists; seeded into the task once it's
        # finally created on first advance().
        self._pending_totals: Dict[str, int] = {}

    def _make_progress(self) -> Progress:
        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self._console,
        )

    def enqueue(self, task_type: str, amount: int = 1):
        """Increments the total for a specific task type."""
        with self._lock:
            if task_type not in self._enabled:
                return
            if task_type in self._tasks:
                progress = self._group_progress[self._GROUPS[task_type]]
                task_id = self._tasks[task_type]
                current_total = progress.tasks[task_id].total or 0
                progress.update(task_id, total=current_total + amount)
            else:
                self._pending_totals[task_type] = self._pending_totals.get(task_type, 0) + amount

    def advance(self, task_type: str, amount: int = 1):
        """Advances the completion counter for a specific task type."""
        with self._lock:
            if task_type not in self._enabled:
                return
            group = self._GROUPS[task_type]
            if task_type not in self._tasks:
                progress = self._group_progress.get(group)
                if progress is None:
                    progress = self._make_progress()
                    progress.start()
                    self._group_progress[group] = progress
                total = self._pending_totals.pop(task_type, 0)
                self._tasks[task_type] = progress.add_task(self._DESCRIPTIONS[task_type], total=total)
            self._group_progress[group].update(self._tasks[task_type], advance=amount)

    def _close_group(self, group: str) -> None:
        """Forces any incomplete bars in `group` to 100% and stops its Live, if it was ever
        started. Must be called with `self._lock` already held."""
        progress = self._group_progress.get(group)
        if progress is None or group in self._closed_groups:
            return
        for task_type, task_id in self._tasks.items():
            if self._GROUPS[task_type] != group:
                continue
            task = progress.tasks[task_id]
            remaining = (task.total or 0) - task.completed
            if remaining > 0:
                progress.update(task_id, advance=remaining)
        progress.stop()
        self._closed_groups.add(group)

    def finish(self):
        """Force-completes and stops any phase groups still open (normally just describe/embed;
        resolve is already closed by phase_end('resolve'))."""
        with self._lock:
            for group in list(self._group_progress):
                self._close_group(group)

    def phase_start(self, phase: str) -> None:
        """Prints a phase-boundary announcement, if one is defined for this phase."""
        msg = self._PHASE_MESSAGES.get((phase, "start"))
        if msg:
            self._console.print(msg)

    def phase_end(self, phase: str) -> None:
        """Closes this phase's bar group (if any), then prints its completion announcement.
        Closing the group first means the announcement always prints after any Live for this
        phase has stopped, so it can never race a live redraw or show stale bar state."""
        with self._lock:
            group = self._GROUPS.get(phase)
            if group:
                self._close_group(group)
        msg = self._PHASE_MESSAGES.get((phase, "end"))
        if msg:
            self._console.print(msg)
