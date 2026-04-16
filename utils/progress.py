from contextlib import contextmanager

from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)


class PipelineProgress:
    """Multi-stage progress tracker using Rich."""

    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )
        self._task_ids = {}

    def add_stage(self, stage_name: str, total: int) -> None:
        task_id = self.progress.add_task(stage_name, total=total)
        self._task_ids[stage_name] = task_id

    def advance(self, stage_name: str, amount: int = 1) -> None:
        if stage_name in self._task_ids:
            self.progress.advance(self._task_ids[stage_name], amount)

    def complete_stage(self, stage_name: str) -> None:
        if stage_name in self._task_ids:
            task = self.progress.tasks[self._task_ids[stage_name]]
            self.progress.update(self._task_ids[stage_name], completed=task.total)

    @contextmanager
    def live(self):
        with self.progress:
            yield self
