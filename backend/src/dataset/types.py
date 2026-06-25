"""Domain types for ARC-AGI tasks.

An :class:`Example` is one input/output grid pair. A :class:`Task` groups the
train / test / arc-gen example subsets under a ``taskNNN`` id. Both are frozen;
grids are stored as tuples so instances stay immutable.
"""

from __future__ import annotations

from dataclasses import dataclass

from dataset.encoding import Grid


@dataclass(frozen=True)
class Example:
    """One ARC-AGI input/output grid pair."""

    input: Grid
    output: Grid


@dataclass(frozen=True)
class Task:
    """An ARC-AGI task: train / test / arc-gen example subsets plus its id."""

    task_id: int
    train: tuple[Example, ...]
    test: tuple[Example, ...]
    arc_gen: tuple[Example, ...]

    def all_examples(self) -> tuple[Example, ...]:
        """Every example pair across train, test and arc-gen."""
        return self.train + self.test + self.arc_gen

    def as_scorer_dict(self) -> dict[str, list[dict[str, Grid]]]:
        """Render as the ``{"train"/"test"/"arc-gen": [{"input","output"}]}``
        dict the official-scorer mirror (:func:`evaluate.audit_one`) consumes."""
        return {
            "train": [{"input": e.input, "output": e.output} for e in self.train],
            "test": [{"input": e.input, "output": e.output} for e in self.test],
            "arc-gen": [{"input": e.input, "output": e.output} for e in self.arc_gen],
        }
