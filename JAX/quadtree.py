from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class QuadtreeSpec:
    depth: int
    splits: Tuple[bool, ...]

    def leaf_blocks(self, height: int, width: int) -> List[Tuple[int, int, int]]:
        blocks: List[Tuple[int, int, int]] = []
        it = iter(self.splits)

        def visit(y: int, x: int, size: int, level: int) -> None:
            if level >= self.depth:
                blocks.append((y, x, size))
                return
            try:
                split = next(it)
            except StopIteration as exc:
                raise ValueError("Not enough split flags for quadtree traversal.") from exc
            if not split:
                blocks.append((y, x, size))
                return
            half = size // 2
            visit(y, x, half, level + 1)
            visit(y, x + half, half, level + 1)
            visit(y + half, x, half, level + 1)
            visit(y + half, x + half, half, level + 1)

        if height != width:
            raise ValueError("QuadtreeSpec currently assumes square frames.")
        size = height
        visit(0, 0, size, 0)
        return blocks


def count_quadtree_nodes(depth: int, splits: Iterable[bool]) -> tuple[int, int]:
    internal = 0
    leaf_nonmax = 0
    it = iter(splits)

    def visit(level: int) -> None:
        nonlocal internal, leaf_nonmax
        if level >= depth:
            return
        try:
            split = next(it)
        except StopIteration as exc:
            raise ValueError("Not enough split flags for quadtree traversal.") from exc
        if split:
            internal += 1
            visit(level + 1)
            visit(level + 1)
            visit(level + 1)
            visit(level + 1)
        else:
            leaf_nonmax += 1

    visit(0)
    return internal, leaf_nonmax


def uniform_quadtree(depth: int) -> QuadtreeSpec:
    def build(level: int) -> List[bool]:
        if level >= depth:
            return []
        return [True] + build(level + 1) + build(level + 1) + build(level + 1) + build(level + 1)

    splits = tuple(build(0))
    return QuadtreeSpec(depth=depth, splits=splits)
