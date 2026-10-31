"""
Unique-Sink Orientation (USO) on hypercubes.

FIX: RandomUSO now uses the canonical orientation (every vertex -> origin)
plus a random hypercube automorphism (coordinate permutation + bit flips).
Automorphisms map USOs to USOs, so the result is guaranteed valid.
"""

import numpy as np
from itertools import product
from typing import Dict, Tuple, Optional


class HypercubeUSO:
    """An orientation of the d-dimensional hypercube."""

    def __init__(self, dim: int, orientation: Dict[Tuple[int, ...], list]):
        self.dim = dim
        self.orientation = orientation
        self._validate()

    def _validate(self):
        expected = set(product([0, 1], repeat=self.dim))
        if set(self.orientation.keys()) != expected:
            raise ValueError("Orientation does not cover all vertices.")
        for v, neighbors in self.orientation.items():
            for u in neighbors:
                if v == u:
                    raise ValueError("Self-loop detected.")
                diff = sum(a != b for a, b in zip(v, u))
                if diff != 1:
                    raise ValueError(
                        f"Edge between {v} and {u} is not a hypercube edge."
                    )

    def is_sink(self, vertex: Tuple[int, ...]) -> bool:
        return len(self.orientation[vertex]) == 0

    def find_sink_bruteforce(self) -> Tuple[int, ...]:
        sinks = [v for v in self.orientation if self.is_sink(v)]
        if len(sinks) != 1:
            raise RuntimeError(
                f"Expected exactly one sink, found {len(sinks)}."
            )
        return sinks[0]

    def find_sink_walk(
        self, start: Optional[Tuple[int, ...]] = None
    ) -> Tuple[int, ...]:
        if start is None:
            start = tuple([0] * self.dim)
        current = start
        visited = set()
        while not self.is_sink(current):
            if current in visited:
                raise RuntimeError(
                    "Cycle detected: orientation is not a valid USO."
                )
            visited.add(current)
            current = sorted(self.orientation[current])[0]
        return current

    def __repr__(self) -> str:
        return (
            f"HypercubeUSO(dim={self.dim}, "
            f"vertices={len(self.orientation)})"
        )


class RandomUSO(HypercubeUSO):
    """A random USO obtained by applying a hypercube automorphism
    to the canonical orientation (every vertex -> origin)."""

    def __init__(self, dim: int, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(dim)
        flips = rng.integers(0, 2, size=dim).astype(int)

        def transform(v):
            return tuple(
                int(v[perm[i]]) ^ int(flips[i]) for i in range(dim)
            )

        canonical = {}
        for v in product([0, 1], repeat=dim):
            out = []
            for i in range(dim):
                if v[i] == 1:
                    u = list(v)
                    u[i] = 0
                    out.append(tuple(u))
            canonical[v] = out

        orientation = {}
        for v, outs in canonical.items():
            tv = transform(v)
            orientation[tv] = [transform(u) for u in outs]

        super().__init__(dim, orientation)
