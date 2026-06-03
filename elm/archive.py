"""Dict-based MAP-Elites archive over *code* genomes.

Genome = a Python source string (a closed-loop placement policy). Cells are
indexed by binning the 2-D behavior measures, using the SAME ranges
([0,1] x [0,1]) and default 20x20 resolution as the pyribs GridArchive in
qd_train.py, so ELM and CMA-ME archives are directly comparable.

Deliberately NOT pyribs: pyribs archives store fixed-length float solution
vectors. Here the solution is variable-length text, so we keep our own map.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass
class Elite:
    source: str
    fitness: float
    measures: tuple            # (m0, m1)
    city_pop: int = 0
    parents: tuple = ()        # ids of parent elite(s) that produced this one
    origin: str = "seed"       # "seed" | "mutate" | "crossover"
    iteration: int = -1        # driver iteration it entered the archive
    eid: int = -1              # unique id
    render: str | None = None  # ascii city; in-memory only (not persisted)


class MapElitesArchive:
    def __init__(self, dims=(20, 20), ranges=((0.0, 1.0), (0.0, 1.0))):
        self.dims = tuple(dims)
        self.ranges = tuple(tuple(r) for r in ranges)
        self.cells: dict[tuple, Elite] = {}
        self._next_id = 0
        # Running history of every successful insertion attempt's best fitness,
        # for plotting QD-score / coverage progress later.
        self.history: list[dict] = []

    # ---- binning ----
    def cell_of(self, measures) -> tuple:
        idx = []
        for v, (lo, hi), d in zip(measures, self.ranges, self.dims):
            t = 0.0 if hi == lo else (v - lo) / (hi - lo)
            b = int(np.clip(int(t * d), 0, d - 1))
            idx.append(b)
        return tuple(idx)

    # ---- insertion ----
    def add(self, source: str, fitness: float, measures, *, city_pop=0,
            parents=(), origin="mutate", iteration=-1, render=None) -> tuple[bool, tuple]:
        """Insert if the cell is empty or we beat its occupant.

        Returns (improved, cell). `improved` is True iff this genome now owns
        its cell (new cell or strictly higher fitness).
        """
        cell = self.cell_of(measures)
        cur = self.cells.get(cell)
        improved = cur is None or fitness > cur.fitness
        if improved:
            e = Elite(
                source=source, fitness=float(fitness),
                measures=(float(measures[0]), float(measures[1])),
                city_pop=int(city_pop), parents=tuple(parents),
                origin=origin, iteration=iteration, eid=self._next_id,
                render=render,
            )
            self._next_id += 1
            self.cells[cell] = e
        return improved, cell

    # ---- sampling ----
    def sample(self, rng, k=1, weighted=False):
        """Return up to k distinct elites. weighted=True biases toward higher
        fitness (softmax); otherwise uniform over filled cells (standard
        MAP-Elites curiosity)."""
        elites = list(self.cells.values())
        if not elites:
            return []
        k = min(k, len(elites))
        if weighted:
            f = np.array([e.fitness for e in elites], dtype=np.float64)
            f = f - f.max()
            p = np.exp(f / (np.std(f) + 1e-9))
            p /= p.sum()
            idx = rng.choice(len(elites), size=k, replace=False, p=p)
        else:
            idx = rng.choice(len(elites), size=k, replace=False)
        return [elites[i] for i in np.atleast_1d(idx)]

    # ---- metrics ----
    @property
    def coverage(self) -> float:
        return len(self.cells) / float(np.prod(self.dims))

    @property
    def qd_score(self) -> float:
        # Sum of (non-negative) fitness over filled cells — the standard QD
        # score. Clamp negatives so shaped/penalty fitnesses don't subtract.
        return float(sum(max(0.0, e.fitness) for e in self.cells.values()))

    @property
    def best(self) -> Elite | None:
        return max(self.cells.values(), key=lambda e: e.fitness, default=None)

    def record_history(self, iteration: int):
        b = self.best
        self.history.append(dict(
            iteration=iteration, filled=len(self.cells),
            coverage=self.coverage, qd_score=self.qd_score,
            best_fitness=(b.fitness if b else 0.0),
            best_city_pop=(b.city_pop if b else 0),
        ))

    # ---- persistence ----
    def save(self, path: str):
        elites = list(self.cells.values())
        np.savez(
            path,
            dims=np.array(self.dims),
            ranges=np.array(self.ranges),
            cells=np.array([list(c) for c in self.cells.keys()], dtype=np.int32)
                  if elites else np.zeros((0, 2), np.int32),
            sources=np.array([e.source for e in elites], dtype=object),
            fitness=np.array([e.fitness for e in elites], dtype=np.float64),
            measures=np.array([e.measures for e in elites], dtype=np.float64)
                     if elites else np.zeros((0, 2)),
            city_pop=np.array([e.city_pop for e in elites], dtype=np.int64),
            origin=np.array([e.origin for e in elites], dtype=object),
            iteration=np.array([e.iteration for e in elites], dtype=np.int64),
            history=np.array(self.history, dtype=object),
        )

    @classmethod
    def load(cls, path: str) -> "MapElitesArchive":
        d = np.load(path, allow_pickle=True)
        arch = cls(dims=tuple(d["dims"]), ranges=tuple(map(tuple, d["ranges"])))
        for cell, src, fit, meas, cp, org, it in zip(
            d["cells"], d["sources"], d["fitness"], d["measures"],
            d["city_pop"], d["origin"], d["iteration"],
        ):
            arch.cells[tuple(int(c) for c in cell)] = Elite(
                source=str(src), fitness=float(fit),
                measures=(float(meas[0]), float(meas[1])), city_pop=int(cp),
                origin=str(org), iteration=int(it), eid=arch._next_id,
            )
            arch._next_id += 1
        arch.history = list(d["history"]) if "history" in d else []
        return arch

    def summary(self) -> str:
        b = self.best
        lines = [
            f"filled={len(self.cells)}/{int(np.prod(self.dims))} "
            f"coverage={self.coverage:.3f} qd_score={self.qd_score:.1f}",
        ]
        if b is not None:
            lines.append(f"best: fitness={b.fitness:.1f} cityPop={b.city_pop} "
                         f"measures=({b.measures[0]:.3f},{b.measures[1]:.3f}) "
                         f"origin={b.origin}")
        return "\n".join(lines)
