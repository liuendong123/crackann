from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crackfann.core.dataset import Dataset
from crackfann.core.types import FilteredQuery, RangePredicate


@dataclass(frozen=True)
class Region:
    low: float
    high: float


class WorkloadGenerator:
    def __init__(self, dataset: Dataset, seed: int = 42) -> None:
        self.dataset = dataset
        self.rng = np.random.default_rng(seed)
        values = dataset.attr_values(0)
        self.attr_min = float(values.min())
        self.attr_max = float(values.max())

    def long_tail(
        self,
        n_queries: int,
        zipf_s: float = 1.2,
        hot_regions: int = 8,
        range_width: float = 0.05,
        k: int = 10,
        broad_ratio: float = 0.0,
        phase: str = "long_tail",
    ) -> list[FilteredQuery]:
        regions = self._regions(hot_regions)
        ranks = np.arange(1, hot_regions + 1, dtype=np.float64)
        probs = 1.0 / np.power(ranks, zipf_s)
        probs /= probs.sum()
        queries = []
        for query_id in range(n_queries):
            region = regions[int(self.rng.choice(hot_regions, p=probs))]
            width = range_width * (4.0 if self.rng.random() < broad_ratio else 1.0)
            queries.append(self._query(query_id, region, width, k, phase))
        return queries

    def template_long_tail(
        self,
        n_queries: int,
        zipf_s: float = 1.4,
        templates: int = 8,
        range_width: float = 0.05,
        k: int = 10,
        boundary_jitter: float = 0.0,
        phase: str = "template_long_tail",
    ) -> list[FilteredQuery]:
        regions = self._template_regions(templates, range_width)
        ranks = np.arange(1, templates + 1, dtype=np.float64)
        probs = 1.0 / np.power(ranks, zipf_s)
        probs /= probs.sum()
        queries = []
        for query_id in range(n_queries):
            region = regions[int(self.rng.choice(templates, p=probs))]
            if boundary_jitter > 0:
                span = self.attr_max - self.attr_min
                delta_low = float(self.rng.normal(0.0, boundary_jitter * span))
                delta_high = float(self.rng.normal(0.0, boundary_jitter * span))
                jittered = Region(
                    max(self.attr_min, min(region.low + delta_low, region.high)),
                    min(self.attr_max, max(region.high + delta_high, region.low)),
                )
                region = jittered
            queries.append(self._query_exact_range(query_id, region, k, phase))
        return queries

    def emerging(
        self,
        cold_queries: int,
        ramp_queries: int,
        stable_queries: int,
        range_width: float = 0.05,
        k: int = 10,
    ) -> list[FilteredQuery]:
        regions = self._regions(8)
        hot = regions[-2]
        queries: list[FilteredQuery] = []
        qid = 0
        for _ in range(cold_queries):
            queries.append(self._query(qid, regions[int(self.rng.integers(0, len(regions)))], range_width, k, "cold"))
            qid += 1
        for step in range(ramp_queries):
            hot_prob = (step + 1) / max(ramp_queries, 1)
            region = hot if self.rng.random() < hot_prob else regions[int(self.rng.integers(0, len(regions)))]
            queries.append(self._query(qid, region, range_width, k, "emerging_ramp"))
            qid += 1
        for _ in range(stable_queries):
            queries.append(self._query(qid, hot, range_width, k, "emerging_stable"))
            qid += 1
        return queries

    def drift(self, n_queries: int, range_width: float = 0.05, k: int = 10, abrupt: bool = True) -> list[FilteredQuery]:
        regions = self._regions(8)
        a = regions[1]
        b = regions[-2]
        queries = []
        for query_id in range(n_queries):
            if abrupt:
                region = a if query_id < n_queries // 2 else b
            else:
                prob_b = query_id / max(n_queries - 1, 1)
                region = b if self.rng.random() < prob_b else a
            phase = "drift_b" if region == b else "drift_a"
            queries.append(self._query(query_id, region, range_width, k, phase))
        return queries

    def recurring(self, n_queries: int, period: int = 500, range_width: float = 0.05, k: int = 10) -> list[FilteredQuery]:
        regions = self._regions(8)
        a = regions[2]
        b = regions[5]
        queries = []
        for query_id in range(n_queries):
            region = a if (query_id // max(period, 1)) % 2 == 0 else b
            phase = "recurring_a" if region == a else "recurring_b"
            queries.append(self._query(query_id, region, range_width, k, phase))
        return queries

    def mixed(self, n_queries: int, range_width: float = 0.05, k: int = 10, broad_ratio: float = 0.15) -> list[FilteredQuery]:
        return self.long_tail(
            n_queries=n_queries,
            zipf_s=1.1,
            hot_regions=10,
            range_width=range_width,
            k=k,
            broad_ratio=broad_ratio,
            phase="mixed",
        )

    def _regions(self, count: int) -> list[Region]:
        edges = np.linspace(self.attr_min, self.attr_max, count + 1)
        return [Region(float(edges[i]), float(edges[i + 1])) for i in range(count)]

    def _template_regions(self, count: int, range_width: float) -> list[Region]:
        span = self.attr_max - self.attr_min
        width = span * range_width
        centers = np.linspace(self.attr_min + width / 2.0, self.attr_max - width / 2.0, count)
        regions = []
        for center in centers:
            low = max(self.attr_min, float(center - width / 2.0))
            high = min(self.attr_max, float(center + width / 2.0))
            regions.append(Region(low, high))
        return regions

    def _query(self, query_id: int, region: Region, width: float, k: int, phase: str) -> FilteredQuery:
        center = float(self.rng.uniform(region.low, region.high))
        half = width * (self.attr_max - self.attr_min) / 2.0
        low = max(self.attr_min, center - half)
        high = min(self.attr_max, center + half)
        return self._query_exact_range(query_id, Region(low, high), k, phase)

    def _query_exact_range(self, query_id: int, region: Region, k: int, phase: str) -> FilteredQuery:
        low = region.low
        high = region.high
        mask = (self.dataset.attr_values(0) >= low) & (self.dataset.attr_values(0) <= high)
        ids = self.dataset.ids[mask]
        if ids.size:
            anchor = int(self.rng.choice(ids))
        else:
            anchor = int(self.rng.integers(0, self.dataset.n))
        vector = self.dataset.vectors[anchor] + self.rng.normal(0.0, 0.05, size=self.dataset.d).astype(np.float32)
        return FilteredQuery(
            query_id=query_id,
            vector=vector.astype(np.float32, copy=False),
            predicates=(RangePredicate(attr_id=0, low=low, high=high),),
            k=k,
            timestamp=query_id,
            phase=phase,
        )
