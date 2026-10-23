"""Reciprocal Rank Fusion shared by the retrieval and experiment paths."""

from collections import defaultdict
from typing import Iterable, Optional, Sequence, Tuple


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[str]], k0: int = 60, limit: Optional[int] = None
) -> list[Tuple[str, float]]:
    scores = defaultdict(float)
    first_seen = {}
    for ranking in rankings:
        seen = set()
        for rank, item in enumerate(ranking, 1):
            item = str(item)
            if item in seen:
                continue
            seen.add(item)
            first_seen.setdefault(item, len(first_seen))
            scores[item] += 1.0 / (k0 + rank)
    fused = sorted(scores.items(), key=lambda pair: (-pair[1], first_seen[pair[0]]))
    return fused[:limit] if limit is not None else fused
