from __future__ import annotations

from typing import Any


def rrf_fuse(results: list[list[dict[str, Any]]], k: int = 60) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    best: dict[str, dict[str, Any]] = {}
    for ranked_list in results:
        for rank, item in enumerate(ranked_list, start=1):
            item_id = str(item["id"])
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            if item_id not in best:
                best[item_id] = item
    fused: list[dict[str, Any]] = []
    for item_id, score in scores.items():
        item = dict(best[item_id])
        item["rrf_score"] = score
        fused.append(item)
    return sorted(fused, key=lambda x: x["rrf_score"], reverse=True)
