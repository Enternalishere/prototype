from __future__ import annotations

from typing import List


def merge_boxes(boxes: List[dict], gap_px: int = 5) -> List[dict]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda item: (item["y"], item["x"]))
    merged = []
    current = dict(boxes[0])
    for box in boxes[1:]:
        if box["x"] - (current["x"] + current["w"]) <= gap_px and abs(box["y"] - current["y"]) <= max(20, current["h"] // 2):
            current["w"] = max(current["w"], box["x"] + box["w"] - current["x"])
            current["h"] = max(current["h"], box["y"] + box["h"] - current["y"])
            current["area"] += box["area"]
            current["x"] = min(current["x"], box["x"])
            current["y"] = min(current["y"], box["y"])
        else:
            merged.append(current)
            current = dict(box)
    merged.append(current)
    return merged
