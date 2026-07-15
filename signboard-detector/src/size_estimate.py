from __future__ import annotations

from typing import List, Dict


def estimate_piece_sizes(boxes: List[dict], board_width_inches: float, board_box: dict, image_shape: tuple) -> List[dict]:
    width_px = max(board_box.get("w", 0), 1)
    if width_px <= 1:
        width_px = max(image_shape[1], 1)
    px_per_inch = width_px / float(board_width_inches)
    estimates = []
    for idx, box in enumerate(boxes, start=1):
        w_in = box["w"] / px_per_inch
        h_in = box["h"] / px_per_inch
        estimates.append(
            {
                "id": idx,
                "bbox": [int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])],
                "width_in": round(w_in, 2),
                "height_in": round(h_in, 2),
            }
        )
    return estimates
