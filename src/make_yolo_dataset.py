from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DETECTOR_ROOT = ROOT / 'signboard-detector'
if str(DETECTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(DETECTOR_ROOT))

from src.make_yolo_dataset import main


def resolve_path(arg: str) -> str:
    p = Path(arg)
    if p.is_absolute() or p.exists():
        return arg
    candidate = DETECTOR_ROOT / arg
    if candidate.exists():
        return str(candidate)
    return arg


def main_wrapper() -> None:
    new_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ('--root', '--labels', '--output') and i + 1 < len(sys.argv):
            new_argv.append(arg)
            new_argv.append(resolve_path(sys.argv[i + 1]))
            i += 2
        else:
            new_argv.append(arg)
            i += 1
    sys.argv = new_argv
    main()


if __name__ == '__main__':
    main_wrapper()
