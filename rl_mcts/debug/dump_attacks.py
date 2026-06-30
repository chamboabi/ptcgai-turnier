"""Dump all_attack() data to JSON.

Run from rl_mcts/:
    python debug/dump_attacks.py
    python debug/dump_attacks.py --out attacks.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cg.api import all_attack


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="Write JSON to file (default: stdout)")
    args = ap.parse_args()

    attacks = all_attack()
    data = [dataclasses.asdict(a) for a in attacks]
    js = json.dumps(data, indent=2)

    if args.out:
        Path(args.out).write_text(js)
        print(f"Wrote {len(attacks)} attacks to {args.out}")
    else:
        print(js)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
