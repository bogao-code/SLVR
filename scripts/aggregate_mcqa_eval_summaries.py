#!/usr/bin/env python3
"""Print a compact table from parallel MCQA eval dirs (combined_summary.json)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="results root (dirs like checkpoint-500_baseline_all).",
    )
    args = ap.parse_args()

    dirs = sorted(p for p in args.root.glob("*_baseline_all") if p.is_dir())
    if not dirs:
        print(f"No dirs matching *_baseline_all under {args.root}")
        return

    for d in dirs:
        p = d / "combined_summary.json"
        name = d.name.replace("_baseline_all", "")
        if not p.is_file():
            print(f"{name}\t(no combined_summary.json)")
            continue

        summary = {}
        try:
            for entry in json.loads(p.read_text(encoding="utf-8")):
                b = entry.get("bench", "")
                summary[b] = entry
        except json.JSONDecodeError as e:
            print(f"{name}\tINVALID JSON\t{e}")
            continue

        parts = [name]
        if "mmvp" in summary:
            e = summary["mmvp"]
            parts.append(
                f"MMVP ind_acc={float(e.get('individual_acc', 0)):.4f} "
                f"pair_acc={float(e.get('pair_acc', 0)):.4f} n={e.get('n', '')}"
            )
        else:
            parts.append("MMVP —")

        if "vstar" in summary:
            e = summary["vstar"]
            parts.append(f"VSTAR acc={float(e.get('accuracy', 0)):.4f} n={e.get('n', '')}")
        else:
            parts.append("VSTAR —")

        print("\t".join(parts))


if __name__ == "__main__":
    main()
