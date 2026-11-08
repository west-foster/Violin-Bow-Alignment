"""Offline check that the estimated violin pose matches the real violin outline.

    python scripts/validate_pose.py             # score every labelled frame
    python scripts/validate_pose.py --compare   # also score the standard (unfitted) model

Writes overlays to data/validation/ (green = outline on an image edge,
yellow = near, red = far) plus _contact_sheet.jpg with every frame.
"""

from __future__ import annotations

import argparse

from bowtrack.annotations import AnnotationStore
from bowtrack.config import load_config, resolve
from bowtrack.validation import summarize, validate_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data/validation")
    ap.add_argument("--compare", action="store_true", help="also score the standard violin model")
    ap.add_argument("--max-median", type=float, default=4.0)
    ap.add_argument("--min-inliers", type=float, default=0.5)
    args = ap.parse_args()

    store = AnnotationStore(resolve(args.data))
    cfg = load_config()
    res = validate_dataset(store, cfg, resolve(args.out), args.max_median, args.min_inliers)
    std = None
    if args.compare and cfg["_fitted"]:
        std = {f: s for f, s, _ in validate_dataset(store, load_config(use_fit=False), None,
                                                       args.max_median, args.min_inliers)}

    print(f"{'frame':36s} {'median':>7s} {'inliers':>8s} {'best shift':>11s}" + ("   standard model" if std else ""))
    for f, s, reason in res:
        if s is None:
            print(f"{f[7:]:36s}  -- {reason}")
            continue
        line = (f"{f[7:]:36s} {s.median_px:6.1f}px {s.inlier_frac * 100:7.0f}% "
                f"({s.best_shift[0]:+3.0f},{s.best_shift[1]:+3.0f})px")
        if std and std.get(f) is not None:
            o = std[f]
            line += f"   {o.median_px:5.1f}px {o.inlier_frac * 100:4.0f}%"
        print(line + (f"   <- {reason}" if reason else ""))
    print()
    print(summarize(res, "fitted model:   " if cfg["_fitted"] else "model:          "))
    if std:
        print(summarize([(f, s, "") for f, s in std.items()], "standard model: "))
    print(f"\noverlays + contact sheet: {resolve(args.out)}")


if __name__ == "__main__":
    main()
