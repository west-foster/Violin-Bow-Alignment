"""Learn your violin's 3D keypoint layout (and the webcam focal length) from labelled frames.

    python scripts/fit_violin_model.py            # fit + report, writes configs/violin_fit.yaml
    python scripts/fit_violin_model.py --dry-run  # report only

Run it again whenever you label more frames. More variety in violin angle
across frames = better-constrained depth. Delete configs/violin_fit.yaml to go
back to the default dimensions.
"""

from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import yaml

from bowtrack.annotations import AnnotationStore
from bowtrack.config import load_config, resolve
from bowtrack.geometry import CameraIntrinsics, GeometryEstimator, ViolinModel
from bowtrack.keypoints import VIOLIN_KEYPOINTS, Keypoints2D
from bowtrack.model_fit import Observation, fit_violin_model


def angles(cfg_points, cfg, cam, entries):
    model = ViolinModel(cfg_points, cfg["violin_model"]["string_length"], cfg["bow"]["hair_length"])
    out = {}
    for e in entries:
        g = GeometryEstimator(model, cam, max_reproj_px=99)(Keypoints2D.from_dict(e["keypoints"]), (e["width"], e["height"]))
        out[e["file"]] = (g.pose.reproj_error_px if g.pose else np.nan, g.bow.angle_deg if g.bow else np.nan,
                          g.pose.dropped if g.pose else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--no-focal", action="store_true", help="keep the focal length fixed")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-points", type=int, default=5, help="violin points a frame needs to be used")
    args = ap.parse_args()

    cfg = load_config(use_fit=False)  # always start from the standard dimensions
    calib = resolve(cfg["camera"]["calibration_file"])
    store = AnnotationStore(resolve(args.data))
    entries = [e for e in store.entries()
               if sum(e["keypoints"].get(n) is not None for n in VIOLIN_KEYPOINTS) >= args.min_points]
    if len(entries) < 5:
        raise SystemExit(f"Need >= 5 frames with {args.min_points}+ violin points labelled (have {len(entries)}).")
    w, h = entries[0]["width"], entries[0]["height"]
    if calib.exists():
        cam = CameraIntrinsics.load(calib).scaled_to(w, h)
        hfov = float(np.degrees(2 * np.arctan(w / (2 * cam.K[0, 0]))))
        fit_focal = False
        print(f"Using checkerboard calibration ({hfov:.1f} deg FOV); focal length held fixed.")
    else:
        hfov = cfg["camera"]["horizontal_fov_deg"]
        cam = CameraIntrinsics.from_fov(w, h, hfov)
        fit_focal = not args.no_focal

    obs = []
    for e in entries:
        names = [n for n in VIOLIN_KEYPOINTS if e["keypoints"].get(n) is not None]
        obs.append(Observation(e["file"], names, np.array([e["keypoints"][n] for n in names], float)))

    base = ViolinModel.from_config(cfg)
    measured = {k: v for k, v in (cfg.get("violin_measurements") or {}).items() if v is not None}
    if measured:
        print("Using ruler measurements:", ", ".join(f"{k}={v}" for k, v in measured.items()))
    res = fit_violin_model(obs, base.points, base.string_length, (w, h), hfov, fit_focal=fit_focal,
                           measured=measured)

    print(f"\n{len(obs)} frames, {sum(len(o.names) for o in obs)} violin keypoints")
    print(f"reprojection RMS: {res.rms_before:.1f} px (standard model) -> {res.rms_after:.1f} px (fitted)")
    if fit_focal:
        print(f"camera horizontal FOV: {hfov:.1f} deg (guess) -> {res.hfov_deg:.1f} deg (fitted)")
    print("\nper-keypoint RMS after fit (high = inconsistent labels for that point):")
    for n, v in sorted(res.per_keypoint_rms.items(), key=lambda kv: -kv[1]):
        print(f"  {n:22s} {v:5.1f} px")
    print("\nfitted parameters (mm) ± 1-sigma uncertainty (large ± = measure it with a ruler):")
    for n, sd in res.param_std.items():
        flag = "   <- poorly determined" if sd > 5 and n not in measured else ""
        print(f"  {n:26s} {res.param_shift[n]:+7.1f} from standard  ± {sd:4.1f}{flag}")
    print("\nfitted 3D positions (mm, violin frame) vs standard:")
    for n in VIOLIN_KEYPOINTS:
        a, b = res.points[n], base.points[n]
        print(f"  {n:22s} [{a[0]:7.1f} {a[1]:6.1f} {a[2]:6.1f}]   moved {np.linalg.norm(a - b):5.1f} mm")

    cam_fit = CameraIntrinsics.from_fov(w, h, res.hfov_deg) if fit_focal else cam
    before = angles(base.points, cfg, cam, entries)
    after = angles(res.points, cfg, cam_fit, entries)
    print("\nper frame: PnP rms px / bow angle   standard -> fitted")
    for e in sorted(entries, key=lambda e: e["file"]):
        b, a = before[e["file"]], after[e["file"]]
        print(f"  {e['file'][7:]:36s} {b[0]:5.1f}px {b[1]:6.1f}°  ->  {a[0]:5.1f}px {a[1]:6.1f}°"
              + (f"   (ignored: {a[2]})" if a[2] else ""))

    if args.dry_run:
        return
    out = resolve("configs/violin_fit.yaml")
    data = {
        "_comment": f"Fitted by scripts/fit_violin_model.py on {date.today()} from {len(obs)} frames; "
                    f"RMS {res.rms_before:.1f} -> {res.rms_after:.1f} px. Delete to use default dimensions.",
        "violin_model": {n: [round(float(v), 1) for v in res.points[n]] for n in VIOLIN_KEYPOINTS},
    }
    if fit_focal:
        data["camera"] = {"horizontal_fov_deg": round(res.hfov_deg, 2)}
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"\nwrote {out} (picked up automatically by the GUI and training)")


if __name__ == "__main__":
    main()
