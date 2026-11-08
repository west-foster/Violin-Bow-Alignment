"""Checkerboard calibration of the webcam (improves 3D accuracy).

Print a checkerboard (e.g. https://markhedleyjones.com/projects/calibration-checkerboard-collection),
measure one square, then:

    python scripts/calibrate_camera.py --cols 9 --rows 6 --square-mm 25

SPACE captures a view when the board is detected (vary angle/distance; ~15 views),
C calibrates and saves configs/camera_calibration.yaml, Q quits.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from bowtrack.camera import fetch_camera
from bowtrack.config import load_config, resolve
from bowtrack.geometry import CameraIntrinsics


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=9, help="inner corners per row")
    ap.add_argument("--rows", type=int, default=6, help="inner corners per column")
    ap.add_argument("--square-mm", type=float, default=25.0)
    args = ap.parse_args()

    pattern = (args.cols, args.rows)
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0 : args.cols, 0 : args.rows].T.reshape(-1, 2) * args.square_mm
    obj_pts, img_pts, size = [], [], None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    c = cfg["camera"]
    with fetch_camera(c["index"], c["width"], c["height"], c["fps"]) as cam:
        while True:
            ok, frame = cam.read(wait_new=True)
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            found, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_FAST_CHECK)
            vis = frame.copy()
            if found:
                cv2.drawChessboardCorners(vis, pattern, corners, found)
            cv2.putText(vis, f"views: {len(obj_pts)}  [SPACE] capture  [C] calibrate  [Q] quit", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if found else (0, 0, 255), 2)
            cv2.imshow("calibrate", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and found:
                obj_pts.append(objp)
                img_pts.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit))
            elif key == ord("c") and len(obj_pts) >= 8:
                break
            elif key == ord("q"):
                return
    cv2.destroyAllWindows()
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    out = resolve(c["calibration_file"])
    CameraIntrinsics(K, dist.ravel(), size[0], size[1]).save(out)
    hfov = np.degrees(2 * np.arctan(size[0] / (2 * K[0, 0])))
    print(f"RMS reprojection {rms:.3f}px, fx={K[0, 0]:.1f}, horizontal FOV {hfov:.1f} deg -> {out}")


if __name__ == "__main__":
    main()
