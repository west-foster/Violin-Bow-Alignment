"""Offline pose check: does the projected violin outline match the real violin's edge?

Used during data preparation / before training only. It is far too slow and
fragile for the live loop, but it is a useful independent check: the pose
comes from the clicked keypoints alone, while this score comes from image
edges the pose never saw.

Score per frame:
  * median_px    median distance from outline samples to the nearest image edge
  * inlier_frac  fraction of samples within `inlier_px` of an edge
  * best_shift   2D shift (px) that would best align the outline: a large
                 consistent shift means the pose/model is off, not just noisy
Parts of the outline hidden by the jaw, hand or bow land far from any edge;
the median and the inlier fraction are robust to that.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import CameraIntrinsics, ViolinModel, ViolinPose


@dataclass
class ContourScore:
    median_px: float
    inlier_frac: float
    best_shift: tuple[float, float]
    shifted_median_px: float
    samples: np.ndarray  # (N, 2) outline sample pixels
    dists: np.ndarray  # (N,) distance to nearest edge, capped

    def ok(self, max_median: float = 4.0, min_inliers: float = 0.5) -> bool:
        return self.median_px <= max_median and self.inlier_frac >= min_inliers


def _resample(poly: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    t = np.r_[0, np.cumsum(seg)]
    s = np.arange(0, t[-1], step)
    return np.c_[np.interp(s, t, poly[:, 0]), np.interp(s, t, poly[:, 1])]


N_ORIENT = 8  # edge orientation bins over 0-180 deg (22.5 deg each)


def edge_distance_maps(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Per orientation bin: distance (px) to the nearest Canny edge *running in that direction*.

    Matching only edges parallel to the outline stops strings, bow hair and
    clothing texture that merely cross the outline from counting as a match.
    Returns (N_ORIENT, roi_h, roi_w).
    """
    x0, y0, x1, y1 = roi
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    med = float(np.median(gray))
    edges = cv2.Canny(gray, int(max(10, 0.33 * med)), int(min(255, 0.66 * med + 20))) > 0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    # Edge direction is perpendicular to the gradient.
    ang = (np.degrees(np.arctan2(gy, gx)) + 90.0) % 180.0
    bins = (ang / (180.0 / N_ORIENT)).astype(int) % N_ORIENT
    maps = np.empty((N_ORIENT, y1 - y0, x1 - x0), np.float32)
    for b in range(N_ORIENT):
        mask = edges & (bins == b)
        maps[b] = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3) if mask.any() else 1e3
    return maps


def contour_score(image: np.ndarray, pose: ViolinPose, cam: CameraIntrinsics, model: ViolinModel,
                  step_px: float = 3.0, max_dist: float = 25.0, inlier_px: float = 4.0,
                  search_px: int = 15) -> ContourScore | None:
    h, w = image.shape[:2]
    outline = cam.project(pose.to_camera(model.parts()["body"]))
    samples = _resample(outline, step_px)
    # Outline direction at each sample (closed curve), as an orientation bin.
    tangent = np.roll(samples, -1, axis=0) - np.roll(samples, 1, axis=0)
    t_bin = ((np.degrees(np.arctan2(tangent[:, 1], tangent[:, 0])) % 180.0) / (180.0 / N_ORIENT)).astype(int)
    inside = (samples[:, 0] >= 0) & (samples[:, 0] < w) & (samples[:, 1] >= 0) & (samples[:, 1] < h)
    samples, t_bin = samples[inside], t_bin[inside] % N_ORIENT
    if len(samples) < 20:
        return None
    pad = search_px + 30
    x0, y0 = np.maximum(samples.min(0).astype(int) - pad, 0)
    x1, y1 = np.minimum(samples.max(0).astype(int) + pad, [w, h])
    maps = edge_distance_maps(image, (x0, y0, x1, y1))
    rh, rw = maps.shape[1:]
    neighbours = [(t_bin + k) % N_ORIENT for k in (-1, 0, 1)]  # +-1 bin tolerance (~34 deg)

    def lookup(pts):
        xi = np.round(pts[:, 0]).astype(int) - x0
        yi = np.round(pts[:, 1]).astype(int) - y0
        ok = (xi >= 0) & (xi < rw) & (yi >= 0) & (yi < rh)
        xi, yi = np.clip(xi, 0, rw - 1), np.clip(yi, 0, rh - 1)
        d = np.min([maps[b, yi, xi] for b in neighbours], axis=0)
        return np.where(ok, np.minimum(d, max_dist), max_dist)

    d = lookup(samples)
    # Coarse-to-fine search for the 2D shift with the lowest robust cost.
    best, best_cost = (0, 0), float(np.mean(d))
    for step in (3, 1):
        cx, cy = best
        rng = range(-search_px, search_px + 1, step) if step == 3 else range(-2, 3)
        for dx in rng:
            for dy in rng:
                sx, sy = (dx, dy) if step == 3 else (cx + dx, cy + dy)
                cost = float(np.mean(lookup(samples + [sx, sy])))
                if cost < best_cost:
                    best, best_cost = (sx, sy), cost
    shifted = lookup(samples + np.array(best, float))
    return ContourScore(float(np.median(d)), float(np.mean(d <= inlier_px)), (float(best[0]), float(best[1])),
                        float(np.median(shifted)), samples, d)


def draw_contour_check(image: np.ndarray, score: ContourScore, inlier_px: float = 4.0) -> np.ndarray:
    out = image.copy()
    for (x, y), d in zip(score.samples, score.dists):
        col = (60, 200, 60) if d <= inlier_px else ((0, 200, 255) if d <= 10 else (40, 40, 230))
        cv2.circle(out, (int(x), int(y)), 2, col, -1, cv2.LINE_AA)
    txt = (f"median {score.median_px:.1f}px  inliers {score.inlier_frac * 100:.0f}%  "
           f"best shift ({score.best_shift[0]:+.0f},{score.best_shift[1]:+.0f})px")
    cv2.putText(out, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def validate_dataset(store, cfg: dict, out_dir=None, max_median: float = 4.0, min_inliers: float = 0.5):
    """Score every labelled frame. Returns [(file, ContourScore | None, reason)].

    Writes per-frame overlays plus a contact sheet to `out_dir` if given.
    """
    from pathlib import Path

    from .geometry import GeometryEstimator
    from .keypoints import Keypoints2D
    from .overlay import draw_keypoints

    model = ViolinModel.from_config(cfg)
    results, thumbs = [], []
    for e in sorted(store.entries(), key=lambda e: e["file"]):
        img = cv2.imread(str(Path(store.root) / e["file"]))
        if img is None:
            results.append((e["file"], None, "image missing"))
            continue
        h, w = img.shape[:2]
        cam = _camera_for(cfg, w, h)
        kp = Keypoints2D.from_dict(e["keypoints"])
        pose = GeometryEstimator(model, cam, max_reproj_px=99).estimate_pose(kp)
        if pose is None:
            results.append((e["file"], None, "fewer than 4 violin points"))
            continue
        score = contour_score(img, pose, cam, model)
        if score is None:
            results.append((e["file"], None, "outline off-screen"))
            continue
        reason = "" if score.ok(max_median, min_inliers) else "outline does not match violin edge"
        results.append((e["file"], score, reason))
        if out_dir is not None:
            vis = draw_contour_check(img, score)
            draw_keypoints(vis, kp, 0.5, labels=False)
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / Path(e["file"]).name), vis)
            x0, y0 = np.maximum(score.samples.min(0).astype(int) - 60, 0)
            x1, y1 = np.minimum(score.samples.max(0).astype(int) + 60, [w, h])
            crop = cv2.resize(vis[y0:y1, x0:x1], (400, 260))
            cv2.putText(crop, Path(e["file"]).stem[-12:], (6, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        1, cv2.LINE_AA)
            thumbs.append(crop)
    if out_dir is not None and thumbs:
        cols = 4
        while len(thumbs) % cols:
            thumbs.append(np.zeros_like(thumbs[0]))
        rows = [np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)]
        cv2.imwrite(str(Path(out_dir) / "_contact_sheet.jpg"), np.vstack(rows))
    return results


def _camera_for(cfg: dict, w: int, h: int) -> CameraIntrinsics:
    from .config import resolve

    calib = resolve(cfg["camera"]["calibration_file"])
    if calib.exists():
        return CameraIntrinsics.load(calib).scaled_to(w, h)
    return CameraIntrinsics.from_fov(w, h, cfg["camera"]["horizontal_fov_deg"])


def summarize(results, label: str = "") -> str:
    scored = [s for _, s, _ in results if s is not None]
    if not scored:
        return f"{label}no frames could be scored"
    med = np.median([s.median_px for s in scored])
    inl = np.median([s.inlier_frac for s in scored])
    shift = np.median([np.hypot(*s.best_shift) for s in scored])
    bad = sum(1 for _, s, r in results if r)
    return (f"{label}median edge distance {med:.1f}px, inliers {inl * 100:.0f}%, typical misalignment "
            f"{shift:.0f}px; {bad}/{len(results)} frames flagged")
