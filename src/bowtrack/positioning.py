"""Is the violin placed so the bow angle can be measured well? If not, say how to move.

One camera cannot see depth, so how well the bow angle is determined depends
on the viewpoint. The quantity that matters is how much the angle would change
if we assumed the wrong string's bowing plane (the bridge is arched, so the
bow is tilted by ~+-20 deg between G and E): seen from a good angle the four
planes give almost the same bow angle, seen from a bad one they differ by 15+
deg. The advisor computes this error for the current pose and for the player
turning their body by +-60 deg, and suggests the smallest turn that fixes it.
It also checks the violin is fully in frame and large enough.

Directions are given from the PLAYER's point of view (they face the camera),
so the player's left is the image's right (frames are not mirrored).
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from .geometry import STRINGS, CameraIntrinsics, ViolinModel, ViolinPose

TEST_ANGLES_DEG = (85.0, 90.0, 95.0)  # bow angles near the green/red boundary
TURNS_DEG = np.arange(-60.0, 60.1, 2.0)  # + = player turns to their left
RAISES_DEG = np.arange(0.0, 40.1, 5.0)  # camera raised to look down onto the violin by this much


@dataclass
class PositionAdvice:
    status: str  # "ok" | "adjust" (move/turn) | "limited" (no turn helps) | "unknown" (no violin)
    text: str
    string_error_deg: float | None = None  # worst bow-angle error if the string is unknown, current view
    turn_deg: float | None = None  # suggested body turn, + = to the player's left
    best_error_deg: float | None = None  # error after that turn
    raise_deg: float | None = None  # if turning isn't enough: look down onto the violin by this much more
    raise_error_deg: float | None = None  # error after raising the camera (and turning)
    view_deg: float | None = None
    framing: str | None = None  # "back" | "closer" | "left" | "right" | "up" | "down" | None
    px_per_mm: float | None = None


def _rot_x(psi: np.ndarray) -> np.ndarray:
    """Rotations about the camera's horizontal axis: +psi = camera raised, looking down more."""
    c, s = np.cos(psi), np.sin(psi)
    R = np.zeros((len(psi), 3, 3))
    R[:, 0, 0], R[:, 1, 1], R[:, 1, 2], R[:, 2, 1], R[:, 2, 2] = 1.0, c, -s, s, c
    return R


def _rot_y(phi: np.ndarray) -> np.ndarray:
    """Rotations about the camera's vertical (image-down) axis, shape (N, 3, 3)."""
    c, s = np.cos(phi), np.sin(phi)
    R = np.zeros((len(phi), 3, 3))
    R[:, 0, 0], R[:, 0, 2], R[:, 1, 1], R[:, 2, 0], R[:, 2, 2] = c, s, 1.0, -s, c
    return R


def string_error_deg(R: np.ndarray, t: np.ndarray, model: ViolinModel, contact_mm: float = 30.0) -> np.ndarray:
    """Worst bow-angle error from assuming the middle (D/A) bowing plane, for poses R (N,3,3) at t.

    For each true string and test angle: build that bow in 3D, form its
    interpretation plane (what the camera sees), re-solve assuming the middle
    plane, and compare angles. Returns (N,) degrees.
    """
    R = np.atleast_3d(R).reshape(-1, 3, 3)
    C = np.einsum("nij,j->ni", R, np.array([contact_mm, 0.0, 0.0])) + t  # contact point, camera frame
    x = R[:, :, 0]
    n_mid = np.einsum("nij,j->ni", R, model.string_plane_normal(None))
    worst = np.zeros(len(R))
    for s in STRINGS:
        ns_v = model.string_plane_normal(s)
        lat = np.cross(ns_v, [1.0, 0.0, 0.0])  # in the plane, perpendicular to the strings
        for a in np.radians(TEST_ANGLES_DEG):
            d_true = np.einsum("nij,j->ni", R, np.cos(a) * np.array([1.0, 0, 0]) + np.sin(a) * lat)
            n = np.cross(C, d_true)  # interpretation plane normal
            d = np.cross(n, n_mid)
            norm = np.linalg.norm(d, axis=1)
            cosang = np.abs(np.einsum("ni,ni->n", d, x)) / np.maximum(norm, 1e-12)
            ang = np.degrees(np.arccos(np.clip(cosang, 0, 1)))
            true = min(np.degrees(a), 180 - np.degrees(a))
            err = np.where(norm > 1e-9, np.abs(ang - true), 90.0)
            worst = np.maximum(worst, err)
    return worst


class PositionAdvisor:
    def __init__(self, model: ViolinModel, max_error_deg: float = 5.0, min_px_per_mm: float = 0.4,
                 margin_frac: float = 0.03, window: int = 15):
        self.model = model
        self.max_error_deg = max_error_deg
        self.min_px_per_mm = min_px_per_mm
        self.margin_frac = margin_frac
        self._hist: deque[PositionAdvice] = deque(maxlen=max(1, window))
        pts = list(model.points.values()) + list(model.parts()["body"])
        self._outline_pts = np.array(pts)

    @classmethod
    def from_config(cls, cfg: dict, model: ViolinModel) -> "PositionAdvisor":
        p = cfg.get("positioning", {})
        return cls(model, float(p.get("max_string_error_deg", 5.0)), float(p.get("min_px_per_mm", 0.4)),
                   float(p.get("margin_frac", 0.03)), int(p.get("window", 15)))

    def reset(self) -> None:
        self._hist.clear()

    # -- single frame ---------------------------------------------------------
    def assess(self, pose: ViolinPose | None, cam: CameraIntrinsics) -> PositionAdvice:
        if pose is None:
            return PositionAdvice("unknown", "Violin not found: get the whole violin in view")
        w, h = cam.width, cam.height

        # Framing: is every part of the violin inside the frame (with a margin)?
        px = cam.project(pose.to_camera(self._outline_pts))
        m = self.margin_frac * np.array([w, h])
        lo, hi = px.min(0), px.max(0)
        framing = None
        if (hi - lo)[0] > w - 2 * m[0] or (hi - lo)[1] > h - 2 * m[1]:
            framing = "back"
        elif lo[0] < m[0]:
            framing = "left"  # cut off at the image's left: move image-right = the player's left
        elif hi[0] > w - m[0]:
            framing = "right"
        elif lo[1] < m[1]:
            framing = "down"
        elif hi[1] > h - m[1]:
            framing = "up"
        px_per_mm = float(cam.K[0, 0] / pose.t[2])
        if framing is None and px_per_mm < self.min_px_per_mm:
            framing = "closer"

        # Viewpoint: error now, and after turning by each candidate angle.
        errs = string_error_deg(np.einsum("nij,jk->nik", _rot_y(-np.radians(TURNS_DEG)), pose.R), pose.t,
                                self.model)
        err_now = float(errs[np.argmin(np.abs(TURNS_DEG))])
        good = np.flatnonzero(errs <= self.max_error_deg)
        i = good[np.argmin(np.abs(TURNS_DEG[good]))] if len(good) else int(np.argmin(errs))
        turn, best = float(TURNS_DEG[i]), float(errs[i])
        raise_deg = raise_err = None
        if best > self.max_error_deg:
            # Turning alone can't fix it (top plate seen too edge-on): try a higher camera too.
            Rt = _rot_y(-np.radians(TURNS_DEG))
            for psi in RAISES_DEG[1:]:
                Rr = np.einsum("ij,njk,kl->nil", _rot_x(np.radians([psi]))[0], Rt, pose.R)
                e = string_error_deg(Rr, pose.t, self.model)
                if e.min() <= self.max_error_deg or psi == RAISES_DEG[-1]:
                    j = int(np.argmin(e))
                    if e[j] < best - 1.0:
                        raise_deg, raise_err, turn = float(psi), float(e[j]), float(TURNS_DEG[j])
                    break
        return PositionAdvice("adjust", "", string_error_deg=err_now, turn_deg=turn, best_error_deg=best,
                              raise_deg=raise_deg, raise_error_deg=raise_err,
                              view_deg=pose.view_angle_to_strings_deg, framing=framing, px_per_mm=px_per_mm)

    # -- smoothed over recent frames -----------------------------------------
    def update(self, pose: ViolinPose | None, cam: CameraIntrinsics, string_known: bool = False) -> PositionAdvice:
        """`string_known`: the player chose the string in the GUI (practising on one string), so
        the string ambiguity mostly disappears and a worse viewpoint is acceptable (2x the error)."""
        a = self.assess(pose, cam)
        self._hist.append(a)
        known = [h for h in self._hist if h.status != "unknown"]
        if len(known) * 2 < len(self._hist):  # mostly not found lately
            return PositionAdvice("unknown", "Violin not found: get the whole violin in view")
        err = float(np.median([h.string_error_deg for h in known]))
        turn = float(np.median([h.turn_deg for h in known]))
        best = float(np.median([h.best_error_deg for h in known]))
        view = float(np.median([h.view_deg for h in known]))
        ppm = float(np.median([h.px_per_mm for h in known]))
        framing, count = Counter(h.framing for h in known).most_common(1)[0]
        if count * 2 < len(known):
            framing = None
        raises = [h.raise_deg for h in known if h.raise_deg is not None]
        raise_deg = raise_err = None
        if len(raises) * 2 > len(known):
            raise_deg = float(np.median(raises))
            raise_err = float(np.median([h.raise_error_deg for h in known if h.raise_error_deg is not None]))
        out = PositionAdvice("adjust", "", string_error_deg=err, turn_deg=turn, best_error_deg=best,
                             raise_deg=raise_deg, raise_error_deg=raise_err, view_deg=view, framing=framing,
                             px_per_mm=ppm)
        out.status, out.text = self._message(out, self.max_error_deg * (2.0 if string_known else 1.0))
        return out

    def _message(self, a: PositionAdvice, max_err: float) -> tuple[str, str]:
        msgs = {
            "back": "Step back: the violin doesn't fit in the frame",
            "left": "Move to your left: the violin is cut off",
            "right": "Move to your right: the violin is cut off",
            "down": "Violin cut off at the top: lower the violin or aim the camera higher",
            "up": "Violin cut off at the bottom: raise the violin or aim the camera lower",
            "closer": "Step closer: the violin is too small in the frame",
        }
        if a.framing in ("back", "left", "right", "down", "up"):
            return "adjust", msgs[a.framing]
        if a.string_error_deg > max_err:
            if a.raise_deg is not None:
                turn = ""
                if abs(a.turn_deg) >= 4:
                    turn = f", and turn ~{abs(a.turn_deg):.0f}° to your {'left' if a.turn_deg > 0 else 'right'}"
                return "adjust", (f"Raise the camera to look down onto the violin's top (~{a.raise_deg:.0f}° more)"
                                  f"{turn}  (±{a.string_error_deg:.0f}° now, ±{a.raise_error_deg:.0f}° after)")
            if abs(a.turn_deg) >= 4 and a.best_error_deg < a.string_error_deg - 1.0:
                side = "left" if a.turn_deg > 0 else "right"
                how = "slightly" if abs(a.turn_deg) < 10 else f"~{abs(a.turn_deg):.0f}°"
                return "adjust", (f"Turn your body {how} to your {side}  "
                                  f"(bow angle ±{a.string_error_deg:.0f}° from here, ±{a.best_error_deg:.0f}° after)")
            if a.framing == "closer":
                return "adjust", msgs["closer"]
            return "limited", f"Viewpoint limits accuracy: bow angle ±{a.string_error_deg:.0f}° depending on the string"
        if a.framing == "closer":
            return "adjust", msgs["closer"]
        return "ok", f"Position good  (view {a.view_deg:.0f}° to the strings)"
