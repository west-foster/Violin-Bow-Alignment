"""Train the FP32 keypoint model on frames labelled in the GUI.

    python scripts/train.py                          # resnet50 teacher
    python scripts/train.py --backbone resnet18      # smaller model
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from bowtrack.annotations import AnnotationStore
from bowtrack.config import load_config, resolve
from bowtrack.dataset import KeypointDataset, split_entries
from bowtrack.detector import pick_device, save_checkpoint
from bowtrack.keypoints import KEYPOINT_NAMES
from bowtrack.model import HeatmapPoseNet, decode_heatmaps


@torch.no_grad()
def evaluate(model, loader, device, conf_thresh=0.35):
    model.eval()
    errs, tp, fp, fn, losses = [], 0, 0, 0, []
    for x, target, xy, vis in loader:
        x, target = x.to(device), target.to(device)
        out = model(x)
        losses.append(torch.nn.functional.mse_loss(out, target).item())
        pred, conf = decode_heatmaps(out)
        vis = vis.numpy()
        det = conf >= conf_thresh
        tp += int((det & vis).sum()); fp += int((det & ~vis).sum()); fn += int((~det & vis).sum())
        d = np.linalg.norm(pred - xy.numpy(), axis=-1)
        errs.append(d[vis])
    errs = np.concatenate(errs) if errs else np.zeros(0)
    return {
        "loss": float(np.mean(losses)),
        "mean_px": float(errs.mean()) if len(errs) else float("nan"),
        "pck5": float((errs < 5).mean()) if len(errs) else 0.0,  # within 5 input px
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
    }


def main():
    cfg = load_config()
    mc = cfg["model"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=mc["backbone"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data", default="data", help="folder containing annotations.json")
    args = ap.parse_args()

    input_hw = (mc["input_height"], mc["input_width"])
    store = AnnotationStore(resolve(args.data))
    entries = store.entries()
    if len(entries) < 20:
        raise SystemExit(f"Only {len(entries)} labelled frames in data/annotations.json - label more in the GUI.")
    train_e, val_e = split_entries(entries)
    print(f"{len(train_e)} train / {len(val_e)} val frames, keypoints: {KEYPOINT_NAMES}")

    train_dl = DataLoader(KeypointDataset(store, train_e, input_hw, mc["heatmap_sigma"], augment=True),
                          batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(KeypointDataset(store, val_e, input_hw, mc["heatmap_sigma"]), batch_size=args.batch,
                        num_workers=args.workers)

    device = pick_device()
    model = HeatmapPoseNet(args.backbone).to(device)
    # Pretrained backbone learns 10x slower than the fresh head.
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": list(model.head.parameters()) + list(model.final.parameters()), "lr": args.lr},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr * 0.1, args.lr],
                                                total_steps=args.epochs * len(train_dl), pct_start=0.1)
    out_path = resolve(args.out or f"checkpoints/teacher_{args.backbone}.pt")
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        t0, run = time.time(), []
        for x, target, _, _ in train_dl:
            x, target = x.to(device), target.to(device)
            # Heatmaps are mostly zeros; scale the loss so gradients aren't tiny.
            loss = torch.nn.functional.mse_loss(model(x), target) * 1000
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run.append(loss.item())
        m = evaluate(model, val_dl, device, cfg["feedback"]["min_confidence"])
        print(f"epoch {epoch + 1:3d}  train {np.mean(run):.3f}  val {m['loss'] * 1000:.3f}  "
              f"err {m['mean_px']:.1f}px  pck5 {m['pck5']:.2f}  P {m['precision']:.2f} R {m['recall']:.2f}  "
              f"{time.time() - t0:.0f}s")
        if m["loss"] < best:
            best = m["loss"]
            save_checkpoint(out_path, model, input_hw, epoch=epoch + 1, metrics=m)
    print(f"best checkpoint -> {out_path}")


if __name__ == "__main__":
    main()
