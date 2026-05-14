# debug_once.py
import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader

# ========== same sys.path trick as your train.py ==========
rootdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, rootdir)

from data.collator import zero_pad_collator
from model import BODY_COARSE_POSE

# --------- match your coarse-part order ----------
PART_ORDER = [
    "RIGHT_ARM_UPPER",
    "RIGHT_ARM_LOWER",
    "LEFT_ARM_UPPER",
    "LEFT_ARM_LOWER",
    "HEAD",

    "LEFT_THUMB_LOWER", "LEFT_THUMB_UPPER",
    "LEFT_INDEX_LOWER", "LEFT_INDEX_UPPER",
    "LEFT_MIDDLE_LOWER", "LEFT_MIDDLE_UPPER",
    "LEFT_RING_LOWER", "LEFT_RING_UPPER",
    "LEFT_LITTLE_LOWER", "LEFT_LITTLE_UPPER",

    "RIGHT_THUMB_LOWER", "RIGHT_THUMB_UPPER",
    "RIGHT_INDEX_LOWER", "RIGHT_INDEX_UPPER",
    "RIGHT_MIDDLE_LOWER", "RIGHT_MIDDLE_UPPER",
    "RIGHT_RING_LOWER", "RIGHT_RING_UPPER",
    "RIGHT_LITTLE_LOWER", "RIGHT_LITTLE_UPPER"
]

EPS = 1e-6

@torch.no_grad()
def coarse_mean(pose_xy: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # pose_xy: [B,T,K,2] -> [B,T,2]
    return pose_xy[:, :, idx, :].mean(dim=2)

@torch.no_grad()
def coarse_weighted(pose_xy: torch.Tensor, conf: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # pose_xy: [B,T,K,2], conf: [B,T,K] -> [B,T,2]
    part_xy = pose_xy[:, :, idx, :]                  # [B,T,P,2]
    part_w  = conf[:, :, idx].unsqueeze(-1)          # [B,T,P,1]
    num = (part_xy * part_w).sum(dim=2)              # [B,T,2]
    den = part_w.sum(dim=2).clamp_min(EPS)           # [B,T,1]
    return num / den

def main():
    # ====== EXACTLY like your train.py: load pkl and set train_dataset=test_dataset ======
    pkl_path = "./temp/train/test_dataset_old.pkl"
    with open(pkl_path, "rb") as f:
        test_dataset = pickle.load(f)
        train_dataset = test_dataset
    print(f"[OK] Load dataset from {pkl_path}")
    print("[INFO] train_dataset = test_dataset (same as your train.py)")

    # ====== batch size: mimic a typical value; change if needed ======
    batch_size = 8
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=zero_pad_collator)

    # ====== get ONE batch ======
    batch = next(iter(loader))
    pose_xy = batch["pose"]["data"]           # [B,T,K,2]
    print(pose_xy.shape)
    conf = batch["pose"]["confidence"]        # [B,T,K]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pose_xy = pose_xy.to(device)
    conf = conf.to(device)

    B, T, K, _ = pose_xy.shape
    print(f"[BATCH] pose_xy shape={tuple(pose_xy.shape)}, conf shape={tuple(conf.shape)}, device={device}")
    print("[CONF] min/max:", float(conf.min()), float(conf.max()))
    print("[CONF] overall zero rate:", float((conf == 0).float().mean()))

    # ====== per-part den stats ======
    den_small_thr = 0.1
    print("\n=== Per-part den stats (den = sum(conf over part indices)) ===")
    stats = []
    for part in PART_ORDER:
        idx = torch.as_tensor(list(BODY_COARSE_POSE[part]), device=device, dtype=torch.long)
        den = conf[:, :, idx].sum(dim=-1)  # [B,T]
        den0_rate = float((den == 0).float().mean())
        small_rate = float((den < den_small_thr).float().mean())
        mean_den = float(den.mean())
        stats.append((part, den0_rate, small_rate, mean_den))

    # sort by den==0 rate desc
    stats.sort(key=lambda x: x[1], reverse=True)
    for part, den0_rate, small_rate, mean_den in stats:
        print(f"{part:20s} den==0:{den0_rate:.3f}  den<{den_small_thr}:{small_rate:.3f}  mean_den:{mean_den:.3f}")

    # ====== mean vs weighted discrepancy ======
    print("\n=== Mean vs Weighted coarse discrepancy (per-part MSE) ===")
    diffs = []
    diffs_lowden = []
    for part in PART_ORDER:
        idx = torch.as_tensor(list(BODY_COARSE_POSE[part]), device=device, dtype=torch.long)
        den = conf[:, :, idx].sum(dim=-1)  # [B,T]

        m = coarse_mean(pose_xy, idx)                 # [B,T,2]
        w = coarse_weighted(pose_xy, conf, idx)       # [B,T,2]
        d = ((m - w) ** 2).mean(dim=-1)               # [B,T]

        diff_all = float(d.mean())
        diffs.append(diff_all)

        low = den < den_small_thr
        diff_low = float(d[low].mean()) if low.any() else None
        if diff_low is not None:
            diffs_lowden.append(diff_low)

        low_str = "None" if diff_low is None else f"{diff_low:.6f}"
        print(f"{part:20s} diff_all:{diff_all:.6f}  diff_lowden:{low_str}")

    print("\n[SUMMARY] avg diff_all over parts:", float(np.mean(diffs)))
    if len(diffs_lowden) > 0:
        print("[SUMMARY] avg diff_lowden over parts:", float(np.mean(diffs_lowden)))
    else:
        print("[SUMMARY] no low-den frames under threshold in this batch (rare)")

    # ====== optional: pick a problematic part and report example frames ======
    worst = stats[0][0]
    idx = torch.as_tensor(list(BODY_COARSE_POSE[worst]), device=device, dtype=torch.long)
    den = conf[:, :, idx].sum(dim=-1)  # [B,T]
    print(f"\n[EXAMPLE] Worst part by den==0 in this batch: {worst}")
    # show first sample's den timeline (first 30 frames)
    den_line = den[0].detach().cpu().numpy()
    print("den[0][:30] =", np.array2string(den_line[:30], precision=3, separator=","))

if __name__ == "__main__":
    main()
