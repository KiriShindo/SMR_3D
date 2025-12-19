# -*- coding: utf-8 -*-
"""
train_left_completion.py

Left(赤)マスク欠損を、(Center/Rightの2ch + 現在電圧) から補完するネットの訓練。

入力:
  - cam_mask_2ch: (2,H,W) = [center, right]  ※ left欠損を想定して入力から外す
  - q_map:        (3,H,W) = [L,C,R] 電圧をH×Wに敷き詰めたもの（0-1 or zscore）

出力:
  - left_hat: (1,H,W)  ※ leftマスクのみ推定（logits）

学習:
  - BCEWithLogits + Dice
  - 追加で overlap penalty（任意、生成leftがcenter/rightに食い込むのを抑制）

ログ:
  - train/val loss CSV
  - loss_curve.png
  - 固定サンプルの GT vs Pred（3ch合成の可視化も）
"""

import os
import csv
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


# =========================
# Dataset
# =========================

class LeftCompletionDataset(Dataset):
    def __init__(
        self,
        beammask3ch_dir,
        csv_path,
        col_L="L1_V",
        col_C="C1_V",
        col_R="R1_V",
        voltage_mode="scale01",   # "scale01" or "zscore"
        left_drop_mode="remove",  # "remove": 入力にleftを入れない（2ch固定）
        mask_suffix="_beam3ch.npy"
    ):
        self.beammask3ch_dir = Path(beammask3ch_dir)
        self.csv_path = Path(csv_path)

        self.col_L = col_L
        self.col_C = col_C
        self.col_R = col_R
        self.voltage_mode = voltage_mode
        self.left_drop_mode = left_drop_mode
        self.mask_suffix = mask_suffix

        self.samples = []
        volts_list = []

        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["filename"]
                stem = Path(fname).stem

                npy_path = self.beammask3ch_dir / f"{stem}{self.mask_suffix}"
                if not npy_path.exists():
                    continue

                vL = float(row[self.col_L])
                vC = float(row[self.col_C])
                vR = float(row[self.col_R])

                self.samples.append({
                    "npy_path": npy_path,
                    "v": (vL, vC, vR),
                    "stem": stem,
                })
                volts_list.append([vL, vC, vR])

        if len(self.samples) == 0:
            raise RuntimeError("有効サンプル0です。beammask3ch_dir/csv/列名/サフィックスを確認して。")

        tmp = np.load(self.samples[0]["npy_path"])
        if tmp.ndim != 3 or tmp.shape[0] != 3:
            raise RuntimeError(f"想定外mask形状: {tmp.shape} (期待: (3,H,W))")
        self.C, self.H, self.W = tmp.shape

        volts_arr = np.array(volts_list, dtype=np.float32)  # (N,3)
        self.v_mean = volts_arr.mean(axis=0)
        self.v_std  = volts_arr.std(axis=0) + 1e-6

        print(f"[INFO] voltage mean={self.v_mean}, std={self.v_std}")
        print(f"[INFO] mask shape=(3,{self.H},{self.W}) total={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _make_qmap(self, v_raw):
        """
        v_raw: (3,) 生電圧
        return: (3,H,W) float32
        """
        v = np.array(v_raw, dtype=np.float32)

        if self.voltage_mode == "scale01":
            # 0-5V を 0-1 に
            v = v / 5.0
        elif self.voltage_mode == "zscore":
            v = (v - self.v_mean) / self.v_std
        else:
            raise ValueError(f"unknown voltage_mode: {self.voltage_mode}")

        q = np.zeros((3, self.H, self.W), dtype=np.float32)
        q[0, :, :] = v[0]
        q[1, :, :] = v[1]
        q[2, :, :] = v[2]
        return q

    def __getitem__(self, idx):
        s = self.samples[idx]

        mask3 = np.load(s["npy_path"]).astype(np.float32)
        mask3 = np.clip(mask3, 0.0, 1.0)

        # GT left
        left_gt = mask3[0:1]  # (1,H,W)

        # 入力2ch（center/right）
        cam_mask2 = mask3[1:3]  # (2,H,W)

        q_map = self._make_qmap(s["v"])  # (3,H,W)

        x5 = np.concatenate([cam_mask2, q_map], axis=0)  # (5,H,W)

        return (
            torch.from_numpy(x5),        # (5,H,W)
            torch.from_numpy(left_gt),   # (1,H,W)
            s["stem"]
        )


# =========================
# Model（軽量UNet風）
# =========================

class LeftMaskCompletionNet(nn.Module):
    def __init__(self, in_ch=5, base=32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base*2, base*2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.mid = nn.Sequential(
            nn.Conv2d(base*2, base*4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base*4, base*4, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base*4 + base*2, base*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base*2, base*2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base*2 + base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(base, 1, 1)  # left logits

    import torch.nn.functional as F

    def forward(self, x):
        e1 = self.enc1(x)         # (B, base, H, W)
        p1 = self.pool1(e1)       # (B, base, H/2, W/2)

        e2 = self.enc2(p1)        # (B, 2base, H/2, W/2) 例: 35x35
        p2 = self.pool2(e2)       # (B, 2base, H/4, W/4) 例: 17x17

        m  = self.mid(p2)         # (B, 4base, H/4, W/4)

        # ★重要：skip(e2)の空間サイズに合わせてアップサンプル
        u2 = F.interpolate(m, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        # ★重要：skip(e1)の空間サイズに合わせてアップサンプル
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.head(d1)      # logits (B,1,H,W)



# =========================
# Loss: BCE + Dice + overlap(optional)
# =========================

class BCEDiceOverlapLoss(nn.Module):
    def __init__(self, bce_w=1.0, dice_w=1.0, overlap_w=0.0):
        super().__init__()
        self.bce_w = bce_w
        self.dice_w = dice_w
        self.overlap_w = overlap_w
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, left_logits, left_gt, cam_mask2=None):
        """
        left_logits: (B,1,H,W)
        left_gt:     (B,1,H,W)
        cam_mask2:   (B,2,H,W)  ※ overlap penaltyに使う（任意）
        """
        bce = self.bce(left_logits, left_gt)

        probs = torch.sigmoid(left_logits)  # (B,1,H,W)
        probs_f = probs.view(probs.size(0), -1)
        gt_f = left_gt.view(left_gt.size(0), -1)
        inter = (probs_f * gt_f).sum(dim=1)
        union = probs_f.sum(dim=1) + gt_f.sum(dim=1) + 1e-6
        dice = (1.0 - 2.0 * inter / union).mean()

        loss = self.bce_w * bce + self.dice_w * dice

        # 生成leftが center/right と重なるのを抑える（任意）
        if (self.overlap_w > 0.0) and (cam_mask2 is not None):
            # cam_mask2: (B,2,H,W) -> (B,1,H,W) 合成
            exist = torch.clamp(cam_mask2.sum(dim=1, keepdim=True), 0.0, 1.0)
            overlap = (probs * exist).mean()
            loss = loss + self.overlap_w * overlap

        return loss


# =========================
# Utils（ログ・可視化）
# =========================

def save_loss_curve(log_dir, epochs, train_losses, val_losses):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    out_path = log_dir / "loss_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[LOG] loss curve saved: {out_path}")

def save_loss_csv(log_dir, epochs, train_losses, val_losses):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "loss_log.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss"])
        for e, tr, vl in zip(epochs, train_losses, val_losses):
            w.writerow([e, tr, vl])
    print(f"[LOG] loss csv saved: {csv_path}")

def save_debug_samples(model, dataset, indices, out_dir, device="cuda", epoch=None):
    """
    fixed samples:
      - 入力(2ch)から復元した left を 3ch に合成して可視化
      - GT 3ch と Pred 3ch を横並び保存
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for i, idx in enumerate(indices):
            x5, left_gt, stem = dataset[idx]
            x5b = x5.unsqueeze(0).to(device)          # (1,5,H,W)
            left_logits = model(x5b)                  # (1,1,H,W)
            left_prob = torch.sigmoid(left_logits)[0,0].cpu().numpy()
            left_bin = (left_prob > 0.5).astype(np.uint8)

            # もともとの入力2ch（center/right）も取り出す
            x5_np = x5.numpy()
            cam_mask2 = x5_np[0:2]                    # (2,H,W)
            center_bin = (cam_mask2[0] > 0.5).astype(np.uint8)
            right_bin  = (cam_mask2[1] > 0.5).astype(np.uint8)

            # GT left
            gt_left_bin = (left_gt.numpy()[0] > 0.5).astype(np.uint8)

            H, W = gt_left_bin.shape

            # GT 3ch可視化
            gt_vis = np.zeros((H, W, 3), dtype=np.uint8)
            gt_vis[gt_left_bin.astype(bool)] = (0, 0, 255)     # left: red (BGR)
            gt_vis[center_bin.astype(bool)]  = (0, 255, 0)     # center: green
            gt_vis[right_bin.astype(bool)]   = (255, 0, 0)     # right: blue

            # Pred 3ch可視化（leftだけ予測、他は入力のまま）
            pred_vis = np.zeros((H, W, 3), dtype=np.uint8)
            pred_vis[left_bin.astype(bool)]  = (0, 0, 255)
            pred_vis[center_bin.astype(bool)]= (0, 255, 0)
            pred_vis[right_bin.astype(bool)] = (255, 0, 0)

            concat = np.concatenate([gt_vis, pred_vis], axis=1)

            if epoch is None:
                out_name = f"gen_{i:02d}_{stem}.png"
            else:
                out_name = f"epoch{epoch:03d}_gen_{i:02d}_{stem}.png"

            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), concat)
            print(f"[LOG] debug sample saved: {out_path}")


# =========================
# Train loop
# =========================

def train_left_completion(
    beammask3ch_dir,
    csv_path,
    col_L="L1_V",
    col_C="C1_V",
    col_R="R1_V",
    voltage_mode="scale01",
    batch_size=32,
    num_epochs=300,
    lr=1e-3,
    val_ratio=0.1,
    device="cuda",
    save_interval=50,
    log_subdir="training_logs_left_completion",
    bce_w=1.0,
    dice_w=1.0,
    overlap_w=0.0,   # まずは 0.0 でOK（必要なら 0.1 とか）
):
    full_ds = LeftCompletionDataset(
        beammask3ch_dir=beammask3ch_dir,
        csv_path=csv_path,
        col_L=col_L, col_C=col_C, col_R=col_R,
        voltage_mode=voltage_mode,
        left_drop_mode="remove",
        mask_suffix="_beam3ch.npy",
    )

    n_total = len(full_ds)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    print(f"[INFO] n_train={n_train}, n_val={n_val}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = LeftMaskCompletionNet(in_ch=5, base=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = BCEDiceOverlapLoss(bce_w=bce_w, dice_w=dice_w, overlap_w=overlap_w)

    beammask3ch_dir = Path(beammask3ch_dir)
    log_dir = beammask3ch_dir / log_subdir
    log_dir.mkdir(parents=True, exist_ok=True)

    # 固定サンプル
    np.random.seed(0)
    num_gen_samples = min(5, len(full_ds))
    fixed_indices = np.random.choice(len(full_ds), size=num_gen_samples, replace=False)
    print(f"[INFO] fixed sample indices: {fixed_indices}")

    epochs_log, train_losses, val_losses = [], [], []
    best_val = float("inf")

    for epoch in range(1, num_epochs + 1):
        # ---- train ----
        model.train()
        train_sum = 0.0

        for x5, left_gt, _ in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            x5 = x5.to(device)              # (B,5,H,W)
            left_gt = left_gt.to(device)    # (B,1,H,W)

            opt.zero_grad()
            left_logits = model(x5)         # (B,1,H,W)

            cam_mask2 = x5[:, 0:2]          # (B,2,H,W) overlap用
            loss = criterion(left_logits, left_gt, cam_mask2=cam_mask2)

            loss.backward()
            opt.step()

            train_sum += loss.item() * x5.size(0)

        train_loss = train_sum / n_train

        # ---- val ----
        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for x5, left_gt, _ in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                x5 = x5.to(device)
                left_gt = left_gt.to(device)
                left_logits = model(x5)
                cam_mask2 = x5[:, 0:2]
                loss = criterion(left_logits, left_gt, cam_mask2=cam_mask2)
                val_sum += loss.item() * x5.size(0)

        val_loss = val_sum / n_val

        epochs_log.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        # best
        if val_loss < best_val:
            best_val = val_loss
            best_path = log_dir / "left_completion_best.pth"
            torch.save({"model_state_dict": model.state_dict()}, best_path)
            print(f"  -> best updated: {best_path}")

        save_loss_csv(log_dir, epochs_log, train_losses, val_losses)

        if (epoch % save_interval == 0) or (epoch == num_epochs):
            ckpt_path = log_dir / f"left_completion_epoch{epoch:03d}.pth"
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
            print(f"[LOG] checkpoint saved: {ckpt_path}")

            save_loss_curve(log_dir, epochs_log, train_losses, val_losses)

            gen_dir = log_dir / f"gen_epoch{epoch:03d}"
            save_debug_samples(model, full_ds, fixed_indices, gen_dir, device=device, epoch=epoch)

    return model, log_dir


# =========================
# main
# =========================

if __name__ == "__main__":
    # ★あなたの環境に合わせてここだけ変える★

    BEAM3CH_DIR = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control_3D/module_controller/IK/1module_babbling_data/normal/roi_aug_shift5_rot30_beammask_3ch"
    CSV_PATH    = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control_3D/module_controller/IK/1module_babbling_data/normal/aug_shift5_rot30/signals_aug.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, log_dir = train_left_completion(
        beammask3ch_dir=BEAM3CH_DIR,
        csv_path=CSV_PATH,
        col_L="L1_V", col_C="C1_V", col_R="R1_V",   # 列名が違うならここを変更
        voltage_mode="scale01",                     # まずは scale01 推奨（運用で /5.0 するなら合わせる）
        batch_size=64,
        num_epochs=5000,
        lr=1e-3,
        val_ratio=0.1,
        device=device,
        save_interval=100,
        log_subdir="training_logs_left_completion",
        bce_w=1.0,
        dice_w=1.0,
        overlap_w=0.0,
    )

    print(f"[INFO] training finished. logs saved in: {log_dir}")
