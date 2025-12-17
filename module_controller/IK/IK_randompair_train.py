import os
import csv
from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ==========================================================
# パス設定（★3chマスク版に合わせて適宜直して）
# ==========================================================

# 3chビームマスク (*.npy: ch0=Left, ch1=Center, ch2=Right) が入っているディレクトリ
BEAMMASK_DIR = Path(
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D"
    r"\module_controller\IK\1module_babbling_data\normal\roi_aug_shift5_rot30_beammask_3ch"
)

# バブリング拡張時に出力した signals_aug.csv
# ヘッダー: filename, orig_index, L1_V, C1_V, R1_V, ...
CSV_AUG_PATH = Path(
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D"
    r"\module_controller\IK\1module_babbling_data\normal\aug_shift5_rot30\signals_aug.csv"
)

# モデル保存先・ログ保存先
MODEL_SAVE_PATH = Path(
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D"
    r"\module_controller\IK\ik_beam3ch_shift5_rot30_randompair_model.pth"
)
LOG_DIR = Path(
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D"
    r"\module_controller\IK\training_logs_beam3ch_shift5_rot30_randompair"
)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# ハイパーパラメータ
# ==========================================================
BATCH_SIZE   = 64
EPOCHS       = 5000
LR           = 1e-3
VAL_RATIO    = 0.1
WEIGHT_DECAY = 1e-4
SAVE_INTERVAL = 100

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ==========================================================
# Dataset 定義（ランダム異orig_indexペア, 3ch版）
# ==========================================================

class IKBeamMixedPairDataset3Ch(Dataset):
    """
    signals_aug.csv の各行を「1サンプル」とみなし、__getitem__ で

      anchor = 自分自身 (index = idx)
      partner =
        - 確率 p_same で「同じ orig_index」の別サンプル
        - それ以外は「orig_index が異なるサンプル」

    を選び，

      入力 x:
        - anchor の 3chビームマスク: mask_i (3, H, W)
        - partner の 3chビームマスク: mask_j (3, H, W)
        - anchor の電圧 q_i を 3ch 一様マップにしたもの
            q_map_i[0,:,:] = L1_V_i / 5.0
            q_map_i[1,:,:] = C1_V_i / 5.0
            q_map_i[2,:,:] = R1_V_i / 5.0
          → q_map_i: (3, H, W)
        → x = concat([mask_i, mask_j, q_map_i], dim=0) → (9, H, W)

      出力 y:
        - partner の電圧 (L1_V^j, C1_V^j, R1_V^j) [0〜5V]

    を返す 3Dロボット用 Dataset。
    """

    def __init__(self, beammask_dir: Path, csv_aug_path: Path, p_same: float = 0.3):
        super().__init__()
        self.beammask_dir = Path(beammask_dir)
        self.p_same = float(p_same)

        # CSV 読み込み
        with open(csv_aug_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if len(rows) == 0:
            raise RuntimeError("signals_aug.csv にデータ行がありません。")

        samples = []
        for row in rows:
            # 必要なカラムが揃っているかチェック
            if not all(k in row for k in ["filename", "orig_index", "L1_V", "C1_V", "R1_V"]):
                continue

            fname = row["filename"]
            stem  = os.path.splitext(fname)[0]

            # ★3chビームマスクのファイル名規約に合わせて suffix を変更
            npy_path = self.beammask_dir / f"{stem}_beam3ch.npy"
            if not npy_path.exists():
                continue

            try:
                orig_idx = int(row["orig_index"])
                vL = float(row["L1_V"])
                vC = float(row["C1_V"])
                vR = float(row["R1_V"])
            except Exception:
                continue

            samples.append({
                "filename": fname,
                "orig_index": orig_idx,
                "npy_path": npy_path,
                "vL": vL,
                "vC": vC,
                "vR": vR,
            })

        if len(samples) < 2:
            raise RuntimeError("有効なサンプルが 2 件未満のためペアを構成できません。")

        self.samples = samples
        self.N = len(samples)

        # orig_index ごとのインデックスリストを作る
        self.indices_by_orig = {}
        for idx, s in enumerate(self.samples):
            o = s["orig_index"]
            self.indices_by_orig.setdefault(o, []).append(idx)

        orig_set = set(self.indices_by_orig.keys())
        if len(orig_set) < 2:
            raise RuntimeError("orig_index が 1種類しかなく、異なる orig_index ペアを作れません。")

        print(f"[INFO] 有効なサンプル数: {self.N}")
        print(f"[INFO] uniq orig_index 数: {len(orig_set)}")
        print(f"[INFO] P_SAME (同origペアの確率): {self.p_same}")

    def __len__(self):
        return self.N

    def _sample_partner_same_orig(self, orig_i: int, avoid_idx: int):
        """同じ orig_index から partner を 1つサンプリング"""
        candidates = self.indices_by_orig[orig_i]
        if len(candidates) == 1:
            # 自分しかいなければ自分を使う（電圧は同じなので実質問題なし）
            return candidates[0]

        # 自分以外からランダムに選ぶ
        while True:
            j = random.choice(candidates)
            if j != avoid_idx:
                return j

    def _sample_partner_diff_orig(self, orig_i: int):
        """orig_index が異なるサンプルから partner を 1つサンプリング"""
        while True:
            j = random.randint(0, self.N - 1)
            if self.samples[j]["orig_index"] != orig_i:
                return j

    def __getitem__(self, idx):
        # anchor
        item_i = self.samples[idx]
        orig_i = item_i["orig_index"]

        # partner を決める
        use_same = (random.random() < self.p_same)

        if use_same and orig_i in self.indices_by_orig:
            j = self._sample_partner_same_orig(orig_i, avoid_idx=idx)
        else:
            j = self._sample_partner_diff_orig(orig_i)

        item_j = self.samples[j]

        # 3chマスクをロード (3, H, W)
        mask_i = np.load(str(item_i["npy_path"])).astype(np.float32)  # (3, H, W)
        mask_j = np.load(str(item_j["npy_path"])).astype(np.float32)  # (3, H, W)

        mask_i_t = torch.from_numpy(mask_i)
        mask_j_t = torch.from_numpy(mask_j)

        # anchor の電圧を 3ch 一様マップに（V/5.0 正規化）
        vL_i = item_i["vL"]
        vC_i = item_i["vC"]
        vR_i = item_i["vR"]

        _, H, W = mask_i_t.shape
        q_map = torch.zeros((3, H, W), dtype=torch.float32)
        q_map[0, :, :] = vL_i / 5.0
        q_map[1, :, :] = vC_i / 5.0
        q_map[2, :, :] = vR_i / 5.0

        # 入力 x: (9, H, W) = mask_i(3) + mask_j(3) + q_map(3)
        x = torch.cat([mask_i_t, mask_j_t, q_map], dim=0)

        # 出力 y: partner の電圧 (L1, C1, R1) そのまま[0〜5]
        vL_j = item_j["vL"]
        vC_j = item_j["vC"]
        vR_j = item_j["vR"]
        y = torch.tensor([vL_j, vC_j, vR_j], dtype=torch.float32)

        return x, y


# ==========================================================
# モデル定義（3ch→9ch入力 & 3次元出力に変更）
# ==========================================================

class IKBeamEncoder(nn.Module):
    """
    9ch入力 (mask_i 3ch + mask_j 3ch + q_map_i 3ch) を受け取る CNN encoder
    """
    def __init__(self, in_ch: int = 9, feat_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(128, feat_dim)

    def forward(self, x):
        x = self.features(x)          # (B, 128, H', W')
        x = self.gap(x)               # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)     # (B, 128)
        x = self.fc(x)                # (B, feat_dim)
        return x


class VoltageHead(nn.Module):
    def __init__(self, in_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(64, 3),  # 出力: [V_L, V_C, V_R]
        )

    def forward(self, x):
        return self.net(x)


class IKBeamNet(nn.Module):
    """
    全体モデル: Encoder(9chマスク+q_map) → MLP → 3次元電圧 [L, C, R]
    """
    def __init__(self, in_ch: int = 9, feat_dim: int = 128):
        super().__init__()
        self.encoder = IKBeamEncoder(in_ch=in_ch, feat_dim=feat_dim)
        self.head    = VoltageHead(in_dim=feat_dim)

    def forward(self, x):
        feat = self.encoder(x)
        out  = self.head(feat)
        return out


# ==========================================================
# 学習・評価ループ（ほぼそのまま）
# ==========================================================

def train_epoch(model, loader, optimizer, criterion, device, epoch, max_epoch):
    model.train()
    total_loss = 0.0
    with tqdm(loader, desc=f"[Epoch {epoch}/{max_epoch}] Train", leave=False) as pbar:
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.6f}")
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device, epoch, max_epoch):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        with tqdm(loader, desc=f"[Epoch {epoch}/{max_epoch}] Val", leave=False) as pbar:
            for x, y in pbar:
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                total_loss += loss.item() * x.size(0)
                pbar.set_postfix(loss=f"{loss.item():.6f}")
    return total_loss / len(loader.dataset)


def save_curve(train_losses, val_losses, epoch):
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses,   label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("SmoothL1 Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_path = LOG_DIR / f"loss_curve_epoch{epoch:04d}.png"
    plt.savefig(out_path)
    plt.close()


# ==========================================================
# メイン
# ==========================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset 構築（3ch & ランダムペア版）
    dataset = IKBeamMixedPairDataset3Ch(BEAMMASK_DIR, CSV_AUG_PATH, p_same=0.1)
    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=True
    )

    # モデル・オプティマイザなど
    model = IKBeamNet(in_ch=9, feat_dim=128).to(device)
    criterion = nn.SmoothL1Loss(beta=0.1)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20
    )

    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, EPOCHS)
        val_loss   = eval_epoch(model, val_loader,   criterion, device, epoch, EPOCHS)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        scheduler.step(val_loss)

        # ベストモデル更新
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val,
            }, MODEL_SAVE_PATH)
            print(f"  → Best model updated: {best_val:.6f}")

        # スナップショット & 学習曲線保存
        if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
            ckpt_path = LOG_DIR / f"ik_beam3ch_randompair_epoch{epoch:04d}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
            }, ckpt_path)
            print(f"  → Saved checkpoint: {ckpt_path}")
            save_curve(train_losses, val_losses, epoch)

    print("\nTraining finished.")
    print(f"Best Val Loss = {best_val:.6f}")
    print(f"Final best model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
