# -*- coding: utf-8 -*-
"""
overlay_mse_summary_noargs.py

複数実験の mse_summary.csv（per-trial ref @ step=1 正規化）を読み込み、
mean ± std を同一グラフにオーバープロットする（argparseなし）。

前提CSV列:
  step, time_sec, mse_norm_mean, mse_norm_std
"""

import os
import csv
from typing import List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"


# =========================================================
# ★ ここだけ編集すればOK
# =========================================================

# 1) 重ねたい mse_summary.csv のパス（順番がそのまま凡例の順）
CSV_PATHS = [
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_5\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\mse_summary.csv",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi_recon\maxmod_5\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\mse_summary.csv",
    # r"C:\path\to\exp_MAX6\mse_summary.csv",
]

# 2) 凡例ラベル（Noneにするとフォルダ名などから自動生成）
# LABELS = [
#     "#Modules=1",
#     "#Modules=2",
#     "#Modules=3",
#     "#Modules=4",
#     "#Modules=5",
#     # "MAX=6",
# ]
LABELS = [
    "normal",
    "left-mask recon",
    # "MAX=6",
]

# 3) x軸： "time" or "step"
X_AXIS = "time"   # "step" にすると step 軸

# 4) 出力PNG
OUT_PNG = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi_recon\overlay_mse.png"

# 5) 表示設定
SHIFT_ORIGIN_TO_FIRST = True     # ★ step=1/time=first を 0 にする
SHOW_BAND = True                # std帯を描く
SHOW_BAND_LEGEND = True         # std帯も凡例に出す（線+帯で2つ出る）
BAND_ALPHA = 0.15               # std帯の透明度
YLIM_BOTTOM_ZERO = True         # y下限0固定


# =========================================================
# 実装
# =========================================================

def _read_mse_summary_csv(path: str) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)

    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    need = ["step", "time_sec", "mse_norm_mean", "mse_norm_std"]
    for k in need:
        if k not in rows[0]:
            raise KeyError(f"Missing column '{k}' in: {path}")

    step = np.array([int(float(row["step"])) for row in rows], dtype=np.int32)
    time_sec = np.array([float(row["time_sec"]) for row in rows], dtype=np.float32)
    mean = np.array([float(row["mse_norm_mean"]) for row in rows], dtype=np.float32)
    std  = np.array([float(row["mse_norm_std"]) for row in rows], dtype=np.float32)

    return {"step": step, "time_sec": time_sec, "mean": mean, "std": std}


def _auto_label_from_path(csv_path: str) -> str:
    # .../exp_xxx/mse_summary.csv -> exp_xxx
    return os.path.basename(os.path.dirname(csv_path.rstrip("/\\")))


def load_series(csv_path: str, label: Optional[str], x_axis: str) -> Dict[str, np.ndarray]:
    d = _read_mse_summary_csv(csv_path)

    if x_axis == "time":
        x = d["time_sec"].astype(np.float32)
    elif x_axis == "step":
        x = d["step"].astype(np.float32)
    else:
        raise ValueError("X_AXIS must be 'time' or 'step'")

    if SHIFT_ORIGIN_TO_FIRST:
        x = x - float(x[0])  # ★ step=1/time=first を 0 に

    if label is None or str(label).strip() == "":
        label = _auto_label_from_path(csv_path)

    return {"x": x, "mean": d["mean"], "std": d["std"], "label": label}


def overlay_plot(series_list: List[Dict[str, np.ndarray]]) -> None:
    plt.figure(figsize=(7, 4.5))

    for s in series_list:
        # mean
        line, = plt.plot(s["x"], s["mean"], marker="o", label=f'{s["label"]} (mean)')

        # std band
        if SHOW_BAND:
            plt.fill_between(
                s["x"],
                s["mean"] - s["std"],
                s["mean"] + s["std"],
                alpha=BAND_ALPHA,
                color=line.get_color(),
                label=(f'{s["label"]} (std)' if SHOW_BAND_LEGEND else None),
            )

    if X_AXIS == "time":
        xlabel = "Time [s]"
    else:
        xlabel = "Time step"

    plt.xlabel(xlabel)
    plt.ylabel("Normalized MSE")
    #plt.title("Overlay: mean ± std (per-trial normalized)")
    plt.grid(True, linestyle="--", alpha=0.5)

    if YLIM_BOTTOM_ZERO:
        plt.ylim(bottom=0.0)

    # 変な余白を作らない自動xlim
    xmin = min(float(s["x"][0]) for s in series_list)
    xmax = max(float(s["x"][-1]) for s in series_list)
    plt.xlim(xmin, xmax)

    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG)
    plt.close()
    print(f"[OK] saved -> {OUT_PNG}")


def main():
    if LABELS is not None and len(LABELS) != len(CSV_PATHS):
        raise ValueError("LABELS length must match CSV_PATHS length (or set LABELS=None).")

    series_list = []
    for i, p in enumerate(CSV_PATHS):
        lab = None if LABELS is None else LABELS[i]
        series_list.append(load_series(p, lab, X_AXIS))

    overlay_plot(series_list)


if __name__ == "__main__":
    main()
