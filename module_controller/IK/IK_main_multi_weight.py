# -*- coding: utf-8 -*-
import os
import time
import json
from pathlib import Path
from tkinter import Tk, filedialog
import csv

import cv2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import torch

# detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

# Arduino serial
from serial import Serial

# ★ あなたのIKモデル定義に合わせて import（パス/ファイル名は適宜）
# from IK_randompair_train import IKBeamNet
from IK_randompair_train import IKBeamNet


# =========================================================
#  ユーザ設定
# =========================================================

matplotlib.rcParams["font.family"] = "Times New Roman"

# Arduino / カメラ
SERIAL_PORT = "COM4"
BAUDRATE = 9600
CAMERA_INDEX = 0
CAM_W, CAM_H = 1920, 1080

BASE_ROOT = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control_3D/IK_only_result_mult_weight2"

# ------------------------
# 実験パラメータ
# ------------------------
WEIGHT = 0
K_GAIN = 1.0
NUM_LOOP = 10
RESET_EACH_LOOP = False
WAIT_TIME = 3.0
LOOP_WAIT = 5.0
MAX_MODULE = 5

# ★ trial回数
N_TRIALS = 1

# ------------------------
# 保存パス生成
# ------------------------
maxmod_dir = os.path.join(BASE_ROOT, f"maxmod_{MAX_MODULE}")
cond_name = (
    f"WEIGHT{WEIGHT}"
    f"K{K_GAIN}"
    f"_LOOP{NUM_LOOP}"
    f"_RESET{int(RESET_EACH_LOOP)}"
    f"_WAIT{WAIT_TIME}"
    f"_LOOPWAIT{LOOP_WAIT}"
    f"_TRIAL{N_TRIALS}"
)
SAVE_ROOT = os.path.join(maxmod_dir, cond_name)
os.makedirs(SAVE_ROOT, exist_ok=True)
print(f"[INFO] Save root: {SAVE_ROOT}")

# ROI設定（USBカメラ側）
ROI_CONFIG_FULL = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control_3D/devision_net/roi_config_full.json"

# =========================================================
#  Detectron2: devnet / seg-net 設定
# =========================================================

# devnet（モジュール検出）
DEVNET_TRAIN_JSON   = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control_3D/devision_net/devnet_data_new/all_dataset/annotations.json"
DEVNET_TRAIN_IMAGES = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control_3D/devision_net/devnet_data_new/all_dataset/"
DEVNET_WEIGHT       = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control_3D/devision_net/devnet_data_new/all_dataset/prm01/model_final.pth"

# seg-net（left/center/right の3クラス）
SEG3_TRAIN_JSON   = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control_3D/module_controller/IK/1module_jsons_3D_dataset/annotations.json"
SEG3_TRAIN_IMAGES = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control_3D/module_controller/IK/1module_jsons_3D_dataset/"
SEG3_WEIGHT       = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control_3D/module_controller/IK/1module_jsons_3D_dataset/prm01/model_final.pth"

# IKモデル（学習済み）
IK_MODEL_PATH = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control_3D/module_controller/IK/ik_beam3ch_shift5_rot30_randompair_model.pth"


# =========================================================
#  Utility: UI / IO
# =========================================================

def select_image_via_dialog():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select ROI image (target)",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    root.destroy()
    if not file_path:
        print("[INFO] No image selected.")
        return None
    return file_path


def open_camera(index=0, width=1920, height=1080):
    print(f"[Camera] Opening camera index {index} ...")
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print("[Camera] Failed to open camera.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def capture_frame(cap, grab_n=5):
    if cap is None or not cap.isOpened():
        print("[Camera] capture_frame called but cap is not opened.")
        return None
    for _ in range(grab_n):
        cap.grab()
    ret, frame = cap.read()
    if not ret:
        print("[Camera] Failed to grab frame.")
        return None
    return frame


def init_serial(port: str, baudrate: int):
    print(f"[Serial] Opening {port}@{baudrate} ...")
    ser = Serial(port, baudrate, timeout=1)

    # Arduino から READY が来るまで待つ（あなたの既存仕様）
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print(f"[Serial] <- {line}")
        if line == "READY":
            break

    print("[Serial] Arduino READY")
    return ser


def reset_voltages_to_zero(ser, n_channels):
    zeros = [0.0] * n_channels
    cmd = "VOLT " + ",".join(f"{v:.1f}" for v in zeros) + "\n"
    print(f"[Serial] -> {cmd.strip()}  (reset)")
    ser.write(cmd.encode())

    while True:
        resp = ser.readline().decode(errors="ignore").strip()
        if resp:
            print(f"[Serial] <- {resp}")
        if resp == "APPLIED":
            break
    print("[Serial] All channels reset to 0.0V.")


def send_voltages_to_arduino_3d(ser, module_volts_lcr, max_module=5, order="interleave"):
    """
    module_volts_lcr: shape (M,3) with [L,C,R] per module (top->bottom)
    order:
      - "interleave":  L1,C1,R1,L2,C2,R2,...
      - "grouped":     L1..LN,C1..CN,R1..RN
    """
    if module_volts_lcr is None or module_volts_lcr.size == 0:
        print("[Serial] No voltages to send.")
        return

    M = int(module_volts_lcr.shape[0])
    M_use = min(max_module, M)

    arr = np.zeros((max_module, 3), dtype=np.float32)
    arr[:M_use, :] = module_volts_lcr[:M_use, :]
    arr = np.clip(arr, 0.0, 5.0)

    if order == "grouped":
        L = arr[:, 0].tolist()
        C = arr[:, 1].tolist()
        R = arr[:, 2].tolist()
        volts = L + C + R
    else:
        volts = []
        for i in range(max_module):
            volts += [float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2])]

    cmd = "VOLT " + ",".join(f"{v:.1f}" for v in volts) + "\n"
    print(f"[Serial] -> {cmd.strip()}")
    ser.write(cmd.encode())

    while True:
        resp = ser.readline().decode(errors="ignore").strip()
        if resp:
            print(f"[Serial] <- {resp}")
        if resp == "APPLIED":
            break
    print("[Serial] Voltages applied.")


def load_roi_config(json_path: str):
    with open(json_path, "r") as f:
        roi = json.load(f)
    return roi["x"], roi["y"], roi["w"], roi["h"]


def crop_with_roi(img, x, y, w, h):
    H, W = img.shape[:2]
    if x >= W or y >= H:
        return np.empty((0, 0, 3), dtype=img.dtype)
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=img.dtype)
    return img[y1:y2, x1:x2].copy()


def make_red_tint(img_bgr):
    red = img_bgr.copy()
    red[:, :, 0] = 0
    red[:, :, 1] = 0
    return red


def make_blue_tint(img_bgr):
    blue = img_bgr.copy()
    blue[:, :, 1] = 0
    blue[:, :, 2] = 0
    return blue


def compute_mse(img1_bgr, img2_bgr):
    a = img1_bgr.astype(np.float32)
    b = img2_bgr.astype(np.float32)
    return float(np.mean((a - b) ** 2))


def compute_mse_and_overlay(img_target_roi, cam_full_frame, step_idx, base_dir):
    x, y, w, h = load_roi_config(ROI_CONFIG_FULL)
    roi_cam = crop_with_roi(cam_full_frame, x, y, w, h)

    if roi_cam.size == 0:
        print(f"[WARN] step {step_idx}: roi_cam is empty, MSE=NaN")
        return float("nan"), None

    H_t, W_t = img_target_roi.shape[:2]
    roi_cam_resized = cv2.resize(roi_cam, (W_t, H_t))

    mse = compute_mse(img_target_roi, roi_cam_resized)

    red  = make_red_tint(roi_cam_resized)
    blue = make_blue_tint(img_target_roi)
    overlay = cv2.addWeighted(red, 0.5, blue, 0.5, 0)

    timeline_dir = os.path.join(base_dir, "timeline_overlays")
    os.makedirs(timeline_dir, exist_ok=True)

    overlay_path = os.path.join(timeline_dir, f"overlay_step_{step_idx:03d}.png")
    cv2.imwrite(overlay_path, overlay)

    print(f"[Timeline] step {step_idx}: MSE={mse:.3f}, overlay saved -> {overlay_path}")
    return mse, overlay_path


def recolor_from_rb(arr_rgb: np.ndarray, gamma=0.8, gain_only=1.2, gain_overlap=1.0) -> np.ndarray:
    """
    加算済みRGBから、R/Bの共通成分を推定して
      - R only   -> Red
      - B only   -> Blue
      - overlap  -> Green
    に再配色する。
    """
    x = arr_rgb.astype(np.float32)
    R, G, B = x[..., 0], x[..., 1], x[..., 2]
    overlap = np.minimum(R, B)
    r_only = np.clip(R - overlap, 0, 255)
    b_only = np.clip(B - overlap, 0, 255)

    r_only = np.clip(r_only * gain_only, 0, 255)
    b_only = np.clip(b_only * gain_only, 0, 255)
    overlap = np.clip(overlap * gain_overlap, 0, 255)

    out = np.stack([r_only, overlap, b_only], axis=-1)  # (H,W,3)
    out01 = np.clip(out / 255.0, 0.0, 1.0)
    out01 = out01 ** gamma
    return (out01 * 255.0).astype(np.uint8)


# =========================================================
#  ★ mse_ref（step0）基準版：MSEプロット＋GIF＋CSV
# =========================================================

def create_mse_plot_and_gif_global_ref(
    mse_list,
    overlay_paths,
    save_dir,
    global_ref,
    gif_duration=1.0,
    recolor=True,
    gamma: float = 0.8,
    gain_only: float = 1.3,
    gain_overlap: float = 0.9,
    save_csv: bool = True,
    exclude_step0: bool = True,
):
    """
    global_ref を全ステップに共通適用:
      mse_norm = mse_raw / global_ref
    ※step0は「見た目上」外す（基準には使わない）
    """
    os.makedirs(save_dir, exist_ok=True)

    mse_arr = np.asarray(mse_list, dtype=np.float32)
    if mse_arr.size == 0:
        print("[Timeline] empty mse_list, skip.")
        return

    # 表示対象（exclude_step0なら1:）
    idx0 = 1 if (exclude_step0 and len(mse_arr) >= 2) else 0

    steps = list(range(len(mse_arr)))[idx0:]
    dt = WAIT_TIME + LOOP_WAIT
    time_axis = [s * dt for s in steps]

    mse_plot = mse_arr[idx0:]

    global_ref = float(max(global_ref, 1e-8))
    mse_norm = (mse_plot / global_ref).astype(np.float32)

    # --- raw step ---
    plt.figure(figsize=(6, 4))
    plt.plot(steps, mse_plot.tolist(), marker="o")
    plt.xlabel("Time step")
    plt.ylabel("MSE (target ROI vs camera ROI)")
    #plt.title("MSE over time")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(0, NUM_LOOP)
    plt.tight_layout()
    path_step = os.path.join(save_dir, "mse_time_series.png")
    plt.savefig(path_step)
    plt.close()

    # --- raw time ---
    plt.figure(figsize=(6, 4))
    plt.plot(time_axis, mse_plot.tolist(), marker="o")
    plt.xlabel("Time [s]")
    plt.ylabel("MSE (target ROI vs camera ROI)")
    #plt.title("MSE over time (sec)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(0, NUM_LOOP * dt)
    plt.tight_layout()
    path_time = os.path.join(save_dir, "mse_time_series_time.png")
    plt.savefig(path_time)
    plt.close()

    # --- norm (global_ref) step ---
    plt.figure(figsize=(6, 4))
    plt.plot(steps, mse_norm.tolist(), marker="o")
    plt.xlabel("Time step")
    plt.ylabel("Normalized MSE")
    #plt.title("Normalized MSE over steps (global_ref)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.ylim(bottom=0.0)
    plt.xlim(0, NUM_LOOP)
    plt.tight_layout()
    path_step_n = os.path.join(save_dir, "mse_time_series_norm_global_ref.png")
    plt.savefig(path_step_n)
    plt.close()

    # --- norm (global_ref) time ---
    plt.figure(figsize=(6, 4))
    plt.plot(time_axis, mse_norm.tolist(), marker="o")
    plt.xlabel("Time [s]")
    plt.ylabel("Normalized MSE")
    #plt.title("Normalized MSE over time (global_ref)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.ylim(bottom=0.0)
    plt.xlim(0, NUM_LOOP * dt)
    plt.tight_layout()
    path_time_n = os.path.join(save_dir, "mse_time_series_time_norm_global_ref.png")
    plt.savefig(path_time_n)
    plt.close()

    # --- CSV ---
    if save_csv:
        csv_path = os.path.join(save_dir, "mse_time_series.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time_sec", "step", "mse_raw", "global_ref", "mse_norm_global_ref"])
            for s, t, m, mn in zip(steps, time_axis, mse_plot.tolist(), mse_norm.tolist()):
                w.writerow([t, s, m, global_ref, mn])
        print(f"[Timeline] MSE CSV saved -> {csv_path}")

    # =========================
    # GIF（再配色→固定パレットで量子化）
    # =========================
    frames_rgb = []
    # overlay_paths も同じように idx0 以降のみ対象
    for p in overlay_paths[idx0:]:
        if p is None:
            continue
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if recolor:
            img_rgb = recolor_from_rb(
                img_rgb,
                gamma=gamma,
                gain_only=gain_only,
                gain_overlap=gain_overlap,
            )
        frames_rgb.append(Image.fromarray(img_rgb, mode="RGB"))

    if not frames_rgb:
        print("[Timeline] No frames for GIF, skip.")
        return

    gif_path = os.path.join(save_dir, "overlay_time_series.gif")
    frame_ms = int(gif_duration * 1000)

    p0 = frames_rgb[0].convert("P", palette=Image.ADAPTIVE, colors=256)
    frames_p = [p0]
    for fr in frames_rgb[1:]:
        frames_p.append(fr.quantize(palette=p0))

    frames_p[0].save(
        gif_path,
        save_all=True,
        append_images=frames_p[1:],
        duration=frame_ms,
        loop=0,
        disposal=2,
        optimize=False,
        format="GIF",
    )
    print(f"[Timeline] GIF saved -> {gif_path} (frames={len(frames_p)})")


# =========================================================
#  ★ trial集約：mean±std（帯） + CSV(long/summary)
# =========================================================

def plot_mean_std_band(save_path_png, x, mean, std, xlabel, ylabel, title, xlim=None):
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)

    plt.figure(figsize=(6, 4))
    plt.plot(x, mean, marker="o")
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.tight_layout()
    plt.savefig(save_path_png)
    plt.close()

def save_mse_trials_long_csv_global_ref(save_path, mse_trials, dt, meta, global_ref, exclude_step0=True):
    """
    long形式：各行が (trial_id, step) の1点
    mse_norm = mse_raw / global_ref で統一
    """
    n_trials = len(mse_trials)
    if n_trials == 0:
        return

    T = min(len(x) for x in mse_trials)
    idx0 = 1 if (exclude_step0 and T >= 2) else 0

    global_ref = float(max(global_ref, 1e-8))

    fieldnames = [
        "target_name", "max_module", "k_gain", "num_loop", "wait_time", "loop_wait", "reset_each_loop",
        "trial_id", "step", "time_sec",
        "mse_raw", "global_ref", "mse_norm_global_ref"
    ]

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for tr in range(n_trials):
            arr = np.asarray(mse_trials[tr][:T], dtype=np.float32)
            for s in range(idx0, T):
                w.writerow({
                    "target_name": meta.get("target_name", ""),
                    "max_module": meta.get("max_module", ""),
                    "k_gain": meta.get("k_gain", ""),
                    "num_loop": meta.get("num_loop", ""),
                    "wait_time": meta.get("wait_time", ""),
                    "loop_wait": meta.get("loop_wait", ""),
                    "reset_each_loop": meta.get("reset_each_loop", ""),
                    "trial_id": tr,
                    "step": s,
                    "time_sec": float(s * dt),
                    "mse_raw": float(arr[s]),
                    "global_ref": float(global_ref),
                    "mse_norm_global_ref": float(arr[s] / global_ref),
                })


def save_mse_summary_csv_global_ref(save_path, steps, times, mean_raw, std_raw, mean_norm, std_norm, meta, global_ref):
    fieldnames = [
        "target_name", "max_module", "k_gain", "num_loop", "wait_time", "loop_wait", "reset_each_loop",
        "global_ref",
        "step", "time_sec",
        "mse_mean", "mse_std",
        "mse_norm_mean", "mse_norm_std"
    ]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s, t, mr, sr, mn, sn in zip(steps, times, mean_raw, std_raw, mean_norm, std_norm):
            w.writerow({
                "target_name": meta.get("target_name", ""),
                "max_module": meta.get("max_module", ""),
                "k_gain": meta.get("k_gain", ""),
                "num_loop": meta.get("num_loop", ""),
                "wait_time": meta.get("wait_time", ""),
                "loop_wait": meta.get("loop_wait", ""),
                "reset_each_loop": meta.get("reset_each_loop", ""),
                "global_ref": float(global_ref),
                "step": int(s),
                "time_sec": float(t),
                "mse_mean": float(mr),
                "mse_std": float(sr),
                "mse_norm_mean": float(mn),
                "mse_norm_std": float(sn),
            })

# =========================================================
#  3D seg mask: (3,H,W) を作る（Detectron2 生mask）
# =========================================================

CLASS_LEFT = 0
CLASS_CENTER = 1
CLASS_RIGHT = 2

COLOR_LEFT_BGR   = (0, 0, 255)   # Red
COLOR_CENTER_BGR = (0, 255, 0)   # Green
COLOR_RIGHT_BGR  = (255, 0, 0)   # Blue


def masks3ch_from_segnet(img_bgr, seg_predictor, score_thresh=0.0, assign_mode="class"):
    """
    assign_mode:
      - "class": pred_classes で left/center/right を決める
      - "xsort": 重心xで 左→中央→右 に並べて ch=0/1/2 に割り当てる
    Returns:
      ok: bool
      mask3: (3,H,W) float32 in {0,1}
      overlay: BGR overlay image
      info: dict (debug)
    """
    out = seg_predictor(img_bgr)
    inst = out["instances"].to("cpu")
    H, W = img_bgr.shape[:2]

    mask3 = np.zeros((3, H, W), dtype=np.float32)

    if len(inst) == 0:
        return False, mask3, img_bgr.copy(), {"reason": "no_instances"}

    scores = inst.scores.numpy()
    classes = inst.pred_classes.numpy()
    masks = inst.pred_masks.numpy().astype(bool)

    if assign_mode == "class":
        for cls_id, ch in [(CLASS_LEFT, 0), (CLASS_CENTER, 1), (CLASS_RIGHT, 2)]:
            idxs = np.where(classes == cls_id)[0]
            if idxs.size == 0:
                continue
            best_i = idxs[np.argmax(scores[idxs])]
            if scores[best_i] < score_thresh:
                continue
            mask3[ch] = np.maximum(mask3[ch], masks[best_i].astype(np.float32))

    elif assign_mode == "xsort":
        keep = np.where(scores >= score_thresh)[0]
        if keep.size == 0:
            return False, mask3, img_bgr.copy(), {"reason": "all_below_thresh"}

        items = []
        for i in keep.tolist():
            m = masks[i]
            if not m.any():
                continue
            ys, xs = np.where(m)
            cx = float(xs.mean())
            area = int(m.sum())
            items.append((cx, scores[i], area, i))

        if len(items) == 0:
            return False, mask3, img_bgr.copy(), {"reason": "no_valid_masks"}

        items = [t for t in items if t[2] >= 20]
        if len(items) == 0:
            return False, mask3, img_bgr.copy(), {"reason": "all_too_small"}

        items = sorted(items, key=lambda t: t[1], reverse=True)[:3]
        items = sorted(items, key=lambda t: t[0])

        for ch, (_, _, _, i) in enumerate(items):
            mask3[ch] = np.maximum(mask3[ch], masks[i].astype(np.float32))

    else:
        raise ValueError(f"unknown assign_mode: {assign_mode}")

    overlay = img_bgr.copy().astype(np.float32)
    alpha = 0.5

    mL = mask3[0] > 0.5
    mC = mask3[1] > 0.5
    mR = mask3[2] > 0.5

    if mL.any():
        overlay[mL] = overlay[mL] * (1 - alpha) + np.array(COLOR_LEFT_BGR, np.float32) * alpha
    if mC.any():
        overlay[mC] = overlay[mC] * (1 - alpha) + np.array(COLOR_CENTER_BGR, np.float32) * alpha
    if mR.any():
        overlay[mR] = overlay[mR] * (1 - alpha) + np.array(COLOR_RIGHT_BGR, np.float32) * alpha

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    info = {
        "n_instances": int(len(inst)),
        "present_left": bool(mL.any()),
        "present_center": bool(mC.any()),
        "present_right": bool(mR.any()),
        "assign_mode": assign_mode,
    }
    return True, mask3.astype(np.float32), overlay, info


# =========================================================
#  bbox → center crop（範囲外は黒+ノイズ補完）
# =========================================================

MEAN_BGR = np.array([128.04, 131.38, 132.08], dtype=np.float32)
STD_BGR  = np.array([  6.56,   6.47,   5.51], dtype=np.float32)
BLEND_WIDTH = 5.0

CROP_H = 70
CROP_W = 70


def sample_border_noise(H, W):
    noise = np.random.normal(MEAN_BGR, STD_BGR, size=(H, W, 3))
    return np.clip(noise, 0, 255).astype(np.uint8)


def make_center_crops_in_memory(img_bgr, boxes, crop_h=CROP_H, crop_w=CROP_W):
    H, W, _ = img_bgr.shape
    boxes_sorted = sorted(boxes, key=lambda b: b[1])  # top->bottom
    half_h = crop_h // 2
    half_w = crop_w // 2
    crops = []

    for box in boxes_sorted:
        x1, y1, x2, y2 = [int(v) for v in box]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        x1c, x2c = cx - half_w, cx + half_w
        y1c, y2c = cy - half_h, cy + half_h

        cropped = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        mask    = np.zeros((crop_h, crop_w), dtype=np.uint8)

        x1_src, y1_src = max(0, x1c), max(0, y1c)
        x2_src, y2_src = min(W, x2c), min(H, y2c)

        x1_dst = x1_src - x1c
        y1_dst = y1_src - y1c
        x2_dst = x1_dst + (x2_src - x1_src)
        y2_dst = y1_dst + (y2_src - y1_src)

        if x2_src > x1_src and y2_src > y1_src:
            cropped[y1_dst:y2_dst, x1_dst:x2_dst] = img_bgr[y1_src:y2_src, x1_src:x2_src]
            mask[y1_dst:y2_dst, x1_dst:x2_dst] = 255

        noise = sample_border_noise(crop_h, crop_w)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        alpha = np.clip(dist / BLEND_WIDTH, 0.0, 1.0).astype(np.float32)
        alpha_3 = alpha[..., None]
        blended = cropped.astype(np.float32) * alpha_3 + noise.astype(np.float32) * (1 - alpha_3)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        crops.append(blended)

    return crops


# =========================================================
#  signals.csv（ヘッダーあり）から target voltages (M,3) を作る
# =========================================================

def load_target_voltages_from_signals(signals_csv_path: Path, img_name_noext: str, n_modules: int):
    """
    signals.csv:
      header: L1,C1,R1,L2,C2,R2,... upto modules
      n.png -> row index = n-1 (because header row is 0)
    """
    if not signals_csv_path.exists():
        print(f"[INFO] signals.csv not found: {signals_csv_path}")
        return None

    try:
        n = int(img_name_noext)  # "12" -> 12
    except ValueError:
        print(f"[WARN] target filename is not integer-like: {img_name_noext}. Skip target_voltages.")
        return None

    df = pd.read_csv(signals_csv_path, header=0)
    row_idx = n - 1
    if row_idx < 0 or row_idx >= len(df):
        print(f"[WARN] row_idx out of range: {row_idx} (n={n})")
        return None

    row = df.iloc[row_idx]

    M = min(int(n_modules), MAX_MODULE)
    out = np.zeros((M, 3), dtype=np.float32)

    for m in range(1, M + 1):
        colL = f"L{m}"
        colC = f"C{m}"
        colR = f"R{m}"
        if colL in df.columns:
            out[m-1, 0] = float(row[colL]) if pd.notna(row[colL]) else 0.0
        if colC in df.columns:
            out[m-1, 1] = float(row[colC]) if pd.notna(row[colC]) else 0.0
        if colR in df.columns:
            out[m-1, 2] = float(row[colR]) if pd.notna(row[colR]) else 0.0

    return out


# =========================================================
#  plot: pred vs target（3枚: L/C/R）
# =========================================================

def plot_voltage_bars_3ch(tv, pv, out_dir, step_idx):
    os.makedirs(out_dir, exist_ok=True)
    M = tv.shape[0]
    x = np.arange(1, M + 1)
    width = 0.35

    names = [
        ("left",   0, "Left"),
        ("center", 1, "Center"),
        ("right",  2, "Right"),
    ]

    for tag, ch, title in names:
        plt.figure(figsize=(6, 4))
        plt.bar(x - width/2, tv[:, ch], width=width, label="Target")
        plt.bar(x + width/2, pv[:, ch], width=width, label="Pred (clipped 0–5V)")
        plt.ylim(0.0, 5.0)
        plt.xticks(x, [str(i) for i in x])
        plt.xlabel("Module index")
        plt.ylabel("Voltage [V]")
        plt.title(f"{title} (N={step_idx})")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"voltage_bar_{tag}_step{step_idx:02d}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()


# =========================================================
#  Feedback loop (3D)
# =========================================================

def run_feedback_loop_3d(
    img_target_roi,
    boxes_target,
    initial_voltages,     # (M,3) start voltages
    first_cam_frame,
    dev_predictor,
    seg3_predictor,
    ik_model,
    device,
    ser,
    base_save_dir,
    num_loops=10,
    wait_sec=5.0,
    start_step_idx=2,
    use_reset_between=True,
    K_gain=1.5,
    reset_wait_sec=3.0,
    cap=None,
    target_voltages=None,  # (M,3) or None
):
    print("[FB-IK-3D] Preparing target center crops & 3ch masks ...")
    target_crops = make_center_crops_in_memory(img_target_roi, boxes_target, crop_h=CROP_H, crop_w=CROP_W)
    num_modules = len(target_crops)
    if num_modules == 0:
        print("[FB-IK-3D] No target modules found. Abort.")
        return [], [], []

    num_modules = min(num_modules, MAX_MODULE)

    target_masks3 = []
    targ_dbg_dir = os.path.join(base_save_dir, "fb_target_debug")
    os.makedirs(targ_dbg_dir, exist_ok=True)

    for m_idx in range(num_modules):
        crop = target_crops[m_idx]
        ok, mask3, overlay, info = masks3ch_from_segnet(crop, seg3_predictor, assign_mode="xsort")
        if not ok:
            print(f"[FB-IK-3D] failed target seg for module {m_idx}, abort.")
            return [], [], []
        target_masks3.append(mask3)

        cv2.imwrite(os.path.join(targ_dbg_dir, f"mod{m_idx}_targ_crop.png"), crop)
        cv2.imwrite(os.path.join(targ_dbg_dir, f"mod{m_idx}_targ_overlay.png"), overlay)
        np.save(os.path.join(targ_dbg_dir, f"mod{m_idx}_targ_mask3.npy"), mask3)

    mse_list_fb = []
    overlay_paths_fb = []
    bar_paths_fb = []

    cam_frame = first_cam_frame.copy()
    x_roi, y_roi, w_roi, h_roi = load_roi_config(ROI_CONFIG_FULL)

    current_voltages = np.array(initial_voltages, dtype=np.float32).copy()
    if current_voltages.shape[0] < num_modules:
        pad = np.zeros((num_modules - current_voltages.shape[0], 3), dtype=np.float32)
        current_voltages = np.vstack([current_voltages, pad])
    elif current_voltages.shape[0] > num_modules:
        current_voltages = current_voltages[:num_modules, :]
    current_voltages = np.clip(current_voltages, 0.0, 5.0)

    current_step_idx = start_step_idx

    for loop in range(num_loops):
        print(f"\n[FB-IK-3D] ===== Feedback loop {loop+1}/{num_loops} =====")

        fb_dir = os.path.join(base_save_dir, f"fb_step_{loop+1:02d}")
        os.makedirs(fb_dir, exist_ok=True)

        cv2.imwrite(os.path.join(fb_dir, "captured_full.png"), cam_frame)

        if use_reset_between:
            print("[FB-IK-3D] Resetting channels to 0.0V ...")
            reset_voltages_to_zero(ser, n_channels=MAX_MODULE * 3)
            print(f"[FB-IK-3D] Waiting {reset_wait_sec} sec after reset ...")
            time.sleep(reset_wait_sec)

        roi_cam = crop_with_roi(cam_frame, x_roi, y_roi, w_roi, h_roi)
        if roi_cam.size == 0:
            print("[FB-IK-3D] ROI from camera is empty, break.")
            break
        cv2.imwrite(os.path.join(fb_dir, "roi_captured_raw.png"), roi_cam)

        H_t, W_t = img_target_roi.shape[:2]
        roi_cam_resized = cv2.resize(roi_cam, (W_t, H_t))
        overlay_local = cv2.addWeighted(
            make_red_tint(roi_cam_resized), 0.5,
            make_blue_tint(img_target_roi), 0.5,
            0
        )

        overlay_rgb = cv2.cvtColor(overlay_local, cv2.COLOR_BGR2RGB)
        overlay_rgb = recolor_from_rb(overlay_rgb, gamma=0.8, gain_only=1.3, gain_overlap=0.9)
        overlay_local_recolor = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(fb_dir, "roi_overlay_red_orig_blue_cap.png"), overlay_local_recolor)

        out_cam = dev_predictor(roi_cam)
        inst_cam = out_cam["instances"].to("cpu")
        if len(inst_cam) == 0:
            print("[FB-IK-3D] No detections in camera ROI, break.")
            break

        boxes_cam = inst_cam.pred_boxes.tensor.numpy()
        masks_cam = inst_cam.pred_masks.numpy()

        vis_cam = Visualizer(roi_cam[:, :, ::-1], metadata=None, scale=1.0)
        out_vis_cam = vis_cam.overlay_instances(masks=masks_cam, boxes=None, labels=None)
        cv2.imwrite(os.path.join(fb_dir, "cam_devnet_vis.png"),
                    out_vis_cam.get_image()[:, :, ::-1].astype("uint8"))

        cam_crops = make_center_crops_in_memory(roi_cam, boxes_cam, crop_h=CROP_H, crop_w=CROP_W)
        eff = min(len(cam_crops), num_modules)

        V_next_raw = np.zeros((num_modules, 3), dtype=np.float32)
        V_next_clipped = np.zeros((num_modules, 3), dtype=np.float32)

        for m_idx in range(eff):
            cam_crop = cam_crops[m_idx]
            targ_mask3 = target_masks3[m_idx]

            ok, cam_mask3, cam_overlay, info = masks3ch_from_segnet(cam_crop, seg3_predictor, assign_mode="xsort")
            if not ok:
                print(f"[FB-IK-3D] seg failed cam module {m_idx}, skip.")
                continue

            cv2.imwrite(os.path.join(fb_dir, f"mod{m_idx}_cam_crop.png"), cam_crop)
            cv2.imwrite(os.path.join(fb_dir, f"mod{m_idx}_cam_overlay.png"), cam_overlay)
            np.save(os.path.join(fb_dir, f"mod{m_idx}_cam_mask3.npy"), cam_mask3)
            np.save(os.path.join(fb_dir, f"mod{m_idx}_targ_mask3.npy"), targ_mask3)

            v_i = current_voltages[m_idx].astype(np.float32)
            Hm, Wm = cam_mask3.shape[1], cam_mask3.shape[2]

            q_map = np.zeros((3, Hm, Wm), dtype=np.float32)
            q_map[0, :, :] = v_i[0] / 5.0
            q_map[1, :, :] = v_i[1] / 5.0
            q_map[2, :, :] = v_i[2] / 5.0

            x = np.concatenate([cam_mask3, targ_mask3, q_map], axis=0)  # (9,H,W)
            x_t = torch.from_numpy(x).unsqueeze(0).to(device)  # (1,9,H,W)

            with torch.no_grad():
                y_hat = ik_model(x_t)  # (1,3)

            v_ik = y_hat[0].detach().cpu().numpy().astype(np.float32)

            dV = v_ik - v_i
            v_next_raw = v_i + K_gain * dV
            v_next_clip = np.clip(v_next_raw, 0.0, 5.0)

            V_next_raw[m_idx] = v_next_raw
            V_next_clipped[m_idx] = v_next_clip

            print(
                f"[FB-IK-3D] Module {m_idx}: "
                f"V_i=({v_i[0]:.3f},{v_i[1]:.3f},{v_i[2]:.3f}) "
                f"V_ik=({v_ik[0]:.3f},{v_ik[1]:.3f},{v_ik[2]:.3f}) "
                f"V_next_clip=({v_next_clip[0]:.3f},{v_next_clip[1]:.3f},{v_next_clip[2]:.3f})"
            )

        np.savetxt(os.path.join(fb_dir, "ik_pred_voltages_raw.txt"), V_next_raw, fmt="%.4f")
        np.savetxt(os.path.join(fb_dir, "ik_pred_voltages_clipped.txt"), V_next_clipped, fmt="%.4f")

        if target_voltages is not None:
            tv = np.array(target_voltages, dtype=np.float32)[:num_modules, :]
            pv = V_next_clipped[:num_modules, :]

            valid = np.any(tv > 0.0, axis=1)
            tvp = tv[valid]
            pvp = pv[valid]
            if tvp.shape[0] > 0:
                plot_voltage_bars_3ch(tvp, pvp, out_dir=fb_dir, step_idx=loop+1)
                bar_paths_fb.append(os.path.join(fb_dir, f"voltage_bar_left_step{loop+1:02d}.png"))

        send_voltages_to_arduino_3d(ser, V_next_clipped, max_module=MAX_MODULE, order="interleave")
        current_voltages = V_next_clipped.copy()

        print(f"[FB-IK-3D] Waiting {wait_sec} sec ...")
        time.sleep(wait_sec)

        cam_next = capture_frame(cap)
        if cam_next is None:
            print("[FB-IK-3D] capture failed, break.")
            break

        mse_t, overlay_t = compute_mse_and_overlay(img_target_roi, cam_next, current_step_idx, base_save_dir)
        mse_list_fb.append(mse_t)
        overlay_paths_fb.append(overlay_t)
        current_step_idx += 1

        with open(os.path.join(base_save_dir, "mse_each_step.txt"), "a", encoding="utf-8") as f:
            f.write(f"{loop+1},{mse_t:.6f}\n")

        cam_frame = cam_next.copy()

    return mse_list_fb, overlay_paths_fb, bar_paths_fb


# =========================================================
#  main
# =========================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # =========================
    # Camera
    # =========================
    cap = open_camera(CAMERA_INDEX, width=CAM_W, height=CAM_H)
    if cap is None:
        print("[ERROR] Could not open camera.")
        return

    # =========================
    # devnet predictor
    # =========================
    try:
        register_coco_instances("DEV_TRAIN", {}, DEVNET_TRAIN_JSON, DEVNET_TRAIN_IMAGES)
    except Exception:
        pass

    cfg_dev = get_cfg()
    cfg_dev.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg_dev.MODEL.WEIGHTS = DEVNET_WEIGHT
    cfg_dev.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg_dev.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg_dev.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dev.DATASETS.TEST = ()
    dev_predictor = DefaultPredictor(cfg_dev)

    # =========================
    # seg-net predictor
    # =========================
    try:
        register_coco_instances("SEG3_TRAIN", {}, SEG3_TRAIN_JSON, SEG3_TRAIN_IMAGES)
    except Exception:
        pass

    cfg_seg = get_cfg()
    cfg_seg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg_seg.MODEL.WEIGHTS = SEG3_WEIGHT
    cfg_seg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg_seg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg_seg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_seg.DATASETS.TEST = ()
    seg3_predictor = DefaultPredictor(cfg_seg)

    # =========================
    # IK model
    # =========================
    ik_model = IKBeamNet(in_ch=9, feat_dim=128).to(device)
    ckpt = torch.load(IK_MODEL_PATH, map_location=device)
    ik_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    ik_model.eval()

    # =========================
    # Serial
    # =========================
    ser = init_serial(SERIAL_PORT, BAUDRATE)

    # =========================
    # 保存先
    # =========================
    os.makedirs(SAVE_ROOT, exist_ok=True)

    # =========================
    # 集約用
    # =========================
    mse_trials = []        # list[list[float]]
    trial_metas = []      # target名など
    REF_STEP = 1           # step=1 を初期
    if RESET_EACH_LOOP:
        dt = WAIT_TIME + LOOP_WAIT
    else:
        dt = LOOP_WAIT

    # ==================================================
    # trial loop（targetごと）
    # ==================================================
    for trial in range(N_TRIALS):
        print("\n============================")
        print(f"[TRIAL] {trial+1}/{N_TRIALS}")
        print("============================")

        # -------- target選択 --------
        img_path = select_image_via_dialog()
        if img_path is None:
            print("[INFO] Target selection cancelled. Skip.")
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print("[WARN] Failed to read target image. Skip.")
            continue

        target_name = os.path.basename(img_path)
        img_stem = os.path.splitext(target_name)[0]

        trial_dir = os.path.join(SAVE_ROOT, f"trial_{trial:02d}_{img_stem}")
        os.makedirs(trial_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(trial_dir, f"{img_stem}_roi_input.png"),
            img_bgr,
        )

        # -------- devnet --------
        out = dev_predictor(img_bgr)
        inst = out["instances"].to("cpu")
        if len(inst) == 0:
            print("[WARN] No detection in target image. Skip.")
            continue

        boxes = inst.pred_boxes.tensor.numpy()
        n_mod = min(len(boxes), MAX_MODULE)

        # -------- target voltages（任意） --------
        target_voltages = None
        try:
            img_p = Path(img_path)
            dataset_dir = img_p.parent.parent
            signals_path = dataset_dir / "signals.csv"
            target_voltages = load_target_voltages_from_signals(
                signals_path,
                img_stem,
                n_modules=n_mod,
            )
        except Exception:
            target_voltages = None

        # -------- step0 / step1 --------
        timeline_mse = []
        timeline_overlays = []

        cam0 = capture_frame(cap)
        mse0, ov0 = compute_mse_and_overlay(
            img_bgr, cam0, step_idx=0, base_dir=trial_dir
        )
        timeline_mse.append(mse0)
        timeline_overlays.append(ov0)

        reset_voltages_to_zero(ser, n_channels=MAX_MODULE * 3)
        time.sleep(WAIT_TIME)

        cam1 = capture_frame(cap)
        mse1, ov1 = compute_mse_and_overlay(
            img_bgr, cam1, step_idx=1, base_dir=trial_dir
        )
        timeline_mse.append(mse1)
        timeline_overlays.append(ov1)

        # -------- feedback loop --------
        mse_fb, ov_fb, _ = run_feedback_loop_3d(
            img_target_roi=img_bgr,
            boxes_target=boxes,
            initial_voltages=np.zeros((n_mod, 3), dtype=np.float32),
            first_cam_frame=cam1,
            dev_predictor=dev_predictor,
            seg3_predictor=seg3_predictor,
            ik_model=ik_model,
            device=device,
            ser=ser,
            base_save_dir=trial_dir,
            num_loops=NUM_LOOP,
            wait_sec=LOOP_WAIT,
            start_step_idx=2,
            use_reset_between=RESET_EACH_LOOP,
            K_gain=K_GAIN,
            reset_wait_sec=WAIT_TIME,
            cap=cap,
            target_voltages=target_voltages,
        )

        timeline_mse.extend(mse_fb)
        timeline_overlays.extend(ov_fb)

        mse_trials.append(timeline_mse)
        trial_metas.append(
            dict(
                trial_id=trial,
                target_name=target_name,
            )
        )

        reset_voltages_to_zero(ser, n_channels=MAX_MODULE * 3)

    # ==================================================
    # 集約（trialごとref = step1）
    # ==================================================
    if len(mse_trials) < 2:
        print("[WARN] Not enough trials for aggregation.")
        return

    T = min(len(x) for x in mse_trials)
    mse_trials = [x[:T] for x in mse_trials]
    arr = np.asarray(mse_trials, dtype=np.float32)  # (N,T)

    idx0 = REF_STEP
    steps = np.arange(T)[idx0:]
    times = steps * dt

    # ---- ★ trialごと正規化 ----
    refs = arr[:, REF_STEP:REF_STEP+1]
    refs = np.maximum(refs, 1e-8)
    arr_norm = arr / refs

    mean_norm = arr_norm.mean(axis=0)[idx0:]
    std_norm  = arr_norm.std(axis=0, ddof=0)[idx0:]

    # ---- 表示用軸 ----
    steps_plot = steps - steps[0]
    times_plot = times - times[0]

    # ---- プロット ----
    plot_mean_std_band(
        os.path.join(SAVE_ROOT, "mse_mean_std_steps_norm.png"),
        steps_plot,
        mean_norm,
        std_norm,
        xlabel="Time step (from initial)",
        ylabel="Normalized MSE (per-trial ref)",
        title=f"MSE mean ± std over {len(arr_norm)} targets",
        xlim=(steps_plot[0], steps_plot[-1]),
    )

    plot_mean_std_band(
        os.path.join(SAVE_ROOT, "mse_mean_std_time_norm.png"),
        times_plot,
        mean_norm,
        std_norm,
        xlabel="Time [s] (from initial)",
        ylabel="Normalized MSE (per-trial ref)",
        title=f"MSE mean ± std over {len(arr_norm)} targets",
        xlim=(times_plot[0], times_plot[-1]),
    )

    # ==================================================
    # ★ CSV 出力（ここが前回抜けてた）
    # ==================================================
    # long
    with open(os.path.join(SAVE_ROOT, "mse_trials_long.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "trial_id", "target_name",
            "step", "time_sec",
            "mse_raw", "mse_norm",
        ])
        for i in range(arr.shape[0]):
            for s in range(idx0, T):
                w.writerow([
                    trial_metas[i]["trial_id"],
                    trial_metas[i]["target_name"],
                    s,
                    s * dt,
                    float(arr[i, s]),
                    float(arr_norm[i, s]),
                ])

    # summary
    with open(os.path.join(SAVE_ROOT, "mse_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "step", "time_sec",
            "mse_norm_mean", "mse_norm_std",
        ])
        for s, t, m, sd in zip(steps, times, mean_norm, std_norm):
            w.writerow([int(s), float(t), float(m), float(sd)])

    print(f"[INFO] All results saved in {SAVE_ROOT}")





if __name__ == "__main__":
    main()
