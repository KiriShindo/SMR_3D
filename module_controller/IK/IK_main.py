# -*- coding: utf-8 -*-
import os
import time
import json
import random
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn

# detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

# Arduino serial
from serial import Serial

# ★ 学習側で定義した 3D IKモデルを import
#   あなたが貼ってくれたクラス名は IKBeamNet なのでそれを使う
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

BASE_ROOT = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control_3D/IK_only_result_sorted"

# ------------------------
# 実験パラメータ
# ------------------------
K_GAIN = 1.0
NUM_LOOP = 10
RESET_EACH_LOOP = True
WAIT_TIME = 3.0
LOOP_WAIT = 5.0
MAX_MODULE = 1

# ------------------------
# 保存パス生成
# ------------------------
maxmod_dir = os.path.join(BASE_ROOT, f"maxmod_{MAX_MODULE}")

cond_name = (
    f"K{K_GAIN}"
    f"_LOOP{NUM_LOOP}"
    f"_RESET{int(RESET_EACH_LOOP)}"
    f"_WAIT{WAIT_TIME}"
    f"_LOOPWAIT{LOOP_WAIT}"
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
# ★あなたの3D segデータ/weightに置き換えてください
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
    print(f"[Camera] captured frame shape: {frame.shape}")
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
    cmd = "VOLT " + ",".join(f"{v:.1f}" for v in zeros) + "/n"
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
        # interleave
        volts = []
        for i in range(max_module):
            volts += [float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2])]

    cmd = "VOLT " + ",".join(f"{v:.1f}" for v in volts) + "/n"
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


def create_mse_plot_and_gif(mse_list, overlay_paths, save_dir, gif_duration=1.0):
    # ---- step軸(shift付き) ----
    steps = list(range(len(mse_list)))
    #steps_shifted = [s - 1 for s in steps]

    plt.figure(figsize=(6, 4))
    plt.plot(steps, mse_list, marker="o")
    plt.xlabel("Time step")
    plt.ylabel("MSE (target ROI vs camera ROI)")
    plt.title("MSE over time")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(0, NUM_LOOP)
    plt.tight_layout()
    path_step = os.path.join(save_dir, "mse_time_series.png")
    plt.savefig(path_step)
    plt.close()
    print(f"[Timeline] MSE plot saved -> {path_step}")

    # ---- time軸(shift付き) ----
    dt = WAIT_TIME + LOOP_WAIT
    time_axis = [s * dt for s in steps]

    plt.figure(figsize=(6, 4))
    plt.plot(time_axis, mse_list, marker="o")
    plt.xlabel("Time [s]")
    plt.ylabel("MSE (target ROI vs camera ROI)")
    plt.title("MSE over time (sec)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(0, NUM_LOOP * dt)
    plt.tight_layout()
    path_time = os.path.join(save_dir, "mse_time_series_time.png")
    plt.savefig(path_time)
    plt.close()
    print(f"[Timeline] MSE(time) plot saved -> {path_time}")

    # ---- GIF ----
    frames = []
    for p in overlay_paths:
        if p is None:
            continue
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(img_rgb))

    if not frames:
        print("[Timeline] No frames for GIF, skip.")
        return

    gif_path = os.path.join(save_dir, "overlay_time_series.gif")
    durations_ms = [int(gif_duration * 1000)] * len(frames)
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=durations_ms, loop=0)
    print(f"[Timeline] GIF saved -> {gif_path}")


# =========================================================
#  3D seg mask: (3,H,W) を作る（Detectron2 生mask）
# =========================================================

# クラスIDの想定: 0=left, 1=center, 2=right
CLASS_LEFT = 0
CLASS_CENTER = 1
CLASS_RIGHT = 2

COLOR_LEFT_BGR   = (0, 0, 255)   # Red
COLOR_CENTER_BGR = (0, 255, 0)   # Green
COLOR_RIGHT_BGR  = (255, 0, 0)   # Blue


def masks3ch_from_segnet(img_bgr, seg_predictor, score_thresh=0.0, assign_mode="class"):
    """
    assign_mode:
      - "class": 既存の方式（pred_classes で left/center/right を決める）
      - "xsort": 検出インスタンスの重心 x で 左→中央→右 に並べて ch=0/1/2 に割り当てる
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
        # 各クラスごとに「スコア最大の1枚」を採用
        for cls_id, ch in [(CLASS_LEFT, 0), (CLASS_CENTER, 1), (CLASS_RIGHT, 2)]:
            idxs = np.where(classes == cls_id)[0]
            if idxs.size == 0:
                continue
            best_i = idxs[np.argmax(scores[idxs])]
            if scores[best_i] < score_thresh:
                continue
            mask3[ch] = np.maximum(mask3[ch], masks[best_i].astype(np.float32))

    elif assign_mode == "xsort":
        # まず score_thresh を満たす候補を集める
        keep = np.where(scores >= score_thresh)[0]
        if keep.size == 0:
            return False, mask3, img_bgr.copy(), {"reason": "all_below_thresh"}

        items = []
        for i in keep.tolist():
            m = masks[i]
            if not m.any():
                continue
            ys, xs = np.where(m)
            cx = float(xs.mean())          # 重心x
            area = int(m.sum())
            items.append((cx, scores[i], area, i))

        if len(items) == 0:
            return False, mask3, img_bgr.copy(), {"reason": "no_valid_masks"}

        # ノイズ除去（小さすぎるmaskを落とす：閾値は必要なら調整）
        items = [t for t in items if t[2] >= 20]
        if len(items) == 0:
            return False, mask3, img_bgr.copy(), {"reason": "all_too_small"}

        # 3本を選ぶ：score上位3つ→cxで左→右に並べる（これが一番安定しやすい）
        items = sorted(items, key=lambda t: t[1], reverse=True)[:3]
        items = sorted(items, key=lambda t: t[0])

        # 左→右 を ch=0/1/2 に割り当て
        for ch, (_, _, _, i) in enumerate(items):
            mask3[ch] = np.maximum(mask3[ch], masks[i].astype(np.float32))

    else:
        raise ValueError(f"unknown assign_mode: {assign_mode}")

    # overlay（可視化専用）
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
#  bbox → center crop（2Dコードの挙動に合わせる：範囲外は黒+ノイズ補完）
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

    df = pd.read_csv(signals_csv_path, header=0)  # headerあり
    row_idx = n - 1
    if row_idx < 0 or row_idx >= len(df):
        print(f"[WARN] row_idx out of range: {row_idx} (n={n})")
        return None

    row = df.iloc[row_idx]

    # L1,C1,R1,L2,C2,R2,... を取り出して (M,3)
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
    """
    tv,pv: (M,3)
    save:
      voltage_bar_left_stepXX.png
      voltage_bar_center_stepXX.png
      voltage_bar_right_stepXX.png
    """
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
    initial_voltages,     # (M,3) start voltages (typically zeros)
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
    """
    3D版:
      input x = concat([mask_cam(3), mask_target(3), q_map(3)]) -> (9,H,W)
      output y_hat = (3,) voltage for next
      update: V_next_raw = V_i + K_gain*(V_ik - V_i), then clip 0..5
    """

    # ---- target crops & target masks3 を先に全部作る ----
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

    # ---- timeline buffers ----
    mse_list_fb = []
    overlay_paths_fb = []
    bar_paths_fb = []  # 3枚のうち代表（ここでは left のパスだけ溜める等）でもOK。今回は3枚全部作るのでdirだけ溜める。

    cam_frame = first_cam_frame.copy()
    x_roi, y_roi, w_roi, h_roi = load_roi_config(ROI_CONFIG_FULL)

    # 現在電圧 (num_modules,3)
    current_voltages = np.array(initial_voltages, dtype=np.float32).copy()
    if current_voltages.shape[0] < num_modules:
        pad = np.zeros((num_modules - current_voltages.shape[0], 3), dtype=np.float32)
        current_voltages = np.vstack([current_voltages, pad])
    elif current_voltages.shape[0] > num_modules:
        current_voltages = current_voltages[:num_modules, :]
    current_voltages = np.clip(current_voltages, 0.0, 5.0)

    current_step_idx = start_step_idx

    for loop in range(num_loops):
        print(f"/n[FB-IK-3D] ===== Feedback loop {loop+1}/{num_loops} =====")

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

        # target vs current ROI overlay (赤青)
        H_t, W_t = img_target_roi.shape[:2]
        roi_cam_resized = cv2.resize(roi_cam, (W_t, H_t))
        overlay_local = cv2.addWeighted(make_red_tint(roi_cam_resized), 0.5, make_blue_tint(img_target_roi), 0.5, 0)
        cv2.imwrite(os.path.join(fb_dir, "roi_overlay_red_orig_blue_cap.png"), overlay_local)

        # devnet detect modules on camera ROI
        out_cam = dev_predictor(roi_cam)
        inst_cam = out_cam["instances"].to("cpu")
        if len(inst_cam) == 0:
            print("[FB-IK-3D] No detections in camera ROI, break.")
            break

        boxes_cam = inst_cam.pred_boxes.tensor.numpy()
        masks_cam = inst_cam.pred_masks.numpy()
        num_cam_masks = masks_cam.shape[0]

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

            # 保存（デバッグ）
            cv2.imwrite(os.path.join(fb_dir, f"mod{m_idx}_cam_crop.png"), cam_crop)
            cv2.imwrite(os.path.join(fb_dir, f"mod{m_idx}_cam_overlay.png"), cam_overlay)
            np.save(os.path.join(fb_dir, f"mod{m_idx}_cam_mask3.npy"), cam_mask3)
            np.save(os.path.join(fb_dir, f"mod{m_idx}_targ_mask3.npy"), targ_mask3)

            v_i = current_voltages[m_idx].astype(np.float32)  # (3,)
            Hm, Wm = cam_mask3.shape[1], cam_mask3.shape[2]

            q_map = np.zeros((3, Hm, Wm), dtype=np.float32)
            q_map[0, :, :] = v_i[0] / 5.0
            q_map[1, :, :] = v_i[1] / 5.0
            q_map[2, :, :] = v_i[2] / 5.0

            x = np.concatenate([cam_mask3, targ_mask3, q_map], axis=0)  # (9,H,W)

            x_t = torch.from_numpy(x).unsqueeze(0).to(device)  # (1,9,H,W)
            with torch.no_grad():
                y_hat = ik_model(x_t)  # (1,3)

            v_ik = y_hat[0].detach().cpu().numpy().astype(np.float32)  # (3,)
            v_ik = v_ik  # 生値

            # feedback
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

        # 保存
        np.savetxt(os.path.join(fb_dir, "ik_pred_voltages_raw.txt"), V_next_raw, fmt="%.4f")
        np.savetxt(os.path.join(fb_dir, "ik_pred_voltages_clipped.txt"), V_next_clipped, fmt="%.4f")

        # pred vs target bars (3枚)
        if target_voltages is not None:
            tv = np.array(target_voltages, dtype=np.float32)
            tv = tv[:num_modules, :]
            pv = V_next_clipped[:num_modules, :]

            # signals.csv 仕様で存在しないモジュールは0の可能性があるので除外（全部0の行）
            valid = np.any(tv > 0.0, axis=1)
            tvp = tv[valid]
            pvp = pv[valid]
            if tvp.shape[0] > 0:
                plot_voltage_bars_3ch(tvp, pvp, out_dir=fb_dir, step_idx=loop+1)
                # 代表として left だけ溜める（GIF作るならここで3つ作るのも可）
                bar_paths_fb.append(os.path.join(fb_dir, f"voltage_bar_left_step{loop+1:02d}.png"))

        # Arduino 送信（clip後）
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

        # MSEログ
        with open(os.path.join(base_save_dir, "mse_each_step.txt"), "a") as f:
            f.write(f"{loop+1},{mse_t:.6f}/n")

        cam_frame = cam_next.copy()

    return mse_list_fb, overlay_paths_fb, bar_paths_fb


# =========================================================
#  main
# =========================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # camera open
    cap = open_camera(CAMERA_INDEX, width=CAM_W, height=CAM_H)
    if cap is None:
        print("[ERROR] Could not open camera. Abort.")
        return

    # t=0 initial camera frame
    print("[INFO] Capturing initial camera frame (t=0) ...")
    initial_cam_frame = capture_frame(cap)

    # devnet
    try:
        register_coco_instances("DEV_TRAIN", {}, DEVNET_TRAIN_JSON, DEVNET_TRAIN_IMAGES)
    except Exception:
        pass
    cfg_dev = get_cfg()
    cfg_dev.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg_dev.MODEL.WEIGHTS = DEVNET_WEIGHT
    cfg_dev.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg_dev.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dev.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg_dev.DATASETS.TEST = ()
    dev_predictor = DefaultPredictor(cfg_dev)
    dev_metadata = MetadataCatalog.get("DEV_TRAIN")

    # seg-net 3class
    try:
        register_coco_instances("SEG3_TRAIN", {}, SEG3_TRAIN_JSON, SEG3_TRAIN_IMAGES)
    except Exception:
        pass
    cfg_seg = get_cfg()
    cfg_seg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg_seg.MODEL.WEIGHTS = SEG3_WEIGHT
    cfg_seg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg_seg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_seg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg_seg.DATASETS.TEST = ()
    seg3_predictor = DefaultPredictor(cfg_seg)

    # IK model load (3D)
    ik_model = IKBeamNet(in_ch=9, feat_dim=128).to(device)
    ckpt = torch.load(IK_MODEL_PATH, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    ik_model.load_state_dict(state)
    ik_model.eval()
    print(f"[INFO] Loaded IK model: {IK_MODEL_PATH}")

    # target image select
    img_path = select_image_via_dialog()
    if img_path is None:
        cap.release()
        return

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[ERROR] Failed to read image: {img_path}")
        cap.release()
        return

    img_name = os.path.splitext(os.path.basename(img_path))[0]

    # save dir
    # folder_name = f"3D_ModuleMax{MAX_MODULE}_K={K_GAIN:.1f}_Loop{NUM_LOOP}_Reset{RESET_EACH_LOOP}_Wait{WAIT_TIME:.1f}_Freq{LOOP_WAIT:.1f}"
    # RESULT_ROOT = os.path.join(BASE_ROOT, folder_name)
    # os.makedirs(RESULT_ROOT, exist_ok=True)

    save_dir = os.path.join(SAVE_ROOT, f"selected_{img_name}")
    os.makedirs(save_dir, exist_ok=True)

    cv2.imwrite(os.path.join(save_dir, f"{img_name}_roi_input.png"), img_bgr)

    # timeline buffers
    timeline_mse = []
    timeline_overlays = []

    if initial_cam_frame is not None:
        mse0, ov0 = compute_mse_and_overlay(img_bgr, initial_cam_frame, step_idx=0, base_dir=save_dir)
        timeline_mse.append(mse0)
        timeline_overlays.append(ov0)

    # devnet on target
    out = dev_predictor(img_bgr)
    inst = out["instances"].to("cpu")
    if len(inst) == 0:
        print("[INFO] No detection in target image.")
        cap.release()
        return

    boxes = inst.pred_boxes.tensor.numpy()
    masks = inst.pred_masks.numpy()

    vis = Visualizer(img_bgr[:, :, ::-1], metadata=dev_metadata, scale=1.0)
    out_vis = vis.overlay_instances(masks=masks, boxes=None, labels=None)
    cv2.imwrite(os.path.join(save_dir, f"{img_name}_devnet_vis.png"),
                out_vis.get_image()[:, :, ::-1].astype("uint8"))

    # ---- load target voltages from signals.csv (headerあり)
    target_voltages = None
    try:
        img_p = Path(img_path)
        dataset_dir = img_p.parent.parent  # .../dataset/roi/n.png -> dataset
        signals_path = dataset_dir / "signals.csv"
        target_voltages = load_target_voltages_from_signals(signals_path, img_name, n_modules=min(len(boxes), MAX_MODULE))
        if target_voltages is not None:
            print(f"[INFO] target_voltages shape: {target_voltages.shape}")
            np.savetxt(os.path.join(save_dir, "target_voltages.txt"), target_voltages, fmt="%.4f")
    except Exception as e:
        print(f"[WARN] Failed to load target_voltages: {e}")
        target_voltages = None

    # initial voltages start from 0V (detected modules)
    n_mod = min(int(boxes.shape[0]), MAX_MODULE)
    initial_voltages = np.zeros((n_mod, 3), dtype=np.float32)

    ser = None
    bar_paths_fb = []
    try:
        ser = init_serial(SERIAL_PORT, BAUDRATE)

        # initial reset
        reset_voltages_to_zero(ser, n_channels=MAX_MODULE * 3)
        print(f"[INFO] Waiting {WAIT_TIME} sec after initial reset ...")
        time.sleep(WAIT_TIME)

        cam_frame_init = capture_frame(cap)
        if cam_frame_init is None:
            print("[WARN] Camera capture failed after reset, abort feedback.")
        else:
            cv2.imwrite(os.path.join(save_dir, "captured_full_init.png"), cam_frame_init)

            mse1, ov1 = compute_mse_and_overlay(img_bgr, cam_frame_init, step_idx=1, base_dir=save_dir)
            timeline_mse.append(mse1)
            timeline_overlays.append(ov1)

            mse_fb, overlays_fb, bar_paths_fb = run_feedback_loop_3d(
                img_target_roi=img_bgr,
                boxes_target=boxes,
                initial_voltages=initial_voltages,
                first_cam_frame=cam_frame_init,
                dev_predictor=dev_predictor,
                seg3_predictor=seg3_predictor,
                ik_model=ik_model,
                device=device,
                ser=ser,
                base_save_dir=save_dir,
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
            timeline_overlays.extend(overlays_fb)

        # end reset
        reset_voltages_to_zero(ser, n_channels=MAX_MODULE * 3)

    finally:
        if ser is not None:
            try:
                ser.close()
            except:
                pass
        if cap is not None:
            try:
                cap.release()
            except:
                pass

    # timeline outputs
    if len(timeline_mse) > 0:
        create_mse_plot_and_gif(
            mse_list=timeline_mse[1:],
            overlay_paths=timeline_overlays[1:],
            save_dir=save_dir,
            gif_duration=1.0,
        )

    print(f"/n[INFO] All results saved in: {save_dir}")


if __name__ == "__main__":
    main()
