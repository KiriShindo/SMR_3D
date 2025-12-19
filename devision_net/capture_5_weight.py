#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import csv
import cv2
import av
import threading
import queue
from serial import Serial

# Windows のみキーボード検知
if os.name == 'nt':
    import msvcrt

# --- 設定 ---
SERIAL_PORT      = 'COM4'
BAUDRATE         = 9600
DEVICE_DSHOW     = 'video=HD Pro Webcam C920'
VIDEO_SIZE       = '1920x1080'
OUT_DIR          = r'C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\devision_net\devnet_data_new\5module_dataset_max_DAC_weight2'
FRAME_QUEUE_SIZE = 1
# ------------------

MUSCLE_NAMES = [
    'L1', 'C1', 'R1',
    'L2', 'C2', 'R2',
    'L3', 'C3', 'R3',
    'L4', 'C4', 'R4',
    'L5', 'C5', 'R5',
]


def kb_hit():
    return msvcrt.kbhit() if os.name == 'nt' else False


def get_char():
    return msvcrt.getch().decode() if os.name == 'nt' else ''


def pattern_dict_to_list(pattern_dict):
    voltages = []
    for module_idx in range(1, 6):
        row = pattern_dict.get(module_idx, {})
        vL = float(row.get('L', 0.0))
        vC = float(row.get('C', 0.0))
        vR = float(row.get('R', 0.0))
        voltages.extend([vL, vC, vR])
    return voltages


# ===== 手動で指定する電圧パターン =====
PATTERN_DICTS = [
    # ねじれ
    {
        1: {'L': 2.0, 'C': 2.0, 'R': 0.0},
        2: {'L': 0.0, 'C': 2.0, 'R': 0.0},
        3: {'L': 0.0, 'C': 2.0, 'R': 2.0},
        4: {'L': 0.0, 'C': 0.0, 'R': 2.0},
        5: {'L': 2.0, 'C': 0.0, 'R': 2.0},
    },
    # C方向
    {
        1: {'L': 1.0, 'C': 4.0, 'R': 1.0},
        2: {'L': 1.0, 'C': 4.0, 'R': 1.0},
        3: {'L': 1.0, 'C': 4.0, 'R': 1.0},
        4: {'L': 1.0, 'C': 4.0, 'R': 1.0},
        5: {'L': 1.0, 'C': 4.0, 'R': 1.0},
    },
    # R方向
    {
        1: {'L': 1.0, 'C': 1.0, 'R': 4.0},
        2: {'L': 1.0, 'C': 1.0, 'R': 4.0},
        3: {'L': 1.0, 'C': 1.0, 'R': 4.0},
        4: {'L': 1.0, 'C': 1.0, 'R': 4.0},
        5: {'L': 1.0, 'C': 1.0, 'R': 4.0},
    },
    # LC方向
    {
        1: {'L': 4.0, 'C': 4.0, 'R': 1.0},
        2: {'L': 4.0, 'C': 4.0, 'R': 1.0},
        3: {'L': 4.0, 'C': 4.0, 'R': 1.0},
        4: {'L': 4.0, 'C': 4.0, 'R': 1.0},
        5: {'L': 4.0, 'C': 4.0, 'R': 1.0},
    },
    # LR方向
    {
        1: {'L': 4.0, 'C': 1.0, 'R': 4.0},
        2: {'L': 4.0, 'C': 1.0, 'R': 4.0},
        3: {'L': 4.0, 'C': 1.0, 'R': 4.0},
        4: {'L': 4.0, 'C': 1.0, 'R': 4.0},
        5: {'L': 4.0, 'C': 1.0, 'R': 4.0},
    },
]


def send_voltages(ser, voltages):
    """VOLT コマンド送信＆APPLIED応答待ち"""
    cmd = 'VOLT ' + ','.join(f'{v:.1f}' for v in voltages) + '\n'
    ser.write(cmd.encode())

    while True:
        resp = ser.readline().decode().strip()
        if resp == 'APPLIED':
            break


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, 'signals.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(MUSCLE_NAMES)

    ser = Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    print(f"Opening serial {SERIAL_PORT}@{BAUDRATE}…")
    while True:
        line = ser.readline().decode().strip()
        if line == 'READY':
            print("Arduino ready.")
            break
        elif line:
            print(f"[Arduino] {line}")

    container = av.open(format='dshow', file=DEVICE_DSHOW, options={'video_size': VIDEO_SIZE})
    frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)

    def frame_reader():
        for packet in container.demux(video=0):
            for frame in packet.decode():
                img = frame.to_ndarray(format='bgr24')
                try:
                    frame_queue.put_nowait(img)
                except queue.Full:
                    _ = frame_queue.get_nowait()
                    frame_queue.put_nowait(img)

    threading.Thread(target=frame_reader, daemon=True).start()

    patterns = [pattern_dict_to_list(p) for p in PATTERN_DICTS]

    print("Starting sequence. Press 'q' to emergency stop.\n")
    img_index = 1

    try:
        for step_idx, pat in enumerate(patterns, start=1):
            # === 全OFF → 3秒待ち ===
            print(f"\n[Step {step_idx}] Setting all channels to 0.0 V...")
            send_voltages(ser, [0.0] * len(MUSCLE_NAMES))
            time.sleep(3)

            # === 次の電圧印加 ===
            on_info = []
            for m in range(1, 6):
                base = (m - 1) * 3
                vL, vC, vR = pat[base:base + 3]
                elems = []
                if vL > 0.0: elems.append(f"L{m}={vL:.1f}")
                if vC > 0.0: elems.append(f"C{m}={vC:.1f}")
                if vR > 0.0: elems.append(f"R{m}={vR:.1f}")
                if elems: on_info.append(f"M{m}({', '.join(elems)})")
            print("Applying voltages:", "; ".join(on_info) if on_info else "ALL OFF")

            send_voltages(ser, pat)
            print(" → Voltages applied.")
            time.sleep(5)  # 安定化待ち

            # === カメラキャプチャ ===
            try:
                img = frame_queue.get(timeout=1)
                img_fn = os.path.join(OUT_DIR, f"{img_index}.png")
                cv2.imwrite(img_fn, img)
                print(f"[Capture] saved {img_index}.png")
            except queue.Empty:
                print(f"[Warning] no frame for step {step_idx}")

            # CSV 追記
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(pat)

            img_index += 1
            time.sleep(4)

    finally:
        try:
            send_voltages(ser, [0.0] * len(MUSCLE_NAMES))
            ser.write(b'q\n')
        except:
            pass
        ser.close()
        container.close()
        print("\nDone. All resources released.")


if __name__ == '__main__':
    main()
