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
PATTERN_DICTS = [
    # ねじれ
    {
        1: {'L': 5.0, 'C': 5.0, 'R': 0.0},
        2: {'L': 0.0, 'C': 5.0, 'R': 0.0},
        3: {'L': 0.0, 'C': 5.0, 'R': 5.0},
        4: {'L': 0.0, 'C': 0.0, 'R': 5.0},
        5: {'L': 5.0, 'C': 0.0, 'R': 5.0},
    },
    {
        1: {'L': 5.0, 'C': 0.0, 'R': 5.0},
        2: {'L': 5.0, 'C': 0.0, 'R': 0.0},
        3: {'L': 5.0, 'C': 5.0, 'R': 0.0},
        4: {'L': 0.0, 'C': 5.0, 'R': 0.0},
        5: {'L': 0.0, 'C': 5.0, 'R': 5.0},
    },
    {
        1: {'L': 0.0, 'C': 5.0, 'R': 5.0},
        2: {'L': 0.0, 'C': 0.0, 'R': 5.0},
        3: {'L': 5.0, 'C': 0.0, 'R': 5.0},
        4: {'L': 5.0, 'C': 0.0, 'R': 0.0},
        5: {'L': 5.0, 'C': 5.0, 'R': 0.0},
    },
    # L方向
    {
        1: {'L': 5.0, 'C': 0.0, 'R': 0.0},
        2: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        3: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        4: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        5: {'L': 0.0, 'C': 0.0, 'R': 0.0},
    },
    {
        1: {'L': 5.0, 'C': 0.0, 'R': 0.0},
        2: {'L': 5.0, 'C': 0.0, 'R': 0.0},
        3: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        4: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        5: {'L': 0.0, 'C': 0.0, 'R': 0.0},
    },
    # C方向
    {
        1: {'L': 0.0, 'C': 5.0, 'R': 0.0},
        2: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        3: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        4: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        5: {'L': 0.0, 'C': 0.0, 'R': 0.0},
    },
    {
        1: {'L': 0.0, 'C': 5.0, 'R': 0.0},
        2: {'L': 0.0, 'C': 5.0, 'R': 0.0},
        3: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        4: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        5: {'L': 0.0, 'C': 0.0, 'R': 0.0},
    },
    # R方向
    {
        1: {'L': 0.0, 'C': 0.0, 'R': 5.0},
        2: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        3: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        4: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        5: {'L': 0.0, 'C': 0.0, 'R': 0.0},
    },
    {
        1: {'L': 0.0, 'C': 0.0, 'R': 5.0},
        2: {'L': 0.0, 'C': 0.0, 'R': 5.0},
        3: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        4: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        5: {'L': 0.0, 'C': 0.0, 'R': 0.0},
    },
    # CR方向
    {
        1: {'L': 0.0, 'C': 5.0, 'R': 5.0},
        2: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        3: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        4: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        5: {'L': 0.0, 'C': 0.0, 'R': 0.0},
    },
    {
        1: {'L': 0.0, 'C': 5.0, 'R': 5.0},
        2: {'L': 0.0, 'C': 5.0, 'R': 5.0},
        3: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        4: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        5: {'L': 0.0, 'C': 0.0, 'R': 0.0},
    },
    # LC方向
    {
        1: {'L': 5.0, 'C': 5.0, 'R': 0.0},
        2: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        3: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        4: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        5: {'L': 0.0, 'C': 0.0, 'R': 0.0},
    },
    {
        1: {'L': 5.0, 'C': 5.0, 'R': 0.0},
        2: {'L': 5.0, 'C': 5.0, 'R': 0.0},
        3: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        4: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        5: {'L': 0.0, 'C': 0.0, 'R': 0.0},
    },
    # LR方向
    {
        1: {'L': 5.0, 'C': 0.0, 'R': 5.0},
        2: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        3: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        4: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        5: {'L': 0.0, 'C': 0.0, 'R': 0.0},
    },
    {
        1: {'L': 5.0, 'C': 0.0, 'R': 5.0},
        2: {'L': 5.0, 'C': 0.0, 'R': 5.0},
        3: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        4: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        5: {'L': 0.0, 'C': 0.0, 'R': 0.0},
    },
    # 全0
    {
        1: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        2: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        3: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        4: {'L': 0.0, 'C': 0.0, 'R': 0.0},
        5: {'L': 0.0, 'C': 0.0, 'R': 0.0},
    },
]

# Windows のみ
if os.name == 'nt':
    import msvcrt

# ==== ここにさっきの PATTERN_DICTS をコピペ ====


# --- 共通設定 ---
SERIAL_PORT = 'COM4'
BAUDRATE    = 9600
DEVICE_DSHOW = 'video=HD Pro Webcam C920'
VIDEO_SIZE   = '1920x1080'
FRAME_QUEUE_SIZE = 1


def kb_hit():
    return msvcrt.kbhit() if os.name == 'nt' else False


def get_char():
    return msvcrt.getch().decode() if os.name == 'nt' else ''


def build_muscle_names(num_modules: int):
    """
    num_modules に応じて [L1,C1,R1,...,LN,CN,RN] を作る
    """
    names = []
    for m in range(1, num_modules + 1):
        names.extend([f"L{m}", f"C{m}", f"R{m}"])
    return names


def pattern_dict_to_list(pattern_dict, num_modules: int):
    """
    PATTERN_DICTS の1要素（モジュール1〜5の辞書）から、
    指定された num_modules ぶんだけ [L1,C1,R1,...,LN,CN,RN] の配列を作る。
    それより下のモジュール(>num_modules)は無視。
    """
    voltages = []
    for m in range(1, num_modules + 1):
        row = pattern_dict.get(m, {})
        vL = float(row.get('L', 0.0))
        vC = float(row.get('C', 0.0))
        vR = float(row.get('R', 0.0))
        voltages.extend([vL, vC, vR])
    return voltages


def send_voltages(ser, voltages):
    """
    VOLT コマンド送信 & APPLIED 応答待ち。
    Arduino側は、渡された個数分だけをチャンネル0〜…に割り当て、
    足りない残りは自動的に 0V にしてくれる。
    """
    cmd = 'VOLT ' + ','.join(f"{v:.1f}" for v in voltages) + '\n'
    ser.write(cmd.encode())
    while True:
        resp = ser.readline().decode().strip()
        if resp == 'APPLIED':
            break
        elif resp:
            print(f"[Arduino] {resp}")


def run_capture(num_modules: int, out_dir: str):
    """
    num_modules: 使用するモジュール数 (1〜5)
    out_dir: 画像とsignals.csvを保存するディレクトリ
    """
    assert 1 <= num_modules <= 5, "num_modules must be 1..5"

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'signals.csv')

    muscle_names = build_muscle_names(num_modules)

    # CSV ヘッダ
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(muscle_names)

    # シリアルオープン & READY待ち
    ser = Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    print(f"Opening serial {SERIAL_PORT}@{BAUDRATE}…")
    while True:
        line = ser.readline().decode().strip()
        if line == 'READY':
            print("Arduino ready.")
            break
        elif line:
            print(f"[Arduino] {line}")

    # カメラオープン
    container = av.open(
        format='dshow',
        file=DEVICE_DSHOW,
        options={'video_size': VIDEO_SIZE}
    )

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

    # PATTERN_DICTS → 実際に送るパターン列
    patterns = [pattern_dict_to_list(p, num_modules) for p in PATTERN_DICTS]

    print(f"\n=== Start capture: {num_modules} modules ===")
    print("Press 'q' for emergency stop.\n")

    img_index = 1

    try:
        for step_idx, pat in enumerate(patterns, start=1):

            # --- まず全0Vを印加 → 3秒待つ ---
            print(f"\n[Step {step_idx}] Set all {num_modules*3}ch to 0.0V")
            send_voltages(ser, [0.0] * (num_modules * 3))
            time.sleep(3)

            # --- 次のパターンを印加 ---
            # どこがONかログに出す
            on_info = []
            for m in range(1, num_modules + 1):
                base = (m - 1) * 3
                vL, vC, vR = pat[base:base + 3]
                elems = []
                if vL > 0.0: elems.append(f"L{m}={vL:.1f}")
                if vC > 0.0: elems.append(f"C{m}={vC:.1f}")
                if vR > 0.0: elems.append(f"R{m}={vR:.1f}")
                if elems:
                    on_info.append(f"M{m}({', '.join(elems)})")

            print(f"[Step {step_idx}] Apply:", "; ".join(on_info) if on_info else "ALL OFF")
            send_voltages(ser, pat)
            print(" → Voltages applied.")
            time.sleep(5)  # 安定化待ち

            # --- フレーム取得 ---
            try:
                img = frame_queue.get(timeout=1)
                img_fn = os.path.join(out_dir, f"{img_index}.png")
                cv2.imwrite(img_fn, img)
                print(f"[Capture] saved {img_index}.png")
            except queue.Empty:
                print(f"[Warning] no frame for step {step_idx}")

            # CSV追記
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(pat)

            img_index += 1
            time.sleep(4)

    finally:
        try:
            # 最後に全0V + q
            send_voltages(ser, [0.0] * (num_modules * 3))
            ser.write(b'q\n')
        except:
            pass
        ser.close()
        container.close()
        print(f"\nDone. Capture finished for {num_modules} modules.")


if __name__ == '__main__':
    # ---- ここから、欲しいモジュール数ごとに呼び出す ----

    # 2モジュール用
    run_capture(
        num_modules=2,
        out_dir=r'C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\devision_net\devnet_data_new\2module_dataset_max_DAC'
    )