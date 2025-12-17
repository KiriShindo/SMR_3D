# -*- coding: utf-8 -*-
"""
motor_babbling_capture_3d_1module_grid.py
---------------------------------------------------------
3次元1モジュールロボット用 motor babbling データ取得。

L1, C1, R1 の3chに対して、
  0.0〜5.0V (0.5V刻み, 計11レベル) の全組み合わせ
  11 x 11 x 11 = 1331 パターンをすべて印加して記録する。

特徴:
- 画像は 1.png, 2.png, ... と連番保存
- signals.csv に [L1_V, C1_V, R1_V] を追記
- signals.csv が既に存在する場合は、
    - データ行数 N に応じて Nパターンぶんを「完了済み」とみなし、
      パターン N+1 から再開（再開時も順番は変わらない）
---------------------------------------------------------
"""

import os
import csv
import time
import cv2
import av
import queue
import threading
from serial import Serial

# ====== 設定パラメータ ======
SERIAL_PORT = 'COM4'
BAUDRATE = 9600
DEVICE_DSHOW = 'video=HD Pro Webcam C920'
VIDEO_SIZE = '1920x1080'

OUT_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\module_controller\IK\1module_babbling_data\normal\raw"

FRAME_QUEUE_SIZE = 1
STEP_INTERVAL = 3.0      # 各ステップ後の待機時間 [s]

MUSCLE_NAMES = ['L1', 'C1', 'R1']
NUM_MUSCLES = len(MUSCLE_NAMES)

# 0.0, 0.5, ..., 5.0
LEVELS = [i * 0.5 for i in range(11)]  # 11レベル → 11^3 = 1331通り


def generate_all_patterns():
    """
    L1, C1, R1 それぞれ LEVELS を取りうる全組み合わせを生成。
    順番:
        for L in LEVELS:
          for C in LEVELS:
            for R in LEVELS:
               (L, C, R)
    """
    patterns = []
    for vL in LEVELS:
        for vC in LEVELS:
            for vR in LEVELS:
                patterns.append((vL, vC, vR))
    return patterns


def setup_output_dir():
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, 'signals.csv')

    start_index = 0  # 次に打つべきパターンのインデックス (0始まり)
    img_index = 1    # 次に保存する画像番号

    if os.path.exists(csv_path):
        # 既存CSVがあれば、データ行数から再開位置を決める
        with open(csv_path, 'r', newline='') as f:
            reader = list(csv.reader(f))

        if len(reader) > 1:
            # 1行目ヘッダを除いたデータ行数 N が、
            # すでに完了したパターン数と一致する前提
            num_data_rows = len(reader) - 1
            start_index = num_data_rows
            img_index = num_data_rows + 1
            print(f"[Info] Resume from pattern index {start_index}, next img_index={img_index}")
        else:
            print("[Info] signals.csv has no data rows. Start from beginning.")
            img_index = 1
    else:
        # 新規作成: ヘッダだけ書く
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['L1_V', 'C1_V', 'R1_V'])
        print("[Info] Created new signals.csv. Start from beginning, img_index=1")

    return csv_path, start_index, img_index


def setup_camera():
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

    th = threading.Thread(target=frame_reader, daemon=True)
    th.start()

    return container, frame_queue


def setup_serial():
    ser = Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    print(f"Opening serial {SERIAL_PORT}@{BAUDRATE}…")
    while True:
        line = ser.readline().decode(errors='ignore').strip()
        if line:
            print(f"[Serial] ← {line}")
        if line == 'READY':
            print("Arduino ready.")
            break
    return ser


def send_voltage_3ch(ser, vL1, vC1, vR1):
    """
    L1, C1, R1 の3ch電圧を Arduino に送信。
    Arduino側は VOLT コマンドを受け取り、
    [L1, C1, R1] の順で解釈している前提。
    """
    pat = [vL1, vC1, vR1]
    cmd = 'VOLT ' + ','.join(f'{v:.1f}' for v in pat) + '\n'
    print(f"[Serial] → {cmd.strip()}")
    ser.write(cmd.encode())
    ser.flush()
    return pat


def main():
    # 全パターン生成（1331 個）
    patterns = generate_all_patterns()
    total_patterns = len(patterns)
    print(f"[Info] Total patterns: {total_patterns} (should be 1331)")

    csv_path, start_index, img_index = setup_output_dir()
    ser = setup_serial()
    container, frame_queue = setup_camera()

    if start_index >= total_patterns:
        print("[Info] All patterns already completed. Nothing to do.")
        return

    print(f"\nStart 3D 1-module grid motor babbling from pattern {start_index}/{total_patterns}.\n")

    try:
        for idx in range(start_index, total_patterns):
            vL1, vC1, vR1 = patterns[idx]
            print(
                f"[Pattern {idx+1}/{total_patterns}] "
                f"L1={vL1:.1f}V, C1={vC1:.1f}V, R1={vR1:.1f}V"
            )

            # 電圧送信
            send_voltage_3ch(ser, vL1, vC1, vR1)

            # 準静的になるまで待つ
            time.sleep(STEP_INTERVAL)

            # 画像キャプチャ
            try:
                frame = frame_queue.get(timeout=1.0)
                img_path = os.path.join(OUT_DIR, f"{img_index}.png")
                cv2.imwrite(img_path, frame)
                print(f"[Saved] {img_index}.png")
            except queue.Empty:
                print("[Warning] Frame not captured")

            # CSVに電圧を追記
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([vL1, vC1, vR1])

            img_index += 1

    except KeyboardInterrupt:
        print("\n[Info] KeyboardInterrupt. Stopping early...")

    finally:
        # 安全のため q を送って全チャネル 0V に戻す想定
        try:
            ser.write(b'q\n')
            ser.flush()
        except Exception:
            pass
        try:
            ser.close()
        except Exception:
            pass
        try:
            container.close()
        except Exception:
            pass

        print("\nDone. Resources released. You can rerun this script to continue remaining patterns.")


if __name__ == "__main__":
    main()
