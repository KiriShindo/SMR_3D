import tkinter as tk
from PIL import Image, ImageTk
from serial import Serial

# --- 設定部分 --- #
SERIAL_PORT  = 'COM4'
BAUDRATE     = 9600
IMAGE_PATH   = 'UI.jpg'
MAX_SIZE     = (630*2, 420*2)  # 最大サイズ (width, height)

# 5×3 レイアウト: L/C/R × 段(1〜5)
POSITIONS = {
    'L1': (220, 130), 'C1': (300, 130), 'R1': (380, 130),
    'L2': (220, 260), 'C2': (300, 260), 'R2': (380, 260),
    'L3': (220, 390), 'C3': (300, 390), 'R3': (380, 390),
    'L4': (220, 520), 'C4': (300, 520), 'R4': (380, 520),
    'L5': (220, 650), 'C5': (300, 650), 'R5': (380, 650),
}

MUSCLE_NAMES = [
    'L1', 'C1', 'R1',
    'L2', 'C2', 'R2',
    'L3', 'C3', 'R3',
    'L4', 'C4', 'R4',
    'L5', 'C5', 'R5',
]


class MuscleControllerApp:
    def __init__(self, root):
        self.root = root
        root.title("Muscle Voltage Controller (Aspect Fit 630×420)")

        # ステータス表示ラベル
        self.status = tk.StringVar(value="Ready")
        status_label = tk.Label(root, textvariable=self.status, anchor='w')
        status_label.pack(fill='x', side='bottom')

        # シリアルポートを開いて READY を待つ
        try:
            self.ser = Serial(SERIAL_PORT, BAUDRATE, timeout=1)
            self.status.set(f"[Serial] Opening {SERIAL_PORT}@{BAUDRATE}...")
            while True:
                line = self.ser.readline().decode(errors='ignore').strip()
                if line:
                    self.status.set(f"[Serial] ← {line}")
                if line == 'READY':
                    break
            self.status.set("Arduino READY")
        except Exception as e:
            self.status.set(f"Serial Error: {e}")
            self.ser = None
            return

        # === 元画像を読み込み、アスペクト比を保ったまま最大630x420に収まるようリサイズ ===
        img = Image.open(IMAGE_PATH)
        img.thumbnail(MAX_SIZE, Image.LANCZOS)  # アスペクト比保持・cropなし
        self.tkimg = ImageTk.PhotoImage(img)

        # === キャンバス作成 ===
        canvas = tk.Canvas(root, width=self.tkimg.width(), height=self.tkimg.height())
        canvas.pack()
        canvas.create_image(0, 0, image=self.tkimg, anchor='nw')

        # === テキストボックス ===
        self.entries = {}
        for name, (x, y) in POSITIONS.items():
            lbl = tk.Label(root, text=name, bg='white')
            canvas.create_window(x, y, window=lbl)
            ent = tk.Entry(root, width=4, font=('Arial', 12), justify='center')
            ent.insert(0, "0.0")
            canvas.create_window(x+35, y, window=ent)
            self.entries[name] = ent

        # === ボタン群 ===
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="印加",   width=8, command=self.apply_voltages).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="リセット", width=8, command=self.reset_voltages).grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="終了",   width=8, command=self.exit_app).grid( row=0, column=2, padx=5)

    def apply_voltages(self):
        if self.ser is None:
            self.status.set("Serial not available.")
            return

        try:
            volts = []
            for name in MUSCLE_NAMES:
                v = float(self.entries[name].get())
                v = max(0.0, min(5.0, v))  # 0〜5Vにクリップ
                volts.append(v)

            cmd = 'VOLT ' + ','.join(f"{v:.1f}" for v in volts) + '\n'
            self.status.set(f"[Serial] → {cmd.strip()}")
            self.ser.write(cmd.encode())

            while True:
                resp = self.ser.readline().decode(errors='ignore').strip()
                if resp:
                    self.status.set(f"[Serial] ← {resp}")
                if resp == 'APPLIED':
                    break

            self.status.set("Voltages applied.")
        except ValueError:
            self.status.set("Input Error: enter valid numbers.")

    def reset_voltages(self):
        if self.ser is None:
            self.status.set("Serial not available.")
            return

        zeros = [0.0] * len(MUSCLE_NAMES)
        cmd = 'VOLT ' + ','.join(f"{v:.1f}" for v in zeros) + '\n'
        self.status.set(f"[Serial] → {cmd.strip()}")
        self.ser.write(cmd.encode())

        while True:
            resp = self.ser.readline().decode(errors='ignore').strip()
            if resp:
                self.status.set(f"[Serial] ← {resp}")
            if resp == 'APPLIED':
                break

        self.status.set("All channels reset to 0.0V.")

    def exit_app(self):
        if self.ser is not None:
            zeros = [0.0] * len(MUSCLE_NAMES)
            cmd = 'VOLT ' + ','.join(f"{v:.1f}" for v in zeros) + '\n'
            try:
                self.ser.write(cmd.encode())
            except:
                pass
            try:
                self.ser.close()
            except:
                pass
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    MuscleControllerApp(root)
    root.mainloop()
