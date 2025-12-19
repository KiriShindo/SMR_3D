from PIL import Image, ImageDraw, ImageFont

# ===== 罫線設定 =====
line_color = (255, 255, 255)   # 白（論文向け）
line_width = 2                # 太さ（1〜3がおすすめ）


# ===== 25枚の画像フルパス =====
image_paths = [
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_1\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_00_1\fb_step_10/roi_overlay_red_orig_blue_cap.png",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_1\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_01_5\fb_step_10/roi_overlay_red_orig_blue_cap.png",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_1\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_02_6\fb_step_10/roi_overlay_red_orig_blue_cap.png",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_1\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_03_8\fb_step_10/roi_overlay_red_orig_blue_cap.png",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_1\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_04_9\fb_step_10/roi_overlay_red_orig_blue_cap.png",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_2\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_00_1\fb_step_10/roi_overlay_red_orig_blue_cap.png",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_2\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_01_10\fb_step_10/roi_overlay_red_orig_blue_cap.png",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_2\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_02_15\fb_step_10/roi_overlay_red_orig_blue_cap.png",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_2\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_03_25\fb_step_10/roi_overlay_red_orig_blue_cap.png",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_2\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_04_30\fb_step_10/roi_overlay_red_orig_blue_cap.png",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_3\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_00_1\fb_step_10/roi_overlay_red_orig_blue_cap.png",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_3\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_01_11\fb_step_10/roi_overlay_red_orig_blue_cap.png",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_3\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_02_16\fb_step_10/roi_overlay_red_orig_blue_cap.png",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_3\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_03_26\fb_step_10/roi_overlay_red_orig_blue_cap.png",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_3\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_04_31\fb_step_10/roi_overlay_red_orig_blue_cap.png",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_4\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_00_1\fb_step_10/roi_overlay_red_orig_blue_cap.png",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_4\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_01_12\fb_step_10/roi_overlay_red_orig_blue_cap.png",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_4\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_02_17\fb_step_10/roi_overlay_red_orig_blue_cap.png",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_4\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_03_27\fb_step_10/roi_overlay_red_orig_blue_cap.png",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_4\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_04_32\fb_step_10/roi_overlay_red_orig_blue_cap.png",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_5\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_00_1\fb_step_10/roi_overlay_red_orig_blue_cap.png",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_5\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_01_13\fb_step_10/roi_overlay_red_orig_blue_cap.png",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_5\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_02_18\fb_step_10/roi_overlay_red_orig_blue_cap.png",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_5\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_03_28\fb_step_10/roi_overlay_red_orig_blue_cap.png",
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\maxmod_5\K1.0_LOOP10_RESET0_WAIT3.0_LOOPWAIT5.0_TRIAL5\trial_04_33\fb_step_10/roi_overlay_red_orig_blue_cap.png",
]

assert len(image_paths) == 25, "画像は25枚ちょうどにしてください"

# ===== ラベル =====
col_labels = ["Twisted", "Front", "Right", "Left", "Back"]
row_labels = ["#1", "#2", "#3", "#4", "#5"]

# ===== 読み込み =====
images = [Image.open(p).convert("RGB") for p in image_paths]

# ===== サイズ統一（最初の画像基準）=====
w, h = images[0].size
images = [img.resize((w, h), Image.BILINEAR) for img in images]

# ===== フォント（Windows想定：なければデフォルトにフォールバック）=====
try:
    font = ImageFont.truetype("C:/Windows/Fonts/times.ttf", size=64)
except Exception:
    font = ImageFont.load_default()

# ===== ヘッダ領域サイズ（余白）=====
# フォント高さに対して top_margin が足りないと文字が見切れるので、必ず確保する
tmp_img = Image.new("RGB", (10, 10))
tmp_draw = ImageDraw.Draw(tmp_img)
col_th = tmp_draw.textbbox((0, 0), "Twisted", font=font)[3]  # 代表文字高さ

top_margin  = int(max(h * 0.25, col_th + 20))  # ★フォント高さ+余白を保証
left_margin = int(max(160, w * 0.35))
pad = 10

# ===== キャンバス作成（余白込み）=====
grid_w, grid_h = 5 * w, 5 * h
canvas = Image.new("RGB", (left_margin + grid_w, top_margin + grid_h), (0, 0, 0))
draw = ImageDraw.Draw(canvas)

# ===== 画像貼り付け（右下へオフセット）=====
for idx, img in enumerate(images):
    r = idx // 5
    c = idx % 5
    x = left_margin + c * w
    y = top_margin + r * h
    canvas.paste(img, (x, y))

# ==========================================================
# 罫線（5x5セルのグリッド線 + 境界 + 外枠）
# ==========================================================
x0 = left_margin
y0 = top_margin
x1 = left_margin + 5 * w
y1 = top_margin + 5 * h

outer_width = max(line_width, 3)
inner_width = line_width

# 縦罫線（画像領域）
for c in range(6):
    x = x0 + c * w
    width = outer_width if (c == 0 or c == 5) else inner_width
    draw.line([(x, y0), (x, y1)], fill=line_color, width=width)

# 横罫線（画像領域）
for r in range(6):
    y = y0 + r * h
    width = outer_width if (r == 0 or r == 5) else inner_width
    draw.line([(x0, y), (x1, y)], fill=line_color, width=width)

# ヘッダ境界（列ラベル下 / 行ラベル右）
draw.line([(x0, y0), (x1, y0)], fill=line_color, width=outer_width)
draw.line([(x0, y0), (x0, y1)], fill=line_color, width=outer_width)

# 全体外枠（キャンバス外周）
W, H = canvas.size
draw.rectangle([(0, 0), (W - 1, H - 1)], outline=line_color, width=outer_width)

# ==========================================================
# ラベル（最後に描く：罫線で潰れない）
# ==========================================================

# 列ラベル（上端）
for c, label in enumerate(col_labels):
    x_center = left_margin + c * w + w // 2
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = x_center - tw // 2
    y = (top_margin - th) // 2
    draw.text((x, y), label, font=font, fill=(255, 255, 255))

# 行ラベル（左端）
for r, label in enumerate(row_labels):
    y_center = top_margin + r * h + h // 2
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = max(pad, left_margin - tw - pad)  # 右寄せ気味
    y = y_center - th // 2
    draw.text((x, y), label, font=font, fill=(255, 255, 255))

# ===== 保存 =====
out_path = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_multi\output_5x5_labeled_noreset.png"
canvas.save(out_path)
print(f"saved -> {out_path}")
