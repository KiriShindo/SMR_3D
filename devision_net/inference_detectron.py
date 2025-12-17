import os
import cv2
import colorsys
import random
import numpy as np
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

# ===== 処理モードフラグ =====
# True  : detectron2 の bbox サイズそのまま切り出し（補完なし）
# False : 70x70 の固定サイズにしてノイズ + distance transform で補完（従来の動作）
USE_ORIGINAL_BBOX = False

# ===== 背景色の統計（BGR） =====
MEAN_BGR = np.array([128.04, 131.38, 132.08], dtype=np.float32)   # 必ず更新
STD_BGR  = np.array([  6.56,   6.47,   5.51], dtype=np.float32)   # 必ず更新
BLEND_WIDTH = 5.0  # 3〜10で調整可


# ===== ノイズ生成関数 =====
def sample_border_noise(H, W):
    noise = np.random.normal(MEAN_BGR, STD_BGR, size=(H, W, 3))
    return np.clip(noise, 0, 255).astype(np.uint8)


# ===== カラー生成関数 =====
def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
        colors.append((r, g, b))
    return colors


# ===== ランダム評価画像抽出 =====
def get_random_eval_images(base_dir, n_folders=6, n_samples=10):
    random_entries = []
    for _ in range(n_samples):
        i = random.randint(1, n_folders)
        roi_dir = os.path.join(base_dir, f"{i}module_dataset_max_DAC", "roi")
        if not os.path.exists(roi_dir):
            continue
        files = [f for f in os.listdir(roi_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not files:
            continue
        chosen = random.choice(files)
        img_path = os.path.join(roi_dir, chosen)
        random_entries.append((img_path, i))
    return random_entries


# ===== bbox切り出し＋ブレンド補完 =====
def crop_and_save(img, boxes, save_dir, crop_h=70, crop_w=70):
    """
    各bboxをcropして保存（上から順に番号付け）

    USE_ORIGINAL_BBOX == True のとき:
        - center_crop_XX.png : detectron2 の bbox をそのまま切り出した画像（補完なし）
        - crop_XX.png        : 保存しない（必要なら追加してもOK）

    USE_ORIGINAL_BBOX == False のとき（従来動作）:
        - crop_XX.png        : 元のbboxをそのまま切り出した画像
        - center_crop_XX.png : bbox中心まわり 70x70 を、ノイズ + distance transform で補完した画像
    """
    os.makedirs(save_dir, exist_ok=True)
    H, W, _ = img.shape

    # y1 が小さい順にソート
    boxes_sorted = sorted(boxes, key=lambda b: b[1])
    half_h = crop_h // 2
    half_w = crop_w // 2

    for idx, box in enumerate(boxes_sorted):
        x1, y1, x2, y2 = [int(v) for v in box]

        # ======== 補完なしモード（bboxそのまま） ========
        if USE_ORIGINAL_BBOX:
            full_crop = img[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
            center_crop_path = os.path.join(save_dir, f"center_crop_{idx:02d}.png")
            cv2.imwrite(center_crop_path, full_crop)
            continue
        # ===============================================

        # ======== ここから従来の 70x70 補完処理 ========
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # 中心基準の切り出し範囲
        x1c, x2c = cx - half_w, cx + half_w
        y1c, y2c = cy - half_h, cy + half_h

        cropped = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        mask    = np.zeros((crop_h, crop_w), dtype=np.uint8)

        # 元画像内での重なり部分を計算
        x1_src, y1_src = max(0, x1c), max(0, y1c)
        x2_src, y2_src = min(W, x2c), min(H, y2c)

        # ペースト先（cropped側）
        x1_dst = x1_src - x1c
        y1_dst = y1_src - y1c
        x2_dst = x1_dst + (x2_src - x1_src)
        y2_dst = y1_dst + (y2_src - y1_src)

        if x2_src > x1_src and y2_src > y1_src:
            cropped[y1_dst:y2_dst, x1_dst:x2_dst] = img[y1_src:y2_src, x1_src:x2_src]
            mask[y1_dst:y2_dst, x1_dst:x2_dst] = 255

        # --- augment_random_affine_roi と同じ境界補完 ---
        noise = sample_border_noise(crop_h, crop_w)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        alpha = np.clip(dist / BLEND_WIDTH, 0.0, 1.0).astype(np.float32)
        alpha_3 = alpha[..., None]

        blended = cropped.astype(np.float32) * alpha_3 + noise.astype(np.float32) * (1 - alpha_3)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        # ---------------------------------------------

        # bbox全体のcrop（参考）
        full_crop = img[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]

        # 保存
        crop_path        = os.path.join(save_dir, f"crop_{idx:02d}.png")
        center_crop_path = os.path.join(save_dir, f"center_crop_{idx:02d}.png")
        cv2.imwrite(crop_path, full_crop)
        cv2.imwrite(center_crop_path, blended)

    print(f"Saved {len(boxes_sorted)} cropped / center-crop images → {save_dir}")


# ===== メイン処理 =====
def main():
    # ===== Detectron2設定 =====
    train_json   = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control_3D/devision_net/devnet_data_new/all_dataset/annotations.json"
    train_images = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control_3D/devision_net/devnet_data_new/all_dataset/"
    register_coco_instances("Train", {}, train_json, train_images)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control_3D/devision_net/devnet_data_new/all_dataset/prm01/model_final.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.DATASETS.TEST = ()
    predictor = DefaultPredictor(cfg)

    # ===== ランダム画像選択 =====
    base_dir = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control_3D/devision_net/devnet_data_new"
    eval_entries = get_random_eval_images(base_dir, n_folders=6, n_samples=10)
    print("Randomly selected evaluation images:")
    for p, i in eval_entries:
        print(f" - {p} (module {i})")

    metadata = MetadataCatalog.get("Train")
    result_root = os.path.join(base_dir, r"result/devnet_only")
    os.makedirs(result_root, exist_ok=True)

    # ===== 推論ループ =====
    for idx, (img_path, module_idx) in enumerate(eval_entries):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Image not found: {img_path}")
            continue

        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")

        if len(instances) == 0:
            print(f"[INFO] No detection in {img_path}")
            continue

        boxes = instances.pred_boxes.tensor.numpy()
        masks = instances.pred_masks.numpy()
        num_masks = masks.shape[0]
        assigned_colors = generate_distinct_colors(num_masks)

        vis = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        out_vis = vis.overlay_instances(
            masks=masks, boxes=None, labels=None, assigned_colors=assigned_colors
        )
        result_img = out_vis.get_image()[:, :, ::-1].astype("uint8")

        # 保存ディレクトリ（例: module3_22）
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        save_dir = os.path.join(result_root, f"module{module_idx}_{img_name}")
        os.makedirs(save_dir, exist_ok=True)

        vis_path = os.path.join(save_dir, f"{img_name}_vis.png")
        cv2.imwrite(vis_path, result_img)
        print(f"Saved visualization → {vis_path}")

        # bboxごとにcrop + (必要なら) blend
        crop_and_save(img, boxes, save_dir, crop_h=70, crop_w=70)

    print("✅ Finished 10 random evaluations with crops / center crops.")

    # ===== center_crop 画像の集約保存 =====
    merged_dir = os.path.join(base_dir, "result", "merged_center_crops")
    os.makedirs(merged_dir, exist_ok=True)

    print(f"/nCollecting all center_crop_*.png into → {merged_dir}")

    # result_root配下の全フォルダを再帰的に探索
    idx = 1
    for root, dirs, files in os.walk(result_root):
        for f in sorted(files):
            if f.startswith("center_crop_") and f.lower().endswith(".png"):
                src_path = os.path.join(root, f)
                dst_path = os.path.join(merged_dir, f"{idx:04d}.png")
                cv2.imwrite(dst_path, cv2.imread(src_path))
                idx += 1

    print(f"✅ Collected {idx-1} center_crop images into {merged_dir}")


if __name__ == "__main__":
    main()
