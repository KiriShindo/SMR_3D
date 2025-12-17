import os
import cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

# ======= 出力時の色 (BGR) =======
LEFT_COLOR_BGR   = (0, 0, 255)   # left  : 赤
CENTER_COLOR_BGR = (0, 255, 0)   # center: 緑
RIGHT_COLOR_BGR  = (255, 0, 0)   # right : 青


def compute_mask_center(mask_bool):
    """
    boolマスクから重心 (cx, cy) を計算する。
    画素がない場合は (0, 0) を返す。
    """
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return 0.0, 0.0
    cx = float(xs.mean())
    cy = float(ys.mean())
    return cx, cy


def main():
    # ===== Detectron2設定 =====
    train_json   = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control_3D/module_controller/IK/1module_jsons_3D_dataset/annotations.json"
    train_images = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control_3D/module_controller/IK/1module_jsons_3D_dataset/"

    try:
        register_coco_instances("Train", {}, train_json, train_images)
    except Exception:
        # すでに登録済みの場合など
        pass

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.WEIGHTS = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control_3D/module_controller/IK/1module_jsons_3D_dataset/prm01/model_final.pth"

    # クラス数はデータセットに合わせて変更（例: left/center/right の3クラスなら 3）
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.DATASETS.TEST = ()

    predictor = DefaultPredictor(cfg)
    metadata  = MetadataCatalog.get("Train")

    # ===== 入出力パス =====
    in_dir = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control_3D/module_controller/IK/1module_babbling_data/normal/aug_shift5_rot30"
    out_dir = os.path.join(
        r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control_3D/module_controller/IK/1module_babbling_data/normal",
        "roi_aug_shift5_rot30_beammask_3ch_0.6"
    )
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(in_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    files.sort()
    print(f"Found {len(files)} images in {in_dir}")

    for idx, fname in enumerate(files, start=1):
        img_path = os.path.join(in_dir, fname)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[WARN] Failed to load: {img_path}")
            continue

        outputs = predictor(img_bgr)
        instances = outputs["instances"].to("cpu")

        if len(instances) < 3:
            print(f"[WARN] <3 instances: {fname}, skip.")
            continue

        scores     = instances.scores.numpy()
        masks_all  = instances.pred_masks.numpy().astype(bool)
        # classes = instances.pred_classes.numpy()  # クラスIDを使いたい場合はここから取得

        # スコア上位3つを取得
        top3_idx = np.argsort(-scores)[:3]
        masks_three = masks_all[top3_idx]  # (3, H, W)

        # --- left / center / right を x座標で決める ---
        centers_x = []
        for m in masks_three:
            cx, cy = compute_mask_center(m)
            centers_x.append(cx)
        centers_x = np.array(centers_x)

        # x が小さい順に並べる → left, center, right
        order = np.argsort(centers_x)  # 0: left, 1: center, 2: right
        masks_sorted = masks_three[order]  # (3, H, W)

        mask_left   = masks_sorted[0].astype(np.float32)
        mask_center = masks_sorted[1].astype(np.float32)
        mask_right  = masks_sorted[2].astype(np.float32)

        # ===== 3chマスクを作成 (ch0=left, ch1=center, ch2=right) =====
        mask_3ch = np.stack([mask_left, mask_center, mask_right], axis=0)  # (3, H, W)

        # ===== カラー overlay 画像の作成 =====
        H, W = mask_left.shape
        overlay = img_bgr.copy()
        alpha = 0.5

        left_bool   = mask_left.astype(bool)
        center_bool = mask_center.astype(bool)
        right_bool  = mask_right.astype(bool)

        # left: 赤
        overlay[left_bool] = (
            overlay[left_bool] * (1.0 - alpha)
            + np.array(LEFT_COLOR_BGR, dtype=np.float32) * alpha
        )
        # center: 緑
        overlay[center_bool] = (
            overlay[center_bool] * (1.0 - alpha)
            + np.array(CENTER_COLOR_BGR, dtype=np.float32) * alpha
        )
        # right: 青
        overlay[right_bool] = (
            overlay[right_bool] * (1.0 - alpha)
            + np.array(RIGHT_COLOR_BGR, dtype=np.float32) * alpha
        )

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # ===== 可視化用のマスク画像（背景黒 + left赤 / center緑 / right青） =====
        vis = np.zeros((H, W, 3), dtype=np.uint8)
        vis[left_bool]   = LEFT_COLOR_BGR
        vis[center_bool] = CENTER_COLOR_BGR
        vis[right_bool]  = RIGHT_COLOR_BGR

        # ===== 出力パス =====
        stem, ext = os.path.splitext(fname)
        out_img_path     = os.path.join(out_dir, f"{stem}_beammask_3ch.png")
        out_npy_path     = os.path.join(out_dir, f"{stem}_beam3ch.npy")
        out_overlay_path = os.path.join(out_dir, f"{stem}_overlay_3ch.png")

        # 保存
        cv2.imwrite(out_img_path, vis)
        cv2.imwrite(out_overlay_path, overlay)
        np.save(out_npy_path, mask_3ch.astype(np.float32))

        print(f"[{idx}/{len(files)}] Saved: {out_img_path}, {out_npy_path}, {out_overlay_path}")

    print(f"/n✅ All done. Saved 3ch masks to: {out_dir}")


if __name__ == "__main__":
    main()
