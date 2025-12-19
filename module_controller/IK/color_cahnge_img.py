import numpy as np
from PIL import Image
def recolor_from_rb(arr_rgb: np.ndarray, gamma=0.8, gain_only=1.2, gain_overlap=1.0) -> np.ndarray:
    """
    加算済みRGBから、R/Bの共通成分を推定して
      - R only   -> Red
      - B only   -> Blue
      - overlap  -> Green
    に再配色する。
    arr_rgb: (H, W, 3) uint8 RGB
    """
    x = arr_rgb.astype(np.float32)
    R, G, B = x[..., 0], x[..., 1], x[..., 2]
    # 重なり（紫の元）
    overlap = np.minimum(R, B)
    # 単色成分
    r_only = np.clip(R - overlap, 0, 255)
    b_only = np.clip(B - overlap, 0, 255)
    # 強調
    r_only = np.clip(r_only * gain_only, 0, 255)
    b_only = np.clip(b_only * gain_only, 0, 255)
    overlap = np.clip(overlap * gain_overlap, 0, 255)
    # 再配色：R-only→赤, overlap→緑, B-only→青
    out = np.stack([r_only, overlap, b_only], axis=-1)
    # ガンマ補正（見やすさ）
    out01 = np.clip(out / 255.0, 0.0, 1.0)
    out01 = out01 ** gamma
    return (out01 * 255.0).astype(np.uint8)
def recolor_image(
    in_img: str,
    out_img: str,
    gamma: float = 0.8,
    gain_only: float = 1.3,
    gain_overlap: float = 0.9,
):
    """
    1枚画像を再配色して保存
    """
    img = Image.open(in_img).convert("RGB")
    arr = np.array(img)
    out_arr = recolor_from_rb(
        arr,
        gamma=gamma,
        gain_only=gain_only,
        gain_overlap=gain_overlap,
    )
    Image.fromarray(out_arr, mode="RGB").save(out_img)
    print("saved:", out_img)
if __name__ == "__main__":
    in_img  = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_sorted\maxmod_5\K1.0_LOOP10_RESET1_WAIT3.0_LOOPWAIT5.0\selected_33\fb_step_10\roi_overlay_red_orig_blue_cap.png"                 # ← 入力画像
    out_img = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control_3D\IK_only_result_sorted\maxmod_5\K1.0_LOOP10_RESET1_WAIT3.0_LOOPWAIT5.0\selected_33\fb_step_10\roi_overlay_red_orig_blue_cap_green.png"  # ← 出力画像
    recolor_image(
        in_img,
        out_img,
        gamma=0.8,
        gain_only=1.3,
        gain_overlap=0.9,
    )