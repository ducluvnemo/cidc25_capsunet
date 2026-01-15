# file: challenge/predict.py

import os
import sys
from pathlib import Path

import torch
import numpy as np
import tifffile as tiff

# PROJECT_ROOT = .../cidc25_casupnet
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from support_src.casup_wrapper import build_casupnet_from_L1


# === ĐƯỜNG DẪN CHUẨN CHO GRAND CHALLENGE ===
# CHÚ Ý: kiểm tra lại 2 path này trên trang Submit của phase (nút "i" màu xanh).
INPUT_FOLDER = Path("/input/images/stacked-neuron-images-with-noise/")
OUTPUT_FOLDER = Path("/output/images/stacked-neuron-images-with-reduced-noise/")


def denoise_one_file(in_path: Path, out_path: Path, model, device):
    """Denoise một stack TIFF và lưu kết quả sang out_path."""
    # Đọc stack noisy: [T_all, H, W]
    stack = tiff.imread(str(in_path)).astype("float32")
    T_all, H, W = stack.shape

    window = 61
    half = window // 2  # 30
    denoised_stack = np.zeros_like(stack, dtype=np.float32)

    with torch.no_grad():
        # Biên: copy noisy
        denoised_stack[:half] = stack[:half]
        denoised_stack[T_all - half :] = stack[T_all - half :]

        # Trượt cửa sổ 61 frame, mỗi lần model trả ra 1 frame
        for t in range(half, T_all - half):
            clip = stack[t - half : t + half + 1]  # [61, H, W]
            clip_t = torch.from_numpy(clip).unsqueeze(0).to(device)  # [1, 61, H, W]

            pred = model(clip_t)  # [1, 1, H, W]
            pred_np = pred.squeeze().cpu().numpy().astype("float32")  # [H, W]

            denoised_stack[t] = pred_np

    # Ghi TIFF có metadata resolution để Grand Challenge validate được
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(
        str(out_path),
        denoised_stack.astype("float32"),
        resolution=(300, 300),
    )


def main():
    # Thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load đúng model L1 bằng wrapper
    model_file = os.path.join(
        "support_src", "src", "GUI", "trained_models", "L1_generalization.pth"
    )
    model = build_casupnet_from_L1(model_file=model_file, device=device)
    model.eval()

    # Đảm bảo thư mục output tồn tại
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Lặp qua TẤT CẢ file .tif trong input
    tif_paths = sorted(INPUT_FOLDER.glob("*.tif"))
    if not tif_paths:
        print(f"No .tif files found in {INPUT_FOLDER}")
        return

    for in_path in tif_paths:
        out_path = OUTPUT_FOLDER / in_path.name
        print(f"Processing {in_path} -> {out_path}")
        denoise_one_file(in_path, out_path, model, device)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
