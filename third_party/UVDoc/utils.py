import os

import torch
import torch.nn.functional as F

from model import UVDocnet

IMG_SIZE = [488, 712]
GRID_SIZE = [45, 31]


def load_model(ckpt_path):
    """
    Load UVDocnet model.
    """
    model = UVDocnet(num_filter=32, kernel_size=5)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model_state"])
    return model


def get_version():
    """
    Returns the version of the various packages used for evaluation.
    """
    import pytesseract

    return {
        "tesseract": str(pytesseract.get_tesseract_version()),
        "pyesseract": os.popen("pip list | grep pytesseract").read().split()[-1],
        "Levenshtein": os.popen("pip list | grep Levenshtein").read().split()[-1],
        "jiwer": os.popen("pip list | grep jiwer").read().split()[-1],
        "matlabengineforpython": os.popen("pip list | grep matlab").read().split()[-1],
    }


def bilinear_unwarping(warped_img, point_positions, img_size):
    """
    Utility function that unwarps an image.
    Unwarp warped_img based on the 2D grid point_positions with a size img_size.
    Args:
        warped_img  :       torch.Tensor of shape BxCxHxW (dtype float)
        point_positions:    torch.Tensor of shape Bx2xGhxGw (dtype float)
        img_size:           tuple of int [w, h]
    """
    # NOTE(paper_pipeline_v18改善):
    # F.grid_sample の padding_mode デフォルトは 'zeros' のため、
    # 予測グリッドが画像外を参照すると外側が黒(0)で埋まり、
    # 「用紙の白い余白が削れた/黒ずんだ」ように見えることがある。
    # 端の欠けを抑制するため、border（端値延長）を使用する。
    upsampled_grid = F.interpolate(
        point_positions, size=(img_size[1], img_size[0]), mode="bilinear", align_corners=True
    )

    # ------------------------------------------------------------------
    # paper_pipeline_v18 改善（ユーザーFB: 余白/端文字が切れる）
    # ------------------------------------------------------------------
    # UVDoc が出力する grid は、端で「内側に寄る」ことがあり、結果として
    # 出力画像が用紙ギリギリ（端の文字が欠ける）になりやすい。
    # そこで、予測 grid の x/y をそれぞれ min/max で線形に再スケールし、
    # 画像全域が入力全域（正規化座標 [-1, 1]）を使うように正規化する。
    #
    # - これにより、端が内側に寄っていても「端が届く」方向に補正され、
    #   端の情報が欠けにくくなる。
    # - 副作用として、補正が強いケースでは歪みが増える可能性がある。
    #
    # grid_sample 形式 (B,H,W,2)
    grid = upsampled_grid.transpose(1, 2).transpose(2, 3)

    try:
        # x/y を別々に正規化（バッチごと）
        gx = grid[..., 0]
        gy = grid[..., 1]
        eps = 1e-6

        gx_min = gx.amin(dim=(1, 2), keepdim=True)
        gx_max = gx.amax(dim=(1, 2), keepdim=True)
        gy_min = gy.amin(dim=(1, 2), keepdim=True)
        gy_max = gy.amax(dim=(1, 2), keepdim=True)

        gx_span = (gx_max - gx_min).clamp_min(eps)
        gy_span = (gy_max - gy_min).clamp_min(eps)

        gx_center = (gx_max + gx_min) * 0.5
        gy_center = (gy_max + gy_min) * 0.5

        # min/max を -1/+1 に合わせる（中心保持 + スケール）
        gx = (gx - gx_center) * (2.0 / gx_span)
        gy = (gy - gy_center) * (2.0 / gy_span)

        grid = torch.stack([gx, gy], dim=-1)
        # 数値誤差のはみ出しはクランプ
        grid = grid.clamp(-1.0, 1.0)
    except Exception:
        # 正規化で落ちても unwarp 自体は継続（安全側）
        pass

    unwarped_img = F.grid_sample(
        warped_img,
        grid,
        align_corners=True,
        padding_mode="border",
    )

    return unwarped_img


def bilinear_unwarping_from_numpy(warped_img, point_positions, img_size):
    """
    Utility function that unwarps an image.
    Unwarp warped_img based on the 2D grid point_positions with a size img_size.
    Accept numpy arrays as input.
    """
    warped_img = torch.unsqueeze(torch.from_numpy(warped_img.transpose(2, 0, 1)).float(), dim=0)
    point_positions = torch.unsqueeze(torch.from_numpy(point_positions.transpose(2, 0, 1)).float(), dim=0)

    unwarped_img = bilinear_unwarping(warped_img, point_positions, img_size)

    unwarped_img = unwarped_img[0].numpy().transpose(1, 2, 0)
    return unwarped_img
