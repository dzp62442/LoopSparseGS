from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from PIL import Image

from lpipsPyTorch import LPIPS
from utils.image_utils import psnr
from utils.loss_utils import ssim


class SavedImageMetricEvaluator:
    """Evaluate saved render/GT pairs on an explicitly selected device."""

    def __init__(self, device: str = "cpu", batch_size: int = 12):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，无法使用 --metric_device cuda")
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正数")
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.lpips_metric = LPIPS(net_type="vgg").to(self.device).eval()

    @staticmethod
    def _load_rgb(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

    def _lpips_per_image(self, render: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """The bundled LPIPS forward only handles batch size one; preserve its math in batch."""
        render_features = self.lpips_metric.net(render)
        gt_features = self.lpips_metric.net(gt)
        differences = [
            (render_feature - gt_feature) ** 2
            for render_feature, gt_feature in zip(render_features, gt_features)
        ]
        layer_scores = [
            linear_layer(difference).mean((2, 3), True)
            for linear_layer, difference in zip(self.lpips_metric.lin, differences)
        ]
        return torch.stack(layer_scores, dim=0).sum(dim=0).flatten()

    def evaluate(self, render_dir: Path, gt_dir: Path,
                 image_names: Sequence[str]) -> Dict[str, float]:
        if not image_names:
            raise ValueError("待评估图像列表不能为空")

        totals = {"l1": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
        image_count = 0
        with torch.inference_mode():
            for start in range(0, len(image_names), self.batch_size):
                batch_names = image_names[start:start + self.batch_size]
                renders = torch.stack([
                    self._load_rgb(render_dir / image_name)
                    for image_name in batch_names
                ]).to(self.device)
                gts = torch.stack([
                    self._load_rgb(gt_dir / image_name)
                    for image_name in batch_names
                ]).to(self.device)
                if renders.shape != gts.shape:
                    raise ValueError(
                        "render/GT 尺寸不一致: {} vs {}".format(
                            tuple(renders.shape), tuple(gts.shape)
                        )
                    )

                batch_count = len(batch_names)
                totals["l1"] += torch.abs(renders - gts).mean((1, 2, 3)).sum().item()
                totals["psnr"] += psnr(renders, gts).sum().item()
                totals["ssim"] += ssim(
                    renders, gts, size_average=False
                ).sum().item()
                totals["lpips"] += self._lpips_per_image(renders, gts).sum().item()
                image_count += batch_count

        return {
            metric_name: total / image_count
            for metric_name, total in totals.items()
        }
