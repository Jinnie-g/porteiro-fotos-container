"""E7 / S6: embeddings DINOv2 congelados (ViT-S/14, pesos oficiais da Meta).

- Sem recorte central: a imagem inteira é redimensionada (os cantos da face são
  justamente o que decide o zoom; o recorte padrão os cortaria).
- Lado em múltiplo de 14 (224 ou 448); o ViT interpola as posições.
- Saídas por imagem: CLS e média dos patch tokens (float16 em cache).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_model(weights: Path):
    import timm
    from timm.models.vision_transformer import checkpoint_filter_fn
    m = timm.create_model("vit_small_patch14_dinov2", pretrained=False, img_size=518, dynamic_img_size=True)
    sd = checkpoint_filter_fn(torch.load(weights, map_location="cpu"), m)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"pesos incompatíveis: {missing} {unexpected}")
    return m.eval()


def preprocess(img: Image.Image, side: int) -> torch.Tensor:
    img = img.convert("RGB").resize((side, side), Image.BICUBIC)
    a = (np.asarray(img, np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(a.transpose(2, 0, 1))


@torch.inference_mode()
def extract(model, images: list, side: int = 448, batch: int = 16) -> dict[str, np.ndarray]:
    """images: lista de PIL.Image ou caminhos."""
    cls, mean = [], []
    for k in range(0, len(images), batch):
        part = [Image.open(p) if not isinstance(p, Image.Image) else p for p in images[k:k + batch]]
        x = torch.stack([preprocess(im, side) for im in part])
        tok = model.forward_features(x)  # [B, 1 + N, D]
        cls.append(tok[:, 0].float().numpy())
        mean.append(tok[:, 1:].mean(1).float().numpy())
    return {"cls": np.concatenate(cls).astype(np.float16), "mean": np.concatenate(mean).astype(np.float16)}
