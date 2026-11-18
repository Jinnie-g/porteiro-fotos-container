"""E1 (inventário e rastreabilidade) e E2 (integridade).

Gera o manifesto: uma linha por imagem, com hash, origem, split original e
contagens das anotações. Nada é apagado: problemas viram colunas.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

# ISO 6346: 3 letras do dono + categoria (U/J/Z) + 6 dígitos + dígito verificador
ISO6346 = re.compile(r"([A-Z]{3}[UJZ])[\s_\-]?(\d{6})[\s_\-]?(\d)?")
UUID_PREFIX = re.compile(r"^[0-9a-f]{8}-")
RF_SUFFIX = re.compile(r"(_resize)?(_(jpe?g|JPE?G|png|PNG))*_jpg$")


def origin_name(rf_name: str) -> str:
    """Nome da foto de origem, sem prefixo do Label Studio nem sufixos do Roboflow."""
    n = UUID_PREFIX.sub("", rf_name)
    n = RF_SUFFIX.sub("", n)
    return n


def container_code(name: str) -> str | None:
    m = ISO6346.search(name.upper())
    return f"{m.group(1)}{m.group(2)}" if m else None


def source_id(name: str) -> str:
    """Família de origem inferida pelo padrão do nome (fonte = câmera/coleta)."""
    n = origin_name(name)
    rules = [
        (r"^aug_", "aug_preaumentada"),
        (r"^CAM\d", "cam_fixa"),
        (r"^(?i:frame)", "frame_video"),
        (r"^DSC[FN]", "camera_dsc"),
        (r"^IMG", "camera_img"),
        (r"^P\d", "camera_p"),
        (r"^\d+$", "numerica"),
    ]
    for pat, sid in rules:
        if re.search(pat, n):
            return sid
    if container_code(n):
        return "codigo_iso"
    if re.search(r"container", n, re.I):
        return "web_container"
    return "outra"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_integrity(path: Path) -> dict:
    """Decodificação completa + orientação EXIF + modo de cor."""
    out = {"integridade": "ok", "formato": None, "modo": None, "largura": None,
           "altura": None, "exif_orient": 1, "icc": False, "erro": None}
    try:
        with Image.open(path) as im:
            out["formato"] = im.format
            im.load()  # força decodificação completa (JPEG truncado falha aqui)
            out["modo"] = im.mode
            out["exif_orient"] = int(im.getexif().get(0x0112, 1) or 1)
            out["icc"] = bool(im.info.get("icc_profile"))
            fixed = ImageOps.exif_transpose(im)
            out["largura"], out["altura"] = fixed.size
            if im.mode != "RGB" or out["exif_orient"] != 1:
                out["integridade"] = "convertido"
    except Exception as e:
        out["integridade"] = "corrompido"
        out["erro"] = repr(e)[:200]
    return out


def _box_stats(meta: dict) -> dict:
    a = meta.get("annotation") or {}
    boxes = a.get("boxes") or []
    W, H = a.get("width") or 1, a.get("height") or 1
    areas = [b["width"] * b["height"] / (W * H) for b in boxes]
    counts = {c: 0 for c in ("rust", "dent", "hole")}
    for b in boxes:
        counts[b["label"]] = counts.get(b["label"], 0) + 1
    union_x0 = min((b["x"] - b["width"] / 2) / W for b in boxes) if boxes else None
    return {
        "n_caixas": len(boxes),
        **{f"n_{c}": v for c, v in counts.items()},
        "caixa_area_max": max(areas) if areas else 0.0,
        "caixa_area_soma": float(sum(areas)),
        "anot_largura": a.get("width"), "anot_altura": a.get("height"),
        "_tem_caixa": union_x0 is not None,
    }


def build_manifest(root: Path) -> pd.DataFrame:
    root = Path(root)
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    rows = []
    for item in index:
        iid = item["id"]
        img_p = root / "images" / f"{iid}.jpg"
        meta_p = root / "meta" / f"{iid}.json"
        row = {"image_id": iid, "caminho": str(img_p.relative_to(root)),
               "rf_nome": item["name"], "split_orig": item.get("split") or "none"}
        row["nome_origem"] = origin_name(item["name"])
        row["source_id"] = source_id(item["name"])
        row["container_code"] = container_code(row["nome_origem"])
        if not img_p.exists():
            row.update(integridade="ausente")
            rows.append(row)
            continue
        row["sha256"] = sha256(img_p)
        row["bytes"] = img_p.stat().st_size
        row.update(check_integrity(img_p))
        if meta_p.exists():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            row.update(_box_stats(meta))
        rows.append(row)
    df = pd.DataFrame(rows).drop(columns=["_tem_caixa"], errors="ignore")
    return df
