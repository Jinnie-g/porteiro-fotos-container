"""E0: download do dataset do Roboflow sem gerar versão.

Usa a API de busca para listar as imagens (com o split atribuído no Roboflow)
e o endpoint de cada imagem para obter caixas, URL do original e o embedding
que o próprio Roboflow calcula. Nada é alterado no projeto remoto.

Estrutura gerada em data_root:
  images/<image_id>.jpg       original como armazenado no Roboflow
  meta/<image_id>.json        resposta completa do endpoint da imagem
  index.json                  lista da busca (id, nome, split, contagens)
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

API = "https://api.roboflow.com"


def _get(url: str, params: dict | None = None, tries: int = 5, **kw) -> requests.Response:
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60, **kw)
            if r.status_code == 200:
                return r
            if r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status()
        except requests.RequestException:
            if i == tries - 1:
                raise
        time.sleep(2 ** i)
    r.raise_for_status()
    return r


def list_images(workspace: str, project: str, key: str, page: int = 250) -> list[dict]:
    url = f"{API}/{workspace}/{project}/search"
    out, offset, total = [], 0, None
    while total is None or offset < total:
        body = {"limit": page, "offset": offset,
                "fields": ["id", "name", "split", "annotations", "tags", "created"]}
        r = requests.post(url, params={"api_key": key}, json=body, timeout=120)
        r.raise_for_status()
        d = r.json()
        total = d["total"]
        out.extend(d["results"])
        offset += page
        if not d["results"]:
            break
    return out


def _fetch_one(workspace: str, project: str, key: str, image_id: str, root: Path) -> str:
    meta_p = root / "meta" / f"{image_id}.json"
    img_p = root / "images" / f"{image_id}.jpg"
    if meta_p.exists() and img_p.exists() and img_p.stat().st_size > 0:
        return "cached"
    meta = _get(f"{API}/{workspace}/{project}/images/{image_id}", {"api_key": key}).json()["image"]
    content = _get(meta["urls"]["original"]).content
    tmp = img_p.with_suffix(".part")
    tmp.write_bytes(content)
    tmp.rename(img_p)
    meta_p.write_text(json.dumps(meta), encoding="utf-8")
    return "ok"


def download(workspace: str, project: str, key: str, root: Path, workers: int = 16,
             limit: int | None = None) -> dict:
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    index = list_images(workspace, project, key)
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    ids = [x["id"] for x in index][:limit]
    stats = {"ok": 0, "cached": 0, "error": 0}
    errors = []
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_fetch_one, workspace, project, key, i, root): i for i in ids}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                stats[fut.result()] += 1
            except Exception as e:  # registra e segue; nada é descartado em silêncio
                stats["error"] += 1
                errors.append({"id": futs[fut], "error": repr(e)})
            if n % 500 == 0:
                print(f"  {n}/{len(ids)} {stats}", flush=True)
    (root / "download_errors.json").write_text(json.dumps(errors, indent=1), encoding="utf-8")
    return {"total_index": len(index), **stats}
