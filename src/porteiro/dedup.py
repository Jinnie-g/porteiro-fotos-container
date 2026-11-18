"""E3 (duplicatas e frames repetidos) e E5 (auditoria de vazamento do split).

- Exatas: SHA-256.
- Quase duplicatas: pHash 64 bits (Hamming) E cosseno de embeddings
  centralizados (a regra "E" evita grupos gigantes por encadeamento).
- Cópias da mesma foto de origem (nome sem sufixos do Roboflow).
- group_id: une clusters de duplicatas e o código ISO 6346 do contêiner,
  porque o mesmo contêiner em ângulos diferentes não é duplicata, mas não
  pode estar em treino e teste ao mesmo tempo.
"""
from __future__ import annotations

from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)

    def labels(self) -> np.ndarray:
        return np.array([self.find(i) for i in range(len(self.p))])


def compute_phash(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    """pHash da imagem e das versões espelhadas (a augmentation aplica flips)."""
    def one(p):
        try:
            with Image.open(Path(root) / p) as im:
                im = im.convert("RGB")
                return (str(imagehash.phash(im)),
                        str(imagehash.phash(im.transpose(Image.FLIP_LEFT_RIGHT))),
                        str(imagehash.phash(im.transpose(Image.FLIP_TOP_BOTTOM))))
        except Exception:
            return (None, None, None)
    v = df["caminho"].map(one)
    return pd.DataFrame(v.tolist(), columns=["phash", "phash_hflip", "phash_vflip"], index=df.index)


def phash_to_u64(h: pd.Series) -> np.ndarray:
    return np.array([int(x, 16) if isinstance(x, str) else 0 for x in h], dtype=np.uint64)


def _bits(codes: np.ndarray) -> np.ndarray:
    return np.unpackbits(codes.view(np.uint8).reshape(len(codes), 8), axis=1).astype(np.uint8)


def hamming_matrix_pairs(codes: np.ndarray, max_d: int, chunk: int = 512, flips: list | None = None):
    """Pares (i<j) com distância de Hamming <= max_d, sem matriz n×n inteira em memória.

    Com `flips`, a distância é o mínimo entre a imagem e as versões espelhadas.
    """
    n = len(codes)
    bits = _bits(codes)
    fbits = [_bits(f) for f in (flips or [])]
    out = []
    for s in range(0, n, chunk):
        a = bits[s:s + chunk]
        d = (a[:, None, :] != bits[None, :, :]).sum(-1)
        for fb in fbits:
            d = np.minimum(d, (fb[s:s + chunk][:, None, :] != bits[None, :, :]).sum(-1))
        ii, jj = np.nonzero(d <= max_d)
        ii = ii + s
        keep = jj > ii
        out.append(np.stack([ii[keep], jj[keep], d[ii[keep] - s, jj[keep]]], 1))
    return np.concatenate(out) if out else np.zeros((0, 3), int)


def center_normalize(emb: np.ndarray) -> np.ndarray:
    e = emb - emb.mean(0, keepdims=True)
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)


def cosine_pairs(e: np.ndarray, min_cos: float, chunk: int = 1024):
    out = []
    for s in range(0, len(e), chunk):
        sim = e[s:s + chunk] @ e.T
        ii, jj = np.nonzero(sim >= min_cos)
        keep = jj > ii + s
        out.append(np.stack([ii[keep] + s, jj[keep], sim[ii[keep], jj[keep]]], 1))
    return np.concatenate(out) if out else np.zeros((0, 3))


def build_groups(df: pd.DataFrame, emb: np.ndarray, cfg: dict) -> tuple[pd.DataFrame, dict, dict]:
    df = df.reset_index(drop=True).copy()
    n = len(df)
    e = center_normalize(emb)
    codes = phash_to_u64(df["phash"])

    flips = [phash_to_u64(df[c]) for c in ("phash_hflip", "phash_vflip") if c in df]
    hp = hamming_matrix_pairs(codes, cfg["phash_max_hamming"], flips=flips)
    cp = cosine_pairs(e, cfg["cosine_min"])
    hset = {(int(i), int(j)) for i, j, _ in hp}
    cset = {(int(i), int(j)) for i, j, _ in cp}
    near = (hset & cset) if cfg.get("require_both", True) else (hset | cset)

    uf = UnionFind(n)
    for _, idx in df.groupby("sha256").groups.items():
        idx = list(idx)
        for k in idx[1:]:
            uf.union(idx[0], k)
    n_exact_extra = int(df.duplicated("sha256").sum())
    for i, j in near:
        uf.union(i, j)
    # cópias aumentadas da mesma foto: mesmo nome de origem E embedding parecido.
    # Nomes numéricos ("106", "2") colidem entre fotos diferentes de datasets
    # mesclados; o cosseno separa essas colisões. O grupo do split continua
    # usando só o nome (conservador).
    raw = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    same_name_min = cfg.get("same_name_cosine_min", 0.80)
    for _, idx in df.groupby("nome_origem").indices.items():
        if len(idx) < 2:
            continue
        S = raw[idx] @ raw[idx].T
        a, b = np.nonzero(np.triu(S >= same_name_min, 1))
        for x, y in zip(a, b):
            uf.union(int(idx[x]), int(idx[y]))
    df["dup_cluster_id"] = uf.labels()

    # group_id = dup clusters + mesmo contêiner (ISO 6346)
    uf2 = UnionFind(n)
    for _, idx in df.groupby("dup_cluster_id").groups.items():
        idx = list(idx)
        for k in idx[1:]:
            uf2.union(idx[0], k)
    for _, idx in df.groupby("nome_origem").groups.items():
        idx = list(idx)
        for k in idx[1:]:
            uf2.union(idx[0], k)
    for _, idx in df.dropna(subset=["container_code"]).groupby("container_code").groups.items():
        idx = list(idx)
        for k in idx[1:]:
            uf2.union(idx[0], k)
    df["group_id"] = uf2.labels()

    # representante por cluster: maior nitidez (blur_pct_fonte) quando disponível
    score = df["blur_pct_fonte"] if "blur_pct_fonte" in df else pd.Series(0.0, index=df.index)
    if "aug_geometrica" in df:
        score = score - 2 * df["aug_geometrica"].astype(float) - (df["cutout_n"] > 0).astype(float)
    best = score.groupby(df["dup_cluster_id"]).idxmax()
    df["is_representative"] = False
    df.loc[best.values, "is_representative"] = True

    csize = df["dup_cluster_id"].value_counts()
    gsize = df["group_id"].value_counts()
    stats = {
        "n_imagens": n,
        "exatas_extras": n_exact_extra,
        "pares_phash": len(hset), "pares_cosseno": len(cset), "pares_ambos": len(hset & cset),
        "pares_quase_dup_usados": len(near),
        "n_dup_clusters": int(csize.size),
        "imagens_em_cluster_maior_que_1": int(csize[csize > 1].sum()),
        "maior_dup_cluster": int(csize.max()),
        "n_grupos": int(gsize.size), "maior_grupo": int(gsize.max()),
        "frac_redundante": float(1 - csize.size / n),
        "n_nomes_origem": int(df["nome_origem"].nunique()),
    }
    return df, stats, {"hp": hp, "cp": cp}


def leakage_audit(df: pd.DataFrame, emb: np.ndarray, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Para cada imagem fora do treino, o vizinho mais próximo no treino."""
    df = df.reset_index(drop=True)
    e = center_normalize(emb)
    codes = phash_to_u64(df["phash"])
    tr = np.nonzero((df["split_orig"] == "train").values)[0]
    rows = []
    for split in ("valid", "test"):
        q = np.nonzero((df["split_orig"] == split).values)[0]
        if len(q) == 0:
            continue
        sim = e[q] @ e[tr].T
        nn = sim.argmax(1)
        best = sim[np.arange(len(q)), nn]
        bits_t = _bits(codes[tr[nn]])
        ham = (_bits(codes[q]) != bits_t).sum(1)
        for c in ("phash_hflip", "phash_vflip"):
            if c in df:
                ham = np.minimum(ham, (_bits(phash_to_u64(df[c])[q]) != bits_t).sum(1))
        for k, i in enumerate(q):
            j = tr[nn[k]]
            rows.append({"split": split, "image_id": df.at[i, "image_id"], "nn_train": df.at[j, "image_id"],
                         "cos": float(best[k]), "hamming": int(ham[k]),
                         "mesmo_sha": df.at[i, "sha256"] == df.at[j, "sha256"],
                         "mesmo_nome_origem": df.at[i, "nome_origem"] == df.at[j, "nome_origem"]})
    a = pd.DataFrame(rows)
    a["acima_limiar"] = (a["cos"] >= cfg["cosine_min"]) & (a["hamming"] <= cfg["phash_max_hamming"])

    def cross(col):
        s = df.dropna(subset=[col]).groupby(col)["split_orig"].nunique()
        ids = s[s > 1].index
        return int(len(ids)), int(df[df[col].isin(ids)].shape[0])
    stats = {}
    for col in ("sha256", "nome_origem", "dup_cluster_id", "container_code", "group_id"):
        ng, ni = cross(col)
        stats[f"{col}_em_mais_de_um_split"] = {"grupos": ng, "imagens": ni}
    for split in ("valid", "test"):
        s = a[a["split"] == split]
        stats[f"{split}_vizinho_treino_acima_limiar"] = {
            "n": int(s["acima_limiar"].sum()), "de": int(len(s)),
            "frac": float(s["acima_limiar"].mean()) if len(s) else 0.0,
            "mesma_foto_origem": int(s["mesmo_nome_origem"].sum()),
            "identicas_sha": int(s["mesmo_sha"].sum())}
    return a, stats


def review_pairs(df: pd.DataFrame, emb: np.ndarray, rng: np.random.Generator,
                 per_band: int = 40) -> pd.DataFrame:
    """Pares amostrados por faixa de distância para calibrar o limiar à mão (E3.3)."""
    df = df.reset_index(drop=True)
    e = center_normalize(emb)
    codes = phash_to_u64(df["phash"])
    n = len(df)
    i = rng.integers(0, n, 200_000)
    j = rng.integers(0, n, 200_000)
    keep = i != j
    i, j = i[keep], j[keep]
    cos = (e[i] * e[j]).sum(1)
    # completa com vizinhos próximos (pares aleatórios quase nunca são parecidos)
    nn_i, nn_j = [], []
    for s in range(0, n, 1024):
        sim = e[s:s + 1024] @ e.T
        sim[np.arange(sim.shape[0]), np.arange(s, s + sim.shape[0])] = -1
        nn_i.append(np.arange(s, s + sim.shape[0]))
        nn_j.append(sim.argmax(1))
    i = np.concatenate([i, *nn_i])
    j = np.concatenate([j, *nn_j])
    cos = (e[i] * e[j]).sum(1)
    bi = np.unpackbits(codes[i].view(np.uint8).reshape(-1, 8), axis=1)
    bj = np.unpackbits(codes[j].view(np.uint8).reshape(-1, 8), axis=1)
    ham = (bi != bj).sum(1)
    bands = [(0.97, 1.01), (0.95, 0.97), (0.92, 0.95), (0.88, 0.92), (0.80, 0.88)]
    rows = []
    for lo, hi in bands:
        idx = np.nonzero((cos >= lo) & (cos < hi))[0]
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, min(per_band, len(idx)), replace=False)
        for k in pick:
            rows.append({"faixa_cos": f"[{lo:.2f},{min(hi, 1):.2f})", "a": df.at[i[k], "image_id"],
                         "b": df.at[j[k], "image_id"], "cos": float(cos[k]), "hamming": int(ham[k]),
                         "julgamento": ""})
    return pd.DataFrame(rows).drop_duplicates(subset=["a", "b"])
