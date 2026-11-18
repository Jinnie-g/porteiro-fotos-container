"""S3: amostragem para rotulagem do zoom (só representantes de cluster)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def tercil(s: pd.Series, labels=("t1", "t2", "t3")) -> pd.Series:
    return pd.qcut(s.rank(method="first"), 3, labels=list(labels)).astype(str)


def sample_for_zoom(unico: pd.DataFrame, full: pd.DataFrame, n_dev: int = 400, n_test: int = 160,
                    random_frac: float = 0.3, n_retest: int = 100, n_aug_clusters: int = 30,
                    seed: int = 42) -> pd.DataFrame:
    """Rodadas:
    - `principal`: dev (30% aleatório + estratificado por fonte × tercil do proxy × nitidez)
      e teste fixo (aleatório, para confirmação);
    - `reteste`: 100 já rotuladas, revistas às cegas em outra ordem (consistência);
    - `variacao_aug`: 2-3 cópias de clusters com augmentation (quanto a augmentation muda o zoom).
    """
    rng = np.random.default_rng(seed)
    u = unico.copy()
    u["t_proxy"] = tercil(u["caixa_area_max"])
    u["t_nitidez"] = tercil(u["blur_pct_fonte"])
    u["estrato_zoom"] = u["source_id"] + "|" + u["t_proxy"] + "|" + u["t_nitidez"]

    dev = u[u["split_novo"] == "dev"]
    n_rand = int(round(n_dev * random_frac))
    rand_ids = list(rng.choice(dev["image_id"], n_rand, replace=False))
    rest = dev[~dev["image_id"].isin(rand_ids)]
    k = n_dev - n_rand
    groups = rest.groupby("estrato_zoom")
    quota = max(1, k // groups.ngroups)
    strat = []
    for _, g in groups:
        strat.extend(rng.choice(g["image_id"], min(quota, len(g)), replace=False))
    if len(strat) < k:
        left = rest[~rest["image_id"].isin(strat)]["image_id"]
        strat.extend(rng.choice(left, k - len(strat), replace=False))
    strat = strat[:k]
    test = u[u["split_novo"] == "teste_fixo"]
    test_ids = list(rng.choice(test["image_id"], min(n_test, len(test)), replace=False))

    rows = ([{"image_id": i, "rodada": "principal", "origem_amostra": "aleatoria"} for i in rand_ids]
            + [{"image_id": i, "rodada": "principal", "origem_amostra": "estratificada"} for i in strat]
            + [{"image_id": i, "rodada": "principal", "origem_amostra": "teste_fixo"} for i in test_ids])
    main_ids = rand_ids + strat
    for i in rng.choice(main_ids, n_retest, replace=False):
        rows.append({"image_id": i, "rodada": "reteste", "origem_amostra": "reteste"})

    # variação da augmentation: clusters com >= 3 cópias e ao menos uma com artefato
    f = full.copy()
    f["art"] = f["aug_geometrica"] | (f["cutout_n"] > 0)
    cand = f.groupby("dup_cluster_id").filter(lambda g: len(g) >= 3 and g["art"].any())
    cids = rng.choice(cand["dup_cluster_id"].unique(), n_aug_clusters, replace=False)
    for c in cids:
        g = cand[cand["dup_cluster_id"] == c].sort_values("is_representative", ascending=False)
        for i in g["image_id"].head(3):
            rows.append({"image_id": i, "rodada": "variacao_aug", "origem_amostra": f"cluster_{c}"})
    out = pd.DataFrame(rows)
    out = out.merge(f[["image_id", "dup_cluster_id", "source_id", "split_novo"]], on="image_id", how="left")
    # ordem de apresentação embaralhada por rodada (o anotador não vê estrato nem fonte)
    out = out.sample(frac=1, random_state=seed).sort_values("rodada", kind="stable")
    out["ordem"] = out.groupby("rodada").cumcount()
    return out.reset_index(drop=True)
