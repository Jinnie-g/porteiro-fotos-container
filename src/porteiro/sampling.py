"""E6: amostra estratificada para rotulagem + subconjunto de calibração (κ)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import ATRIBUTOS

COLUNAS_ANOTACAO = (["anotador", "versao_guia"] + list(ATRIBUTOS)
                    + ["n_castanhas", "n_vertices", "n_direcoes", "pontos_bem_espalhados",
                       "indeterminado", "face", "tipo_conteiner", "observacao"])


def labeling_sample(df: pd.DataFrame, n: int = 800, random_frac: float = 0.3,
                    calib_n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Só representantes de cada cluster de duplicatas, fora do teste fixo.

    ~30% puramente aleatório; o restante estratificado por fonte × tercil de
    nitidez × tercil da área da maior caixa (proxy grosseiro de zoom).
    """
    rng = np.random.default_rng(seed)
    pool = df[df["is_representative"] & (df["split_novo"] == "dev")].copy()
    # a planificação é avaliada em fotos reais: evita cópias com augmentation quando há alternativa
    pool = pool[~pool["aug_geometrica"] & (pool["cutout_n"] == 0)] if "aug_geometrica" in pool else pool
    pool["t_nitidez"] = pd.qcut(pool["blur_pct_fonte"].rank(method="first"), 3, labels=["baixa", "media", "alta"])
    pool["t_area"] = pd.qcut(pool["caixa_area_max"].rank(method="first"), 3, labels=["pequena", "media", "grande"])
    pool["estrato_rot"] = pool["source_id"] + "|" + pool["t_nitidez"].astype(str) + "|" + pool["t_area"].astype(str)
    n = min(n, len(pool))
    n_rand = int(round(n * random_frac))
    rand_ids = rng.choice(pool.index, n_rand, replace=False)
    rest = pool.drop(rand_ids)
    k = n - n_rand
    strata = rest.groupby("estrato_rot")
    quota = max(1, k // strata.ngroups)
    picked = []
    for _, g in strata:
        picked.extend(rng.choice(g.index, min(quota, len(g)), replace=False))
    if len(picked) < k:
        left = rest.drop(picked).index
        picked.extend(rng.choice(left, k - len(picked), replace=False))
    picked = picked[:k]
    out = pd.concat([pool.loc[rand_ids].assign(origem_amostra="aleatoria"),
                     pool.loc[picked].assign(origem_amostra="estratificada")])
    out = out.sample(frac=1, random_state=seed)
    out["calibracao_kappa"] = False
    out.iloc[:calib_n, out.columns.get_loc("calibracao_kappa")] = True
    base = out[["image_id", "rf_nome", "nome_origem", "source_id", "container_code", "origem_amostra",
                "estrato_rot", "calibracao_kappa"]].reset_index(drop=True)
    for c in COLUNAS_ANOTACAO:
        base[c] = ""
    return base
