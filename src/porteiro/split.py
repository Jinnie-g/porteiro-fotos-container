"""E5: split agrupado e estratificado. Fold 0 = teste fixo (nunca usado para
escolher amostras ou limiares); os demais formam o `dev` para validação cruzada."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def strat_label(df: pd.DataFrame) -> pd.Series:
    """Estrato = fonte + classe de avaria predominante (com classes raras agrupadas)."""
    cls = df[["n_rust", "n_dent", "n_hole"]].fillna(0)
    main = cls.idxmax(axis=1).str.replace("n_", "", regex=False).where(cls.sum(axis=1) > 0, "sem_caixa")
    lab = df["source_id"].astype(str) + "|" + main
    vc = lab.value_counts()
    return lab.where(lab.map(vc) >= 30, "raro|" + main)


def assign_folds(df: pd.DataFrame, n_splits: int = 6, seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    gsize = df["group_id"].value_counts()
    limit = len(df) / n_splits
    if gsize.max() > limit:
        raise ValueError(f"Grupo com {gsize.max()} imagens não cabe em um fold (~{limit:.0f}). "
                         "Revise os limiares de E3 antes de dividir.")
    y = strat_label(df)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df["fold"] = -1
    for k, (_, te) in enumerate(sgkf.split(df, y, groups=df["group_id"])):
        df.iloc[te, df.columns.get_loc("fold")] = k
    df["split_novo"] = np.where(df["fold"] == 0, "teste_fixo", "dev")
    df["estrato"] = y
    return df


def check_no_leak(df: pd.DataFrame) -> None:
    for col in ("group_id", "dup_cluster_id", "sha256"):
        n = df.groupby(col)["fold"].nunique()
        assert (n == 1).all(), f"{col} aparece em mais de um fold"
