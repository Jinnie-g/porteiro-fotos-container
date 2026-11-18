"""S0 / seção 15 (D4, D8): dataset consolidado `unico` como VISÃO do manifesto.

Nenhum arquivo é apagado nem alterado: o consolidado é o filtro
`is_representative = True`, com o motivo da escolha registrado. Os invariantes
abaixo são verificados a cada execução (D8).
"""
from __future__ import annotations

import pandas as pd


def representative_reason(df: pd.DataFrame) -> pd.Series:
    clean = ~df["aug_geometrica"] & (df["cutout_n"] == 0)
    size = df.groupby("dup_cluster_id")["image_id"].transform("size")
    has_clean = clean.groupby(df["dup_cluster_id"]).transform("any")
    reason = pd.Series("foto_unica", index=df.index)
    reason[(size > 1) & clean] = "copia_limpa_mais_nitida"
    reason[(size > 1) & ~clean & ~has_clean] = "sem_copia_limpa_mais_nitida"
    reason[(size == 1) & ~clean] = "foto_unica_com_artefato"
    return reason.where(df["is_representative"], "copia")


def build_unico(manifest: pd.DataFrame) -> pd.DataFrame:
    df = manifest.copy()
    df["motivo_representante"] = representative_reason(df)
    df["n_copias_cluster"] = df.groupby("dup_cluster_id")["image_id"].transform("size")
    df["artefato_aug"] = df["aug_geometrica"] | (df["cutout_n"] > 0)
    unico = df[df["is_representative"]].copy()
    check_invariants(df, unico)
    return unico


def check_invariants(full: pd.DataFrame, unico: pd.DataFrame) -> None:
    """D8: propriedades que toda versão do consolidado precisa manter."""
    # 1) exatamente um representante por cluster
    assert unico["dup_cluster_id"].is_unique, "cluster com mais de um representante"
    assert set(unico["dup_cluster_id"]) == set(full["dup_cluster_id"]), "cluster sem representante"
    # 2) nada some: toda imagem do manifesto continua lá
    assert full["image_id"].is_unique
    # 3) o representante prefere cópia sem artefato quando o cluster tem uma
    clean = ~full["aug_geometrica"] & (full["cutout_n"] == 0)
    has_clean = clean.groupby(full["dup_cluster_id"]).any()
    rep_clean = (~unico["aug_geometrica"] & (unico["cutout_n"] == 0)).groupby(unico["dup_cluster_id"]).any()
    bad = has_clean & ~rep_clean.reindex(has_clean.index, fill_value=False)
    assert not bad.any(), f"{int(bad.sum())} clusters com cópia limpa e representante sujo"
    # 4) um cluster nunca atravessa folds
    assert (full.groupby("dup_cluster_id")["fold"].nunique() == 1).all()
