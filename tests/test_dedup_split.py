import numpy as np
import pandas as pd
import pytest

from porteiro import dedup, split


def _df(n=60, seed=0):
    rng = np.random.default_rng(seed)
    codes = [f"{int(x):016x}" for x in rng.integers(0, 2 ** 63, n, dtype=np.int64)]
    df = pd.DataFrame({
        "image_id": [f"i{k}" for k in range(n)], "sha256": [f"s{k}" for k in range(n)],
        "nome_origem": [f"foto{k}" for k in range(n)], "container_code": [None] * n,
        "phash": codes, "split_orig": rng.choice(["train", "valid", "test"], n),
        "source_id": ["a"] * n, "n_rust": rng.integers(0, 3, n), "n_dent": rng.integers(0, 3, n),
        "n_hole": rng.integers(0, 3, n),
    })
    emb = rng.normal(size=(n, 32)).astype(np.float32)
    return df, emb


CFG = {"phash_max_hamming": 6, "cosine_min": 0.95, "require_both": True, "same_name_cosine_min": 0.8}


def test_exact_and_same_name_copies_are_clustered():
    df, emb = _df()
    df.loc[1, "sha256"] = df.loc[0, "sha256"]
    df.loc[3, "nome_origem"] = df.loc[2, "nome_origem"]
    emb[3] = emb[2] + 0.01
    df.loc[5, "nome_origem"] = df.loc[4, "nome_origem"]  # colisão de nome, imagens diferentes
    out, stats, _ = dedup.build_groups(df, emb, CFG)
    c = out.set_index("image_id")["dup_cluster_id"]
    g = out.set_index("image_id")["group_id"]
    assert c["i0"] == c["i1"] and c["i2"] == c["i3"]
    assert c["i4"] != c["i5"] and g["i4"] == g["i5"]  # separa no dedup, une no split
    assert stats["exatas_extras"] == 1


def test_flip_invariant_phash_pairs():
    codes = np.array([0x0F0F0F0F0F0F0F0F, 0xF0F0F0F0F0F0F0F0], dtype=np.uint64)
    flips = [np.array([0xF0F0F0F0F0F0F0F0, 0x0F0F0F0F0F0F0F0F], dtype=np.uint64)]
    assert len(dedup.hamming_matrix_pairs(codes, 6)) == 0
    assert len(dedup.hamming_matrix_pairs(codes, 6, flips=flips)) == 1


def test_grouped_split_has_no_leak_and_rejects_giant_group():
    df, emb = _df(120)
    df.loc[:9, "nome_origem"] = "mesma"
    emb[:10] = emb[0]
    out, _, _ = dedup.build_groups(df, emb, CFG)
    f = split.assign_folds(out, n_splits=4)
    split.check_no_leak(f)
    assert f.loc[f["nome_origem"] == "mesma", "fold"].nunique() == 1
    out.loc[:, "group_id"] = 0
    with pytest.raises(ValueError):
        split.assign_folds(out, n_splits=4)
