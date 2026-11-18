import numpy as np
import pandas as pd
import pytest

from porteiro import stats as S
from porteiro import zoom_model as Z
from porteiro.consolidate import build_unico, check_invariants


def test_kappa_perfect_and_chance():
    a = ["Z0", "Z1", "Z2", "Z1", "Z0"]
    assert S.cohen_kappa(a, a, ["Z0", "Z1", "Z2"]) == pytest.approx(1.0)
    assert S.cohen_kappa(a, a, ["Z0", "Z1", "Z2"], "linear") == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 2, 5000), rng.integers(0, 2, 5000)
    assert abs(S.cohen_kappa(x, y, [0, 1])) < 0.05


def test_weighted_kappa_penalizes_far_disagreement_more():
    a = ["Z0"] * 10 + ["Z2"] * 10
    near = ["Z1"] * 10 + ["Z2"] * 10
    far = ["Z2"] * 10 + ["Z2"] * 10
    lab = ["Z0", "Z1", "Z2"]
    assert S.cohen_kappa(a, near, lab, "linear") > S.cohen_kappa(a, far, lab, "linear")


def test_wilson_and_bh():
    p, lo, hi = S.wilson(19, 20)
    assert lo < p < hi <= 1 and lo > 0.7
    q = S.benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.all(q >= np.array([0.01, 0.04, 0.03, 0.5]) - 1e-12) and q[3] == pytest.approx(0.5)


def test_cramers_v_and_cliff():
    x = ["a"] * 50 + ["b"] * 50
    assert S.cramers_v(x, x)[0] == pytest.approx(1.0)
    assert S.cliffs_delta([5, 6, 7], [1, 2, 3]) == pytest.approx(1.0)


def test_thresholds_reach_target_precision_and_route():
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(40), np.zeros(160)].astype(int)
    p = np.clip(np.r_[rng.normal(0.8, 0.1, 40), rng.normal(0.2, 0.15, 160)], 0, 1)
    t = Z.choose_thresholds(y, p, 0.95)
    sel = p >= t["tau_alto"]
    assert y[sel].mean() >= 0.95 and (1 - y[p < t["tau_baixo"]]).mean() >= 0.95
    r = Z.route(p, t)
    assert set(r) <= {"sem_zoom", "com_zoom", "revisao"}
    assert (r == "sem_zoom").sum() == sel.sum()


def test_platt_model_learns_separable_signal_without_test_leak():
    rng = np.random.default_rng(2)
    n = 240
    y = (rng.random(n) < 0.25).astype(int)
    X = rng.normal(size=(n, 5)) + y[:, None] * 2.0
    g = np.arange(n) // 2
    o = Z.repeated_oof(X, y, g, lambda r: Z.PlattModel("lr", seed=r), n_splits=4, n_repeats=1)
    m = Z.metrics(y, o[0])
    assert m["pr_auc"] > 0.9 and 0 <= m["ece"] < 0.15
    # ruído puro: o OOF não pode parecer bom (sinal de vazamento)
    Xn = rng.normal(size=(n, 5))
    on = Z.repeated_oof(Xn, y, g, lambda r: Z.PlattModel("lr", seed=r), n_splits=4, n_repeats=1)
    assert Z.metrics(y, on[0])["pr_auc"] < 0.5


def _manifest():
    rows = []
    for c in range(6):
        for k in range(3):
            rows.append({"image_id": f"{c}_{k}", "dup_cluster_id": c, "fold": c % 3,
                         "aug_geometrica": k > 0, "cutout_n": 0, "is_representative": k == 0})
    return pd.DataFrame(rows)


def test_unico_invariants_hold_and_detect_violation():
    m = _manifest()
    u = build_unico(m)
    assert len(u) == 6 and (u["motivo_representante"] == "copia_limpa_mais_nitida").all()
    bad = m.copy()
    bad.loc[bad["image_id"] == "0_0", "is_representative"] = False
    bad.loc[bad["image_id"] == "0_1", "is_representative"] = True  # cópia suja como representante
    with pytest.raises(AssertionError):
        check_invariants(bad, bad[bad["is_representative"]])
