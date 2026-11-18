"""Seção 16: divisão do dataset em sem zoom e com zoom (S0 a S10).

Uso:
  python scripts/extract_dino.py      # E7: embeddings DINOv2 (uma vez)
  python scripts/run_zoom.py          # S0-S10 -> results/zoom_<data>/

Rótulos: labels/zoom_rotulos.csv (rodada `principal` = rótulo usado; `reteste` e
`variacao_aug` medem confiabilidade). Trocar por rótulos humanos não muda o código.
"""
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from porteiro import env, viz  # noqa: E402
from porteiro import stats as S  # noqa: E402
from porteiro import zoom_model as Z  # noqa: E402
from porteiro.consolidate import build_unico  # noqa: E402
from porteiro.zoom_explore import box_proxies, classe_principal, dip_tests, embedding_clusters, medoids  # noqa: E402

SEED = 42
env.seed_all(SEED)
run = sorted((ROOT / "results").glob("run_*"))[-1]
OUT = ROOT / "results" / f"zoom_{date.today().isoformat()}"
FIG = OUT / "figuras"
FIG.mkdir(parents=True, exist_ok=True)
IMG = ROOT / "data" / "roboflow" / "images"
P = lambda i: IMG / f"{i}.jpg"  # noqa: E731
R = {}  # tudo que vai para o relatório


def save(df, name):
    df.to_csv(OUT / name, index=False)
    return df


# ---------------- S0: pré-requisitos ----------------
full = pd.read_parquet(run / "manifesto.parquet")
unico = build_unico(full)  # verifica os invariantes D8
u = unico.merge(box_proxies(unico, ROOT / "data" / "roboflow" / "meta"), on=["image_id", "caixa_area_max",
                                                                              "caixa_area_soma", "n_caixas"])
u["classe_principal"] = classe_principal(u)
R["S0"] = {"manifesto": len(full), "unico": len(u), "teste_fixo": int((u["split_novo"] == "teste_fixo").sum()),
           "motivos": u["motivo_representante"].value_counts().to_dict()}
print("S0", R["S0"])

# ---------------- rótulos ----------------
labels = pd.read_csv(ROOT / "labels" / "zoom_rotulos.csv")
amostra = pd.read_csv(ROOT / "labels" / "amostra_zoom.csv")
main = labels[labels["rodada"] == "principal"][["image_id", "zoom_status", "anotador"]]
main = main.merge(amostra[amostra["rodada"] == "principal"][["image_id", "origem_amostra"]], on="image_id")
u = u.merge(main, on="image_id", how="left")
u["rotulado"] = u["zoom_status"].notna()

d224 = Z.load_dino(ROOT / "data" / "features" / "dino_vits14_224.npz")
d448 = Z.load_dino(ROOT / "data" / "features" / "dino_vits14_448.npz")

# ---------------- S1: exploração sem rótulos ----------------
dips = pd.concat([dip_tests(u["log_area_max"], u["source_id"]).assign(por="fonte"),
                  dip_tests(u["log_area_max"], u["classe_principal"]).assign(por="classe")])
dips["q_bh"] = S.benjamini_hochberg(dips["p"].values)
save(dips, "s1_dip_tests.csv")
pos448 = {k: i for i, k in enumerate(d448["ids"])}
E = d448["mean"][[pos448[i] for i in u["image_id"]]]
sil, labs_k = embedding_clusters(E)
best_k = int(sil.sort_values("silhueta", ascending=False).iloc[0]["k"])
u["cluster_emb"] = labs_k[best_k]
save(sil, "s1_silhueta_kmeans.csv")
med = medoids(E, labs_k[best_k], per=6)
viz.grid([([P(u["image_id"].iat[i]) for i in idx[:4]], f"cluster {c} (n={int((u['cluster_emb'] == c).sum())})")
          for c, idx in med.items()], FIG / "s1_medoides_clusters.jpg", size=140, cols=2)
u["t_proxy"] = pd.qcut(u["caixa_area_max"].rank(method="first"), 3, labels=["pequena", "media", "grande"]).astype(str)
save(pd.crosstab([u["source_id"]], u["t_proxy"]).reset_index(), "s1_proxy_x_fonte.csv")
save(pd.crosstab(u["classe_principal"], u["t_proxy"]).reset_index(), "s1_proxy_x_classe.csv")
R["S1"] = {"best_k": best_k, "silhueta": sil.to_dict("records")}

# ---------------- S4: confiabilidade ----------------
lab = labels.pivot_table(index="image_id", columns="rodada", values="zoom_status", aggfunc="first")
rt = lab.dropna(subset=["principal", "reteste"])
rt = rt[(rt["principal"] != "IND") & (rt["reteste"] != "IND")]
order = ["Z0", "Z1", "Z2"]
R["S4"] = {
    "n_reteste": len(rt),
    "kappa_ponderado_linear": S.cohen_kappa(rt["principal"], rt["reteste"], order, "linear"),
    "kappa_binario_Z0": S.cohen_kappa(rt["principal"] == "Z0", rt["reteste"] == "Z0", [False, True]),
    "concordancia": float((rt["principal"] == rt["reteste"]).mean()),
}
save(pd.crosstab(rt["principal"], rt["reteste"]).reset_index(), "s4_confusao_reteste.csv")
va = labels[labels["rodada"] == "variacao_aug"].merge(full[["image_id", "dup_cluster_id", "aug_geometrica",
                                                            "cutout_n", "is_representative"]], on="image_id")
per_c = va.groupby("dup_cluster_id")["zoom_status"].agg(lambda s: s.nunique())
va_bin = va.assign(z0=va["zoom_status"] == "Z0").groupby("dup_cluster_id")["z0"].nunique()
R["S4"]["aug_clusters"] = int(per_c.size)
R["S4"]["aug_clusters_com_rotulo_diferente"] = int((per_c > 1).sum())
R["S4"]["aug_clusters_Z0_diferente"] = int((va_bin > 1).sum())
print("S4", R["S4"])

# ---------------- S5: análise dos rótulos ----------------
L = u[u["rotulado"] & (u["zoom_status"] != "IND")].copy()
L["Z0"] = (L["zoom_status"] == "Z0").astype(int)
rnd = L[L["origem_amostra"] == "aleatoria"]
R["S5"] = {"n_rotuladas": int(u["rotulado"].sum()), "n_IND": int((u["zoom_status"] == "IND").sum()),
           "prev_Z0_aleatoria": S.wilson(int(rnd["Z0"].sum()), len(rnd)),
           "prev_Z0_todas": S.wilson(int(L["Z0"].sum()), len(L)),
           "dist": L["zoom_status"].value_counts().to_dict()}
prev_rows = []
for col in ("source_id", "classe_principal", "split_orig", "artefato_aug", "t_proxy", "cluster_emb"):
    t = L.groupby(col)["zoom_status"].value_counts().unstack(fill_value=0)
    for g, r in t.iterrows():
        n = int(r.sum())
        p, lo, hi = S.wilson(int(r.get("Z0", 0)), n)
        prev_rows.append({"variavel": col, "grupo": str(g), "n": n, **{k: int(r.get(k, 0)) for k in order},
                          "prev_Z0": p, "ic_lo": lo, "ic_hi": hi})
save(pd.DataFrame(prev_rows), "s5_prevalencia.csv")
assoc = []
for col in ("source_id", "classe_principal", "artefato_aug", "split_orig", "t_proxy", "cluster_emb"):
    v, p = S.cramers_v(L[col].astype(str), L["zoom_status"])
    assoc.append({"variavel": col, "cramers_v": v, "p": p, "info_mutua_bits": S.mutual_info(L[col].astype(str),
                                                                                             L["zoom_status"])})
assoc = pd.DataFrame(assoc)
assoc["q_bh"] = S.benjamini_hochberg(assoc["p"].values)
save(assoc, "s5_associacoes.csv")
R["S5"]["assoc"] = assoc.to_dict("records")
print("S5", R["S5"]["prev_Z0_aleatoria"], R["S5"]["dist"])

# ---------------- S6/S7: modelagem e avaliação ----------------
dev = L[L["origem_amostra"] != "teste_fixo"].reset_index(drop=True)
tst = L[L["origem_amostra"] == "teste_fixo"].reset_index(drop=True)
y, g = dev["Z0"].values, dev["group_id"].values
F = Z.feature_sets(dev, d224, d448)
exps, oofs = [], {}
oofs["Z2_majoritaria"] = Z.repeated_oof(F["Z2a_so_fonte"], y, g, lambda r: Z.Majority())
for name, X in F.items():
    oofs[name] = Z.repeated_oof(X, y, g, lambda r: Z.PlattModel("lr", seed=SEED + r))
    print("  ", name, round(np.mean([average_precision_score(y, o) for o in oofs[name]]), 3), flush=True)
for name, o in oofs.items():
    aps = [average_precision_score(y, oo) for oo in o]
    m = Z.metrics(y, o.mean(0))
    exps.append({"exp": name, "modelo": "maj" if "majoritaria" in name else "LR+Platt",
                 "pr_auc_media": float(np.mean(aps)), "pr_auc_dp_repeticoes": float(np.std(aps)),
                 **{k: m[k] for k in ("roc_auc", "bal_acc", "ece", "brier")}})
exps = pd.DataFrame(exps)
cand = exps[~exps["exp"].str.startswith("Z2")]
best = cand.sort_values("pr_auc_media", ascending=False).iloc[0]["exp"]
# Z6 (antigo E6): SVM linear no melhor conjunto
o_svm = Z.repeated_oof(F[best], y, g, lambda r: Z.PlattModel("svm", seed=SEED + r))
oofs[f"{best}_svm"] = o_svm
m = Z.metrics(y, o_svm.mean(0))
exps.loc[len(exps)] = {"exp": f"{best}_svm", "modelo": "SVM+Platt",
                       "pr_auc_media": float(np.mean([average_precision_score(y, oo) for oo in o_svm])),
                       "pr_auc_dp_repeticoes": float(np.std([average_precision_score(y, oo) for oo in o_svm])),
                       **{k: m[k] for k in ("roc_auc", "bal_acc", "ece", "brier")}}
# IC por grupo (bootstrap) da PR-AUC e diferenças pareadas contra os baselines
ci = []
for name, o in oofs.items():
    p0 = o.mean(0)
    pt, lo, hi = S.group_bootstrap(g, lambda idx: average_precision_score(y[idx], p0[idx]) if y[idx].any()
                                   else float("nan"), n_boot=1000, seed=SEED)
    ci.append({"exp": name, "pr_auc_ic_lo": lo, "pr_auc_ic_hi": hi})
exps = exps.merge(pd.DataFrame(ci), on="exp")
pb = oofs[best].mean(0)
for base in ("Z2_majoritaria", "Z2a_so_fonte", "Z2b_trivial_metadados_fonte"):
    pbase = oofs[base].mean(0)
    d, lo, hi = S.group_bootstrap(g, lambda idx: average_precision_score(y[idx], pb[idx])
                                  - average_precision_score(y[idx], pbase[idx]) if y[idx].any() else float("nan"),
                                  n_boot=1000, seed=SEED)
    R.setdefault("dif_pareada", []).append({"contra": base, "dif_pr_auc": d, "ic_lo": lo, "ic_hi": hi,
                                            "exclui_zero": bool(lo > 0)})
save(exps.sort_values("pr_auc_media", ascending=False), "s6_matriz_experimentos.csv")
R["melhor"] = best
print("melhor:", best)

# limiares e abstenção (fora-da-amostra, dev)
p_dev = oofs[best].mean(0)
T = Z.choose_thresholds(y, p_dev, target=0.95)
rt_dev = Z.route(p_dev, T)
R["limiares"] = T
cr = Z.coverage_risk(y, p_dev)
save(cr, "s7_cobertura_risco.csv")
accA = rt_dev == "sem_zoom"
R["dev_roteamento"] = {
    "A_n": int(accA.sum()), "A_precisao": S.wilson(int(y[accA].sum()), int(accA.sum())),
    "cobertura_Z0": float(y[accA].sum() / y.sum()), "abstencao": float((rt_dev == "revisao").mean()),
    "B_precisao": S.wilson(int((1 - y[rt_dev == "com_zoom"]).sum()), int((rt_dev == "com_zoom").sum())),
}
# cabeça ordinal secundária (Z0/Z1/Z2): multinomial, OOF agrupado
yo = dev["zoom_status"].map({"Z0": 0, "Z1": 1, "Z2": 2}).values
po = np.zeros(len(yo), int)
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(F[best], yo, g):
    mo = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000))
    po[te] = mo.fit(F[best][tr], yo[tr]).predict(F[best][te])
R["ordinal"] = {"kappa_ponderado": S.cohen_kappa(yo, po, [0, 1, 2], "linear"),
                "acuracia": float((yo == po).mean())}
save(pd.crosstab(pd.Series(yo, name="rotulo").map({0: "Z0", 1: "Z1", 2: "Z2"}),
                 pd.Series(po, name="predito").map({0: "Z0", 1: "Z1", 2: "Z2"})).reset_index(),
     "s7_confusao_ordinal.csv")

# fatias por fonte / artefato / nitidez
sl = []
dev["p"] = p_dev
dev["rota"] = rt_dev
dev["t_nitidez"] = pd.qcut(dev["blur_pct_fonte"].rank(method="first"), 3, labels=["baixa", "media", "alta"]).astype(str)
for col in ("source_id", "artefato_aug", "t_nitidez", "classe_principal"):
    for gname, s in dev.groupby(col):
        a = s["rota"] == "sem_zoom"
        sl.append({"fatia": col, "grupo": str(gname), "n": len(s), "n_Z0": int(s["Z0"].sum()),
                   "pr_auc": average_precision_score(s["Z0"], s["p"]) if 0 < s["Z0"].sum() < len(s) else np.nan,
                   "A_n": int(a.sum()), "A_precisao": s.loc[a, "Z0"].mean() if a.any() else np.nan})
save(pd.DataFrame(sl), "s7_fatias.csv")

# leave-one-source-out
loso = []
Xb = F[best]
for src in dev["source_id"].unique():
    te = (dev["source_id"] == src).values
    if te.sum() < 15 or y[te].sum() == 0 or y[~te].sum() < 5:
        continue
    mdl = Z.PlattModel("lr", seed=SEED).fit(Xb[~te], y[~te], g[~te])
    pp = mdl.predict_proba(Xb[te])
    a = pp >= T["tau_alto"]
    loso.append({"fonte": src, "n": int(te.sum()), "n_Z0": int(y[te].sum()),
                 "pr_auc": average_precision_score(y[te], pp), "pr_auc_cv": average_precision_score(y[te], p_dev[te]),
                 "A_n": int(a.sum()), "A_precisao": float(y[te][a].mean()) if a.any() else np.nan})
save(pd.DataFrame(loso), "s7_leave_one_source_out.csv")

# curva de aprendizado
lc = []
rng = np.random.default_rng(SEED)
groups_u = np.unique(g)
for n_target in (100, 200, 300, len(y)):
    vals = []
    for rep in range(3):
        if n_target >= len(y):
            idx = np.arange(len(y))
        else:
            gg = rng.permutation(groups_u)
            idx, k = [], 0
            while len(idx) < n_target:
                idx.extend(np.nonzero(g == gg[k])[0])
                k += 1
            idx = np.array(idx)
        if y[idx].sum() < 8:
            continue
        o = Z.repeated_oof(Xb[idx], y[idx], g[idx], lambda r: Z.PlattModel("lr", seed=SEED + r + rep),
                           n_repeats=1, seed=SEED + rep)
        vals.append(average_precision_score(y[idx], o[0]))
    lc.append({"n_rotuladas": int(n_target), "pr_auc_media": float(np.mean(vals)), "pr_auc_dp": float(np.std(vals)),
               "n_reps": len(vals)})
save(pd.DataFrame(lc), "s7_curva_aprendizado.csv")

# erros fora-da-amostra (todos os FP e FN do roteamento em dev)
fp = dev[(dev["rota"] == "sem_zoom") & (dev["Z0"] == 0)]
fn = dev[(dev["rota"] == "com_zoom") & (dev["Z0"] == 1)]
ab = dev[dev["rota"] == "revisao"]
viz.grid([([P(r.image_id)], f"FP {r.zoom_status} p={r.p:.2f}") for r in fp.itertuples()], FIG / "s7_falsos_positivos_A.jpg",
         size=150, cols=6)
viz.grid([([P(r.image_id)], f"FN p={r.p:.2f}") for r in fn.itertuples()], FIG / "s7_falsos_negativos_A.jpg",
         size=150, cols=6)
save(pd.concat([fp.assign(erro="FP"), fn.assign(erro="FN")])[["image_id", "erro", "zoom_status", "p",
                                                                 "source_id", "artefato_aug"]], "s7_erros.csv")
R["erros"] = {"FP": len(fp), "FN": len(fn), "abstencoes": len(ab)}

# confirmação no teste fixo (uma vez)
Ftr_all = Z.feature_sets(dev, d224, d448)[best]
Fte = Z.feature_sets(tst, d224, d448)[best]
final_dev = Z.PlattModel("lr", seed=SEED).fit(Ftr_all, y, g)
pt = final_dev.predict_proba(Fte)
yt = tst["Z0"].values
rtt = Z.route(pt, T)
a = rtt == "sem_zoom"
mt = Z.metrics(yt, pt)
pt_ap, lo, hi = S.group_bootstrap(tst["group_id"].values, lambda idx: average_precision_score(yt[idx], pt[idx])
                                  if yt[idx].any() else float("nan"), n_boot=1000, seed=SEED)
R["teste_fixo"] = {"n": len(yt), "n_Z0": int(yt.sum()), **mt, "pr_auc_ic": [lo, hi],
                   "A_n": int(a.sum()), "A_precisao": S.wilson(int(yt[a].sum()), int(a.sum())),
                   "cobertura_Z0": float(yt[a].sum() / max(1, yt.sum())), "abstencao": float((rtt == "revisao").mean())}
print("teste", R["teste_fixo"])

# robustez a corrupções controladas (Z-9), no teste fixo
from porteiro.features_dino import extract, load_model  # noqa: E402
from porteiro import geometry  # noqa: E402

model_dino = load_model(ROOT / "data" / "models" / "dinov2_vits14_pretrain.pth")
corr_dir = ROOT / "data" / "corrupted"
corr_dir.mkdir(parents=True, exist_ok=True)


def corrupt(img, kind):
    if kind == "desfoque":
        return cv2.GaussianBlur(img, (0, 0), 2.0)
    if kind == "ruido":
        return np.clip(img + np.random.default_rng(0).normal(0, 15, img.shape), 0, 255).astype(np.uint8)
    if kind == "jpeg_q20":
        return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 20])[1], cv2.IMREAD_COLOR)
    if kind == "escuro_x0.5":
        return (img * 0.5).astype(np.uint8)
    return img


rob = []
from PIL import Image  # noqa: E402

for kind in ("original", "desfoque", "ruido", "jpeg_q20", "escuro_x0.5"):
    paths = []
    for iid in tst["image_id"]:
        q = corr_dir / kind / f"{iid}.jpg"
        q.parent.mkdir(parents=True, exist_ok=True)
        if not q.exists():
            cv2.imwrite(str(q), corrupt(cv2.imread(str(P(iid))), kind), [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths.append(q)
    f448 = extract(model_dino, paths, 448)
    f224 = extract(model_dino, paths, 224) if "224" in best else None
    feats = {"Z3a_dino_cls_224": f224["cls"] if f224 else None, "Z3b_dino_cls_448": f448["cls"],
             "Z4a_dino_media_448": f448["mean"],
             "Z4b_dino_concat_448": np.hstack([f448["cls"], f448["mean"]])}
    if best == "Z6_dino_concat_opencv" or best == "Z5_opencv":
        ocv = pd.DataFrame([geometry.describe(q, None) for q in paths])[Z.OPENCV_COLS].fillna(0).to_numpy(float)
        feats["Z5_opencv"] = ocv
        feats["Z6_dino_concat_opencv"] = np.hstack([f448["cls"], f448["mean"], ocv])
    Xc = feats[best].astype(np.float32)
    pc = final_dev.predict_proba(Xc)
    ac = Z.route(pc, T) == "sem_zoom"
    rob.append({"corrupcao": kind, "pr_auc": average_precision_score(yt, pc), "A_n": int(ac.sum()),
                "A_precisao": float(yt[ac].mean()) if ac.any() else np.nan})
rob = pd.DataFrame(rob)
rob["queda_pr_auc"] = rob["pr_auc"].iloc[0] - rob["pr_auc"]
save(rob, "s7_robustez.csv")

# critérios de aceite
fsrc = pd.DataFrame(sl)
fsrc = fsrc[(fsrc["fatia"] == "source_id") & (fsrc["n"] >= 30) & (fsrc["A_n"] > 0)]
dp = {d["contra"]: d for d in R["dif_pareada"]}
crit = [
    ("Precisão de A (limite inferior do IC, teste fixo) >= 0,90", R["teste_fixo"]["A_precisao"][1], 0.90),
    ("κ ponderado modelo × rótulo (ordinal, dev) >= 0,6", R["ordinal"]["kappa_ponderado"], 0.6),
    ("ECE (teste fixo) <= 0,05", R["teste_fixo"]["ece"], 0.05),
    ("Precisão de A >= 0,85 em cada fonte com n >= 30", fsrc["A_precisao"].min() if len(fsrc) else np.nan, 0.85),
    ("Ganho sobre o trivial (IC pareado exclui zero)", dp["Z2b_trivial_metadados_fonte"]["ic_lo"], 0.0),
    ("Ganho sobre o só-fonte (IC pareado exclui zero)", dp["Z2a_so_fonte"]["ic_lo"], 0.0),
]
crit = pd.DataFrame([{"criterio": c, "valor": v, "limite": t,
                      "passou": bool(v <= t) if "ECE" in c else bool(v >= t if "IC pareado" not in c else v > t)}
                     for c, v, t in crit])
save(crit, "s7_criterios_aceite.csv")

# ---------------- S8: roteamento e consolidação ----------------
Fall = Z.feature_sets(L.reset_index(drop=True), d224, d448)[best]
final = Z.PlattModel("lr", seed=SEED).fit(Fall, L["Z0"].values, L["group_id"].values)
Fu = Z.feature_sets(u, d224, d448)[best]
u["p_Z0"] = final.predict_proba(Fu)
u["destino_modelo"] = Z.route(u["p_Z0"].values, T)
lab_route = u["zoom_status"].map({"Z0": "sem_zoom", "Z1": "com_zoom", "Z2": "com_zoom", "IND": "revisao"})
u["destino"] = lab_route.fillna(u["destino_modelo"])
u["origem_destino"] = np.where(u["zoom_status"].notna(), "rotulo", "modelo")
u["geometria_suficiente"] = pd.NA  # coluna prevista (seção 15.0); depende de rótulo de geometria ainda inexistente
u["versao_modelo"] = f"zoom-{best}-LR-Platt"
cols = ["image_id", "rf_nome", "nome_origem", "source_id", "group_id", "dup_cluster_id", "split_novo",
        "zoom_status", "p_Z0", "destino_modelo", "destino", "origem_destino", "geometria_suficiente",
        "artefato_aug", "classe_principal", "n_rust", "n_dent", "n_hole", "caixa_area_max", "versao_modelo"]
dsets = {}
for d in ("sem_zoom", "com_zoom", "revisao"):
    dsets[d] = u.loc[u["destino"] == d, cols]
    dsets[d].to_parquet(OUT / f"dataset_{d}.parquet", index=False)
    s = dsets[d].sample(min(24, len(dsets[d])), random_state=SEED)
    viz.grid([([P(r.image_id)], f"{r.origem_destino} p={r.p_Z0:.2f}") for r in s.itertuples()],
             FIG / f"s8_dataset_{d}.jpg", size=150, cols=6)
# propaga às cópias só para rastreio
prop = full[["image_id", "dup_cluster_id", "is_representative"]].merge(
    u[["dup_cluster_id", "destino"]].rename(columns={"destino": "destino_propagado"}), on="dup_cluster_id", how="left")
prop.to_parquet(OUT / "destino_propagado_copias.parquet", index=False)
R["S8"] = {d: {"n": len(v), "por_origem": v["origem_destino"].value_counts().to_dict(),
               "por_split": v["split_novo"].value_counts().to_dict()} for d, v in dsets.items()}
R["S8"]["copias_propagadas"] = prop["destino_propagado"].value_counts().to_dict()
print("S8", {d: len(v) for d, v in dsets.items()})

# ---------------- S9: comparação dos dois datasets ----------------
A, B = dsets["sem_zoom"].merge(u[["image_id"] + Z.QUALITY_COLS], on="image_id"), \
    dsets["com_zoom"].merge(u[["image_id"] + Z.QUALITY_COLS], on="image_id")
AB = pd.concat([A.assign(ds="sem_zoom"), B.assign(ds="com_zoom")], ignore_index=True)
cmp = []
for col in ("source_id", "classe_principal"):
    v, p = S.cramers_v(AB[col], AB["ds"])
    cmp.append({"variavel": col, "tipo": "categorica", "estatistica": "V de Cramér", "valor": v, "p": p})
for c in ("rust", "dent", "hole"):
    v, p = S.cramers_v(AB[f"n_{c}"] > 0, AB["ds"])
    cmp.append({"variavel": f"tem_{c}", "tipo": "categorica", "estatistica": "V de Cramér", "valor": v, "p": p,
                "sem_zoom": float((A[f"n_{c}"] > 0).mean()), "com_zoom": float((B[f"n_{c}"] > 0).mean())})
for col in ["caixa_area_max", "blur_score", "noise_sigma", "lum_mean", "faixa_dinamica", "color_cast", "bw_efetiva"]:
    ks = st.ks_2samp(A[col].dropna(), B[col].dropna())
    cmp.append({"variavel": col, "tipo": "continua", "estatistica": "delta de Cliff (A vs B)",
                "valor": S.cliffs_delta(A[col], B[col]), "p": float(ks.pvalue),
                "sem_zoom": float(A[col].median()), "com_zoom": float(B[col].median())})
cmp = pd.DataFrame(cmp)
cmp["q_bh"] = S.benjamini_hochberg(cmp["p"].values)
save(cmp, "s9_comparacao_A_B.csv")
save(pd.crosstab(AB["source_id"], AB["ds"]).reset_index(), "s9_fontes_A_B.csv")

# ---------------- S10: registro ----------------
info = env.run_info(ROOT)
(OUT / "run_info.json").write_text(json.dumps({**info, "semente": SEED, "manifesto": str(run.name),
                                               "resultados": R}, indent=1, default=str), encoding="utf-8")
(OUT / "pip_freeze.txt").write_text(env.pip_freeze(), encoding="utf-8")
u[["image_id", "cluster_emb", "zoom_status", "p_Z0", "destino_modelo", "destino", "origem_destino"]].to_parquet(
    OUT / "unico_com_destino.parquet", index=False)
print("ok ->", OUT)
