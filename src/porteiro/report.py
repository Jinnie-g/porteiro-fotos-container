"""Relatório automático (Markdown) + pranchas de revisão visual de uma execução."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import viz


def _md(df: pd.DataFrame, floatfmt: str = "{:.3g}") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def _crosstab(df, a, b):
    t = pd.crosstab(df[a], df[b], margins=True, margins_name="total")
    return t.reset_index().rename(columns={a: f"{a} \\ {b}"})


QCOLS = ["aug_geometrica", "cutout_n", "fill_frac", "blur_score", "noise_sigma", "bw_efetiva", "jpeg_quality_est", "lum_mean", "faixa_dinamica",
         "clip_frac", "color_cast", "border_frac", "caixa_area_max", "st_coerencia", "corr_pico", "paralelismo"]


def write(df, audit, pairs, amostra, dstats, astats, info, cfg, data: Path, out: Path) -> None:
    fig = out / "figuras"
    rng = np.random.default_rng(cfg["seed"])
    P = lambda iid: data / "images" / f"{iid}.jpg"  # noqa: E731
    n = len(df)
    L = [f"# Relatório de tratamento do dataset — {info['data_utc'][:10]}", "",
         f"Projeto Roboflow `{cfg['roboflow']['workspace']}/{cfg['roboflow']['project']}` · "
         f"pipeline v{info['versao_pipeline']} · commit `{(info['git_commit'] or '')[:10]}` · semente {cfg['seed']}",
         "", "Gerado automaticamente por `scripts/run_treatment.py` (etapas E1 a E6 do guia). "
         "Limiares são valores iniciais a calibrar.", ""]

    # E1/E2
    L += ["## E1/E2 Inventário e integridade", "",
          f"- Imagens no manifesto: **{n}**",
          "- Dimensões mais comuns: " + ", ".join(
              f"{int(w)}×{int(h)} ({c})" for (w, h), c in df.groupby(['largura', 'altura']).size()
              .sort_values(ascending=False).head(5).items()),
          f"- SHA-256 distintos: {df['sha256'].nunique()} · nomes de origem distintos: {df['nome_origem'].nunique()}"
          f" · contêineres (ISO 6346) identificados: {df['container_code'].nunique()} "
          f"({df['container_code'].notna().mean():.0%} das imagens)", "",
          _md(df["integridade"].value_counts().rename_axis("integridade").reset_index(name="n")), "",
          "### Fontes inferidas pelo nome × split original", "", _md(_crosstab(df, "source_id", "split_orig")), "",
          "### Anotações (caixas de avaria)", "",
          _md(df.groupby("split_orig")[["n_caixas", "n_rust", "n_dent", "n_hole"]].sum().reset_index()), "",
          f"- Imagens sem nenhuma caixa: {(df['n_caixas'] == 0).sum()}", ""]

    # E3
    d = dstats
    L += ["## E3 Duplicatas e grupos", "",
          f"Regra: SHA-256 idêntico, **ou** mesma foto de origem (nome sem sufixos do Roboflow **e** cosseno bruto ≥ "
          f"{cfg['dedup'].get('same_name_cosine_min', 0.8)}), **ou** "
          f"pHash ≤ {cfg['dedup']['phash_max_hamming']} (invariante a flip) **e** cosseno ≥ {cfg['dedup']['cosine_min']} "
          "(embeddings do Roboflow, centralizados). `group_id` une ainda todo o nome de origem e o mesmo código "
          "de contêiner (conservador para o split).", "",
          f"- Cópias exatas extras (SHA-256): **{d['exatas_extras']}**",
          f"- Pares pHash: {d['pares_phash']} · pares cosseno: {d['pares_cosseno']} · ambos: {d['pares_ambos']}",
          f"- Nomes de origem distintos: **{d['n_nomes_origem']}**",
          f"- Clusters de duplicatas: **{d['n_dup_clusters']}** para {n} imagens → "
          f"**{d['frac_redundante']:.1%} redundantes**; maior cluster: {d['maior_dup_cluster']}",
          f"- Grupos (cluster + contêiner): {d['n_grupos']}; maior grupo: {d['maior_grupo']}", "",
          "Redundância por fonte:", "",
          _md(df.groupby("source_id").agg(imagens=("image_id", "size"), clusters=("dup_cluster_id", "nunique"))
              .assign(frac_redundante=lambda t: 1 - t.clusters / t.imagens).reset_index()), ""]
    big = df["dup_cluster_id"].value_counts()
    items = []
    for cid in big[big > 1].index[:12]:
        ids = df.loc[df["dup_cluster_id"] == cid, "image_id"].tolist()[:4]
        items.append(([P(i) for i in ids], f"cluster {cid}: {big[cid]} imgs"))
    viz.grid(items, fig / "e3_maiores_clusters.jpg", size=150, cols=2)
    L += ["![maiores clusters](figuras/e3_maiores_clusters.jpg)", ""]

    # E5
    L += ["## E5 Auditoria do split original do Roboflow", ""]
    rows = [{"chave": k.replace("_em_mais_de_um_split", ""), **v} for k, v in astats.items() if "mais_de_um" in k]
    L += ["Chaves que aparecem em mais de um split (vazamento por grupo):", "", _md(pd.DataFrame(rows)), ""]
    rows = [{"split": k.split("_")[0], **v} for k, v in astats.items() if "vizinho" in k]
    L += ["Vizinho mais próximo no treino acima do limiar de quase duplicata:", "", _md(pd.DataFrame(rows)), ""]
    leak = audit[audit["acima_limiar"] | audit["mesmo_nome_origem"]].sort_values("cos", ascending=False)
    pick = leak.sample(min(16, len(leak)), random_state=cfg["seed"]) if len(leak) else leak
    viz.grid([([P(r.image_id), P(r.nn_train)], f"{r.split} x train cos={r.cos:.3f} h={r.hamming}")
              for r in pick.itertuples()], fig / "e5_vazamento_exemplos.jpg", size=150, cols=4)
    L += ["![exemplos de vazamento](figuras/e5_vazamento_exemplos.jpg)", "",
          "### Novo split agrupado (StratifiedGroupKFold)", "",
          f"{cfg['split']['n_splits']} folds; fold 0 = teste fixo. Nenhum `group_id`, `dup_cluster_id` ou SHA-256 "
          "atravessa folds (verificado por `split.check_no_leak`).", "",
          _md(_crosstab(df, "split_orig", "split_novo")), ""]

    # Artefatos de augmentation (E4, achado específico deste dataset)
    df["_aug_any"] = df["aug_geometrica"] | (df["cutout_n"] > 0)
    by_name = df.groupby("nome_origem").agg(n=("image_id", "size"), limpas=("_aug_any", lambda v: int((~v).sum())))
    L += ["## Artefatos de augmentation já embutidos no dataset", "",
          "O dataset foi exportado de outra versão do Roboflow com augmentation e reimportado: os nomes trazem "
          "`.rf.<hash>` e cada foto de origem aparece várias vezes. Para a planificação isso é crítico, porque "
          "rotação, cisalhamento e perspectiva **artificiais** mudam justamente a geometria que se quer estimar, e "
          "o preenchimento (preto ou cinza constante) não é uma borda retangular.", "",
          f"- `aug_geometrica` (preenchimento constante não retangular ligado a canto/borda): "
          f"**{int(df['aug_geometrica'].sum())}** imagens ({df['aug_geometrica'].mean():.1%})",
          f"- `cutout` (retângulos de cor constante colados): **{int((df['cutout_n'] > 0).sum())}** imagens "
          f"({(df['cutout_n'] > 0).mean():.1%})",
          f"- Imagens sem nenhum dos dois artefatos: **{int((~df['_aug_any']).sum())}** "
          f"({(~df['_aug_any']).mean():.1%})",
          f"- Fotos de origem (nomes) com pelo menos uma cópia limpa: {int((by_name['limpas'] > 0).sum())} de "
          f"{len(by_name)}", "",
          _md(df.groupby("source_id").agg(n=("image_id", "size"), aug_geometrica=("aug_geometrica", "mean"),
                                          cutout=("cutout_n", lambda v: (v > 0).mean()),
                                          fill_frac_med=("fill_frac", "median")).reset_index()), "",
          _md(df.groupby("split_orig").agg(n=("image_id", "size"), aug_geometrica=("aug_geometrica", "mean"),
                                           cutout=("cutout_n", lambda v: (v > 0).mean())).reset_index()), ""]
    for name, sel in (("aug_geometrica", df[df["aug_geometrica"]]), ("cutout", df[df["cutout_n"] > 0]),
                      ("limpas", df[~df["_aug_any"]])):
        s_ = sel.sample(min(18, len(sel)), random_state=cfg["seed"])
        viz.grid([([P(r.image_id)], f"{r.source_id} fill={r.fill_frac:.0%}") for r in s_.itertuples()],
                 fig / f"aug_{name}.jpg", size=150, cols=6)
        L += [f"**{name}**", "", f"![{name}](figuras/aug_{name}.jpg)", ""]
    df.drop(columns=["_aug_any"], inplace=True)

    # E4
    L += ["## E4 Caracterização", "", "Medianas por split original (só região de conteúdo):", "",
          _md(df.groupby("split_orig")[QCOLS].median().reset_index()), "",
          "Medianas por fonte:", "", _md(df.groupby("source_id")[QCOLS].median().reset_index()), "",
          "### Bordas pretas e colagens", "",
          f"- Com borda preta detectada: **{(df['border_frac'] > 0).sum()}** "
          f"({(df['border_frac'] > 0).mean():.1%}); `safe_to_crop`: {df['safe_to_crop'].sum()}",
          f"- Borda escura gradual (conteúdo real, não recortar): {df['borda_gradual'].sum()}",
          f"- Possíveis colagens (faixa preta no meio): {df['colagem'].sum()}",
          f"- Flags: " + ", ".join(f"{k or 'nenhuma'}={v}" for k, v in df['flags'].fillna('').value_counts().items()),
          "", "Bordas por lado (imagens com borda > 0):", "",
          _md(pd.DataFrame({s: [(df[f'border_{s}'] > 0).sum(), df.loc[df[f'border_{s}'] > 0, f'border_{s}'].median()]
                            for s in ('top', 'bottom', 'left', 'right')}, index=["n", "mediana_frac"])
              .T.reset_index().rename(columns={"index": "lado"})), ""]
    sel = df[df["border_frac"] > 0].sort_values("border_frac", ascending=False)
    sel = pd.concat([sel.head(6), sel.sample(min(6, len(sel)), random_state=1)]).drop_duplicates("image_id")
    viz.grid([([P(r.image_id)], f"borda {r.border_frac:.0%} crop={r.safe_to_crop}") for r in sel.itertuples()],
             fig / "e4_bordas.jpg", size=170, cols=6)
    col = df[df["colagem"]].head(12)
    viz.grid([([P(r.image_id)], f"colagem {r.source_id}") for r in col.itertuples()],
             fig / "e4_colagens.jpg", size=170, cols=6)
    for name, s in (("mais_desfocadas", df.nsmallest(12, "blur_score")), ("mais_escuras", df.nsmallest(12, "lum_mean")),
                    ("menor_banda_efetiva", df.nsmallest(12, "bw_efetiva"))):
        viz.grid([([P(r.image_id)], f"{r.source_id}") for r in s.itertuples()], fig / f"e4_{name}.jpg",
                 size=150, cols=6)
    L += ["![bordas](figuras/e4_bordas.jpg)", "", "![colagens](figuras/e4_colagens.jpg)", "",
          "Extremos: ![](figuras/e4_mais_desfocadas.jpg) ![](figuras/e4_mais_escuras.jpg) "
          "![](figuras/e4_menor_banda_efetiva.jpg)", ""]

    # E6
    L += ["## E6 Amostra para rotulagem", "",
          f"`amostra_rotulagem.csv`: **{len(amostra)}** representantes de cluster, fora do teste fixo; "
          f"{(amostra['origem_amostra'] == 'aleatoria').sum()} aleatórias e "
          f"{(amostra['origem_amostra'] == 'estratificada').sum()} estratificadas (fonte × nitidez × área da caixa). "
          f"As primeiras {amostra['calibracao_kappa'].sum()} linhas marcadas em `calibracao_kappa` são a rodada "
          "de calibração com 2 anotadores (seção 2.6).", "",
          _md(amostra["source_id"].value_counts().rename_axis("source_id").reset_index(name="n")), "",
          "`pares_revisao_dedup.csv`: pares por faixa de cosseno para julgar à mão e calibrar o limiar (E3.3).", ""]
    (out / "RELATORIO.md").write_text("\n".join(L), encoding="utf-8")
