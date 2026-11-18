"""Gera results/zoom_<data>/RELATORIO.md a partir das saídas de run_zoom.py."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((ROOT / "results").glob("zoom_*"))[-1]
info = json.loads((D / "run_info.json").read_text())
R = info["resultados"]


def md(df, fmt="{:.3g}"):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns)
    return head + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.values)


def ci(t):
    return f"{t[0]:.2f} (IC 95% {t[1]:.2f}–{t[2]:.2f})"


rd = lambda n: pd.read_csv(D / n)  # noqa: E731
s5 = rd("s5_prevalencia.csv")
L = [f"# Relatório — divisão em sem zoom e com zoom ({info['data_utc'][:10]})", "",
     f"Gerado por `scripts/run_zoom.py` + `scripts/report_zoom.py` · commit `{(info['git_commit'] or '')[:10]}` · "
     f"semente {info['semente']} · manifesto `{info['manifesto']}`. Seção 16 do guia (S0 a S10).", "",
     "> **Rótulos provisórios.** Os rótulos de zoom desta execução foram feitos por anotador automático "
     "(Claude, inspeção visual) seguindo `docs/guia_anotacao_zoom.md`. Servem para validar o pipeline; a "
     "anotação humana (2 anotadores, κ, adjudicação) substitui `labels/zoom_rotulos.csv` sem mudar o código.", "",
     "## S0 Pré-requisitos", "",
     f"- Manifesto: {R['S0']['manifesto']} imagens → dataset único: **{R['S0']['unico']} representantes** "
     f"(teste fixo: {R['S0']['teste_fixo']}). Invariantes D8 verificados.",
     "- Motivo da escolha do representante: " + ", ".join(f"{k} = {v}" for k, v in R["S0"]["motivos"].items()), "",
     "## S1 Exploração sem rótulos", "",
     "Proxies das caixas de avaria (só para explorar e estratificar; não entram no modelo).", "",
     "Teste de dip de Hartigan sobre log10 da área da maior caixa (p pequeno = mais de um pico):", "",
     md(rd("s1_dip_tests.csv")), "",
     f"k-means no embedding DINOv2 (média dos patches, 448): melhor k = {R['S1']['best_k']} pela silhueta.", "",
     md(rd("s1_silhueta_kmeans.csv")), "", "![medoides](figuras/s1_medoides_clusters.jpg)", "",
     "## S4 Confiabilidade dos rótulos", "",
     f"- Reteste às cegas de {R['S4']['n_reteste']} imagens (mesmo anotador, outra ordem): concordância "
     f"{R['S4']['concordancia']:.0%}, κ ponderado {R['S4']['kappa_ponderado_linear']:.2f}, κ binário (Z0 × resto) "
     f"{R['S4']['kappa_binario_Z0']:.2f}. **É consistência intra-anotador, não κ entre anotadores.**",
     f"- Variação por augmentation: em {R['S4']['aug_clusters_com_rotulo_diferente']} de {R['S4']['aug_clusters']} "
     f"clusters as cópias receberam zoom diferente; em {R['S4']['aug_clusters_Z0_diferente']} a decisão Z0 mudou.", "",
     md(rd("s4_confusao_reteste.csv")), "",
     "## S5 Análise dos rótulos", "",
     f"- Rotuladas: {R['S5']['n_rotuladas']} (IND = {R['S5']['n_IND']}); distribuição: {R['S5']['dist']}",
     f"- Prevalência de Z0 na amostra aleatória: **{ci(R['S5']['prev_Z0_aleatoria'])}**; em todas as rotuladas: "
     f"{ci(R['S5']['prev_Z0_todas'])}", "",
     md(s5[s5["variavel"].isin(["source_id", "classe_principal", "artefato_aug"])]), "",
     "Associação com o zoom (V de Cramér, informação mútua, q de Benjamini-Hochberg):", "",
     md(rd("s5_associacoes.csv")), "",
     "## S6 Modelagem — matriz de experimentos", "",
     "LR (L2, `class_weight` balanceado) com C e Platt ajustados em CV interno agrupado; CV externa agrupada "
     "5 folds × 5 repetições em `dev`. PR-AUC com IC por bootstrap de grupos.", "",
     md(rd("s6_matriz_experimentos.csv")), "",
     f"Modelo escolhido: **{R['melhor']}**. Diferença pareada de PR-AUC (bootstrap por grupo):", "",
     md(pd.DataFrame(R["dif_pareada"])), "",
     "## S7 Avaliação", "",
     f"Limiares (fora-da-amostra, precisão-alvo 0,95): τ_alto = {R['limiares']['tau_alto']:.3f}, "
     f"τ_baixo = {R['limiares']['tau_baixo']:.3f}. Entre os dois → revisão.", "",
     f"- Dev (OOF): A = {R['dev_roteamento']['A_n']} imagens, precisão {ci(R['dev_roteamento']['A_precisao'])}, "
     f"cobertura de Z0 {R['dev_roteamento']['cobertura_Z0']:.0%}, abstenção {R['dev_roteamento']['abstencao']:.0%}; "
     f"precisão de B {ci(R['dev_roteamento']['B_precisao'])}",
     f"- **Teste fixo** (n = {R['teste_fixo']['n']}, Z0 = {R['teste_fixo']['n_Z0']}): PR-AUC "
     f"{R['teste_fixo']['pr_auc']:.2f} (IC {R['teste_fixo']['pr_auc_ic'][0]:.2f}–{R['teste_fixo']['pr_auc_ic'][1]:.2f}), "
     f"ROC-AUC {R['teste_fixo']['roc_auc']:.2f}, ECE {R['teste_fixo']['ece']:.3f}; A = {R['teste_fixo']['A_n']} "
     f"imagens, precisão {ci(R['teste_fixo']['A_precisao'])}, cobertura de Z0 {R['teste_fixo']['cobertura_Z0']:.0%}",
     f"- Cabeça ordinal (Z0/Z1/Z2): κ ponderado {R['ordinal']['kappa_ponderado']:.2f}, acurácia "
     f"{R['ordinal']['acuracia']:.0%}", "", md(rd("s7_confusao_ordinal.csv")), "",
     "### Critérios de aceite", "", md(rd("s7_criterios_aceite.csv")), "",
     "### Leave-one-source-out", "", md(rd("s7_leave_one_source_out.csv")), "",
     "### Curva de aprendizado", "", md(rd("s7_curva_aprendizado.csv")), "",
     "### Robustez a corrupções (teste fixo)", "", md(rd("s7_robustez.csv")), "",
     "### Fatias", "", md(rd("s7_fatias.csv")), "",
     f"### Erros (dev, OOF): {R['erros']['FP']} falsos positivos de A, {R['erros']['FN']} falsos negativos, "
     f"{R['erros']['abstencoes']} abstenções", "",
     "![FN](figuras/s7_falsos_negativos_A.jpg)", "",
     "## S8 Roteamento — os três conjuntos", "",
     "Rótulo manual quando existe; senão o modelo. Cópias recebem o destino do representante só para rastreio.", "",
     md(pd.DataFrame([{"dataset": k, "n": v["n"], **{f"origem_{a}": b for a, b in v["por_origem"].items()},
                       **{f"split_{a}": b for a, b in v["por_split"].items()}}
                      for k, v in R["S8"].items() if k != "copias_propagadas"]).fillna(0)), "",
     f"Cópias propagadas (rastreio): {R['S8']['copias_propagadas']}", "",
     "**Sem zoom**", "", "![sem zoom](figuras/s8_dataset_sem_zoom.jpg)", "",
     "**Com zoom**", "", "![com zoom](figuras/s8_dataset_com_zoom.jpg)", "",
     "**Revisão**", "", "![revisão](figuras/s8_dataset_revisao.jpg)", "",
     "## S9 Comparação dos dois datasets", "",
     md(rd("s9_comparacao_A_B.csv")), "", md(rd("s9_fontes_A_B.csv")), ""]
(D / "RELATORIO.md").write_text("\n".join(L), encoding="utf-8")
print("ok", D / "RELATORIO.md")
