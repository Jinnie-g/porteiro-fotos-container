# O Porteiro de Fotos — tratamento e triagem do dataset

Implementação do *Guia metodológico: triagem de imagens e tratamento do dataset para
planificação*:

- **Fase F0/F1 (seções 3, 4, 13 e 15):** E1 inventário, E2 integridade, E3 duplicatas,
  E4 caracterização, E5 auditoria do split, E6 amostra para rotulagem e o dataset único
  (um representante por foto, com invariantes testados).
- **Seção 16:** divisão em **sem zoom** e **com zoom** (S0 a S10): exploração sem rótulos,
  guia de anotação, amostragem, rotulagem e confiabilidade, modelo DINOv2 + regressão
  logística calibrada com abstenção, avaliação agrupada, roteamento e comparação dos datasets.

## Como rodar

```bash
pip install -r requirements.txt
export ROBOFLOW_API_KEY=...            # nunca no código nem no Git
python scripts/download_roboflow.py    # baixa sem criar versão no Roboflow
python scripts/run_treatment.py        # gera results/run_<data>/
python -m pytest -q tests
```

### Seção 16 — sem zoom × com zoom

```bash
python scripts/make_label_sheets.py principal   # pranchas numeradas para rotular (S4)
python scripts/extract_dino.py                  # DINOv2 ViT-S/14, 224 e 448 px (CPU ~10 min)
python scripts/run_zoom.py                      # S0-S10 -> results/zoom_<data>/
python scripts/report_zoom.py                   # RELATORIO.md da execução
python scripts/export_datasets.py               # data/datasets/{sem_zoom,com_zoom,revisao}/
```

Os pesos do DINOv2 vêm de `dl.fbaipublicfiles.com` (salvos em `data/models/`). Os rótulos ficam
em `labels/zoom_rotulos.csv` (`image_id, zoom_status, anotador, rodada, versao_guia`); trocar
pelos rótulos humanos e rodar de novo não exige mudar código. Guia de anotação:
`docs/guia_anotacao_zoom.md`; cartão do modelo e datasheet em `docs/`.

O download usa a API de busca do Roboflow (lista imagens e split) e o endpoint de cada
imagem (caixas, URL do original, embedding). Nada é modificado no projeto remoto.
Imagens ficam em `data/` (fora do Git); só as saídas leves vão para `results/`.

## Mapa guia → código

| Etapa | Módulo | Produz |
|---|---|---|
| E0 | `roboflow_io.py` | imagens, metadados e embeddings do Roboflow |
| E1, E2 | `ingest.py` | manifesto com SHA-256, nome de origem, fonte, código ISO 6346, integridade |
| E4 | `quality.py` | bordas pretas por lado, colagem, `safe_to_crop`, **preenchimento de rotação**, **cutout**, nitidez, ruído, banda efetiva, JPEG, iluminação, cor |
| E4 | `geometry.py` | Canny 4×4, tensor de estrutura, LSD, paralelismo, Shi-Tomasi, periodicidade das corrugações |
| E3 | `dedup.py` | `dup_cluster_id`, `group_id`, representantes, pHash invariante a flip, pares para calibrar o limiar |
| E5 | `dedup.py`, `split.py` | auditoria de vazamento do split original; novo split agrupado (fold 0 = teste fixo) |
| E6 | `sampling.py` | amostra estratificada de 800 imagens + 100 para calibração de κ |
| 2.4 | `labels.py` | classe P1–P4 derivada por regra dos atributos rotulados |
| Registro | `env.py` | commit, versões, `pip freeze`, semente |
| 15 (D4, D8) | `consolidate.py` | dataset único (visão `is_representative`), motivo do representante, invariantes |
| 16 S1 | `zoom_explore.py` | proxies de enquadramento, teste de dip, k-means no embedding, medoides |
| 16 S3 | `zoom_sampling.py` | amostra de rotulagem (dev estratificado + teste fixo), reteste, variação da augmentation |
| 16 S4, S5, S7, S9 | `stats.py` | κ (simples e ponderado), Wilson, V de Cramér, informação mútua, BH, delta de Cliff, bootstrap por grupo |
| E7 / 16 S6 | `features_dino.py` | DINOv2 ViT-S/14 congelado, sem recorte central, CLS e média dos patches |
| 16 S6-S8 | `zoom_model.py` | conjuntos de características, LR/SVM + Platt sem vazamento, CV agrupada repetida, limiares, abstenção, roteamento |

## Saídas da seção 16 (`results/zoom_<data>/`)

- `RELATORIO.md` (automático) e `ANALISE.md` (interpretação)
- `dataset_sem_zoom.parquet`, `dataset_com_zoom.parquet`, `dataset_revisao.parquet`
- `s1_*` a `s9_*.csv`: uma tabela por passo do guia (matriz de experimentos, critérios de aceite,
  leave-one-source-out, curva de aprendizado, robustez, comparação A × B)
- `figuras/`: medoides dos clusters, amostras de cada dataset, falsos negativos

## Saídas de uma execução (`results/run_<data>/`)

- `RELATORIO.md` — números e figuras gerados automaticamente
- `ANALISE.md` — interpretação e decisões (escrita à mão)
- `manifesto.parquet` — uma linha por imagem com todas as colunas medidas, grupos e folds
- `auditoria_vazamento.csv` — vizinho mais próximo no treino de cada imagem de valid/test
- `pares_revisao_dedup.csv` — pares por faixa de similaridade para julgar à mão (E3.3)
- `amostra_rotulagem.csv` — amostra para rotulagem com as colunas de atributos vazias
- `run_info.json`, `pip_freeze.txt` — reprodutibilidade
