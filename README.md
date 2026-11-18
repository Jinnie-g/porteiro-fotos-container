# O Porteiro de Fotos — tratamento e triagem do dataset

Implementação do *Guia metodológico: triagem de imagens e tratamento do dataset para
planificação* (seções 3, 4 e 13). Esta versão cobre a fase **F0/F1** (semanas 1 e 2):
E1 inventário, E2 integridade, E3 duplicatas, E4 caracterização, E5 auditoria do split
e E6 amostra para rotulagem. E7 (DINOv2) e E8/E9 (modelo de triagem) dependem dos
rótulos de planificabilidade e de GPU, e ficam para a próxima etapa.

## Como rodar

```bash
pip install -r requirements.txt
export ROBOFLOW_API_KEY=...            # nunca no código nem no Git
python scripts/download_roboflow.py    # baixa sem criar versão no Roboflow
python scripts/run_treatment.py        # gera results/run_<data>/
python -m pytest -q tests
```

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

## Saídas de uma execução (`results/run_<data>/`)

- `RELATORIO.md` — números e figuras gerados automaticamente
- `ANALISE.md` — interpretação e decisões (escrita à mão)
- `manifesto.parquet` — uma linha por imagem com todas as colunas medidas, grupos e folds
- `auditoria_vazamento.csv` — vizinho mais próximo no treino de cada imagem de valid/test
- `pares_revisao_dedup.csv` — pares por faixa de similaridade para julgar à mão (E3.3)
- `amostra_rotulagem.csv` — amostra para rotulagem com as colunas de atributos vazias
- `run_info.json`, `pip_freeze.txt` — reprodutibilidade
