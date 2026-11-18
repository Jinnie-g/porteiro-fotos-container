# Datasheet — datasets sem zoom, com zoom e revisão

Formato de Gebru et al. (2021). Execução `results/zoom_2026-11-18/`.

## Motivação

Separar o dataset de avarias de contêineres em duas entradas com hipóteses geométricas diferentes:
**sem zoom** (face inteira, 4 cantos → homografia completa, entrada do planificador) e **com zoom**
(face cortada ou detalhe → ramo com zoom). Casos incertos ficam em **revisão**.

## Composição

| Dataset | Imagens | Por rótulo | Pelo modelo | Caixas de avaria |
|---|---|---|---|---|
| `sem_zoom` | 137 | 75 | 62 | 658 |
| `com_zoom` | 1.674 | 481 | 1.193 | 3.637 |
| `revisao` | 138 | 4 | 134 | 671 |

- **Unidade:** só os 1.949 representantes do dataset único (seção 15): uma imagem por foto de origem.
  As 3.812 cópias de augmentation recebem o destino do seu representante apenas para rastreio
  (`destino_propagado_copias.parquet`) e não entram nos datasets.
- **Origem:** projeto Roboflow `giovanna-alcantara/container-damage-detection-lt-6ylwp`
  (CC BY 4.0), 640 × 640 px, caixas rust/dent/hole.
- **Nenhum pixel foi alterado:** os arquivos exportados são os originais.
- **Colunas principais:** `zoom_status` (rótulo, quando existe), `p_Z0`, `destino_modelo`, `destino`,
  `origem_destino`, `geometria_suficiente` (vazia: depende de rótulo ainda inexistente),
  `group_id`, `dup_cluster_id`, `split_novo`, `versao_modelo`.

## Diferenças entre sem zoom e com zoom (S9)

- **Fontes** diferem bastante (V de Cramér 0,31): `web_container`, `numerica` e `outra` concentram
  o sem zoom; `codigo_iso` e `camera_dsc` são quase só com zoom.
- **Escala das avarias:** a maior caixa ocupa 1,3% da imagem no sem zoom contra 6,4% no com zoom
  (delta de Cliff −0,51).
- **Classes:** ferrugem é mais comum no sem zoom (46% × 33% das imagens); amassado no com zoom
  (58% × 45%); furo não difere (12% × 10%).
- **Qualidade:** imagens sem zoom são mais nítidas (Laplaciano mediano 384 × 213) e com menos
  dominante de cor.

## Coleta e rotulagem

Rótulos de zoom provisórios por anotador automático (guia v0). Ver o cartão do modelo e
`docs/guia_anotacao_zoom.md`.

## Usos e cuidados

- O dataset sem zoom é pequeno (137). Para a planificação, combine com o conjunto real 4 + 1 da
  seção 2.7 e priorize o planificador clássico (plano B da seção 16.6).
- Como fonte e zoom estão associados, avalie segmentadores por ramo **e** por fonte.
- Regenerar: `python scripts/run_zoom.py && python scripts/export_datasets.py`.
