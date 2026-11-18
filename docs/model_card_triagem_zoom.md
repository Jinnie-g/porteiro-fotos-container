# Cartão do modelo — triagem de zoom (Z0 × resto)

Formato de Mitchell et al. (2019). Números da execução `results/zoom_2026-11-18/`.

## Detalhes

- **O que faz:** estima `p_Z0`, a probabilidade de a imagem mostrar uma face inteira de contêiner
  (os 4 cantos no quadro). Com dois limiares, encaminha a imagem para o dataset **sem zoom**, o
  dataset **com zoom** ou a **revisão** humana (abstenção).
- **Arquitetura:** DINOv2 ViT-S/14 congelado (pesos oficiais da Meta), token CLS com a imagem inteira
  redimensionada para 224 × 224 px (sem recorte central) → padronização → regressão logística L2
  (`class_weight` balanceado, C escolhido por CV interna agrupada) → calibração de Platt ajustada em
  predições fora-da-amostra.
- **Versão:** `zoom-Z3a_dino_cls_224-LR-Platt`, pipeline 0.2.0. Semente 42.
- **Limiares:** τ_alto = 0,861 (vai para sem zoom), τ_baixo = 0,406 (abaixo vai para com zoom).
  Escolhidos para precisão-alvo de 0,95 nos dois lados, fora-da-amostra.
- **Cabeça secundária:** regressão logística multinomial Z0/Z1/Z2 com as mesmas características.

## Uso pretendido

- Separar o dataset de avarias de contêineres em sem zoom (entrada do planificador) e com zoom
  (ramo com zoom), mandando os casos incertos para revisão.
- **Fora do escopo:** decidir se a imagem é planificável (`geometria_suficiente` ainda não é
  rotulada); medir avarias; imagens de outros domínios.

## Dados

- **Treino/avaliação:** 556 representantes rotulados (398 em `dev`, 158 no teste fixo), sem cópias
  de augmentation. Grupos por foto de origem e contêiner, nunca divididos entre folds.
- **Rótulos provisórios** feitos por anotador automático (Claude, visão) com o guia v0
  (`docs/guia_anotacao_zoom.md`). Consistência no reteste às cegas: κ ponderado 0,95. **Não há
  ainda κ entre anotadores humanos.**
- **Prevalência de Z0:** 13% (IC 95% 8–21%) na amostra aleatória.

## Métricas (CV agrupada 5 × 5 em dev; confirmação no teste fixo)

| Métrica | Dev (OOF) | Teste fixo |
|---|---|---|
| PR-AUC | 0,78 (IC 0,71–0,88) | 0,74 (IC 0,50–0,90) |
| ROC-AUC | 0,95 | 0,93 |
| ECE | 0,023 | 0,039 |
| Precisão de A (sem zoom) | 15/15 = 1,00 (IC 0,80–1,00) | 5/5 = 1,00 (IC 0,57–1,00) |
| Cobertura de Z0 | 25% | 33% |
| Abstenção | 12% | 7% |
| κ ponderado (ordinal) | 0,69 | — |

Ganho de PR-AUC sobre o baseline só-fonte: +0,58 (IC 0,47–0,66); sobre o trivial: +0,53 (IC 0,42–0,63).

## Limitações e riscos

- **Cobertura baixa:** para manter a precisão de 0,95, só 25–33% das fotos Z0 vão direto para o
  dataset sem zoom; o restante cai em revisão ou, em 16 casos do dev, em com zoom (falsos negativos).
- **Teste pequeno:** só 15 Z0 no teste fixo; o limite inferior do IC da precisão de A (0,57) não
  atinge o critério de 0,90.
- **Fonte:** o zoom está associado à fonte (V de Cramér 0,40). O modelo supera com folga o
  baseline só-fonte, mas no leave-one-source-out a fonte `codigo_iso` (1 Z0 em 50) tem PR-AUC 0,13.
- **Erros típicos:** vistas da porta ou de cima, cenas noturnas, rotação artificial.
- **Robustez:** queda de PR-AUC ≤ 0,04 com desfoque, ruído, JPEG q20 e escurecimento.
