# Análise da divisão em sem zoom e com zoom — 18/nov

Números completos em `RELATORIO.md`; cartão do modelo e datasheet em `docs/`.

## Resultado

O dataset único (1.949 fotos distintas) foi dividido em **137 imagens sem zoom**, **1.674 com zoom** e
**138 em revisão**. O modelo de triagem (DINOv2 congelado + regressão logística calibrada) separa Z0
do resto com PR-AUC 0,78 na validação agrupada e 0,74 no teste fixo, contra 0,24 do baseline que só
conhece a fonte.

## O que os dados mostraram

1. **Fotos sem zoom são raras:** 13% (IC 8–21%) na amostra aleatória. O guia previa esse risco
   ("poucos Z0", seção 16.6): o dataset sem zoom é pequeno e o plano B é priorizar o planificador
   clássico e o conjunto real 4 + 1.
2. **Não há picos claros de zoom nos proxies:** o teste de dip não rejeitou unimodalidade em nenhuma
   fonte ou classe (q ≥ 0,99). A área das caixas sozinha não separa níveis de zoom; os proxies servem
   para estratificar, como o guia diz, e não como rótulo.
3. **O embedding enxerga o enquadramento:** os 5 clusters do k-means no DINOv2 têm a associação mais
   forte com o zoom (V de Cramér 0,42), mais que a fonte (0,40). É o que torna o modelo viável.
4. **Zoom e fonte estão confundidos** (V = 0,40): `codigo_iso` e `camera_dsc` quase não têm Z0;
   `web_container` e `numerica` têm muitos. Por isso o baseline só-fonte e o leave-one-source-out
   são obrigatórios. O modelo supera o só-fonte com IC pareado bem longe de zero (+0,58).
5. **Augmentation quase não muda o zoom:** em 3 de 30 clusters as cópias receberam zoom diferente, e
   em 2 a decisão Z0 mudou (rotação que corta um canto).

## Decisões de modelagem

- **224 px venceu 448 px** no token CLS (0,78 × 0,71). Com ~400 exemplos, o vetor de 224 px
  generaliza melhor; a resolução maior não trouxe informação de enquadramento (experimento Z-3).
- **Onde está a informação** (Z-4): a 448 px, a média dos patches (0,74) e a concatenação (0,74) superam o
  CLS (0,71); mesmo assim o melhor geral é o CLS a 224 px (0,78).
- **OpenCV sozinho é fraco** (0,32) e não complementa o DINOv2 (Z-6: 0,75 < 0,78).
- **SVM ≈ LR** (0,79 × 0,78, dentro do ruído): mantida a LR, mais simples e já calibrada.
- **Curva de aprendizado** sobe até 300 rotuladas (0,63 → 0,80) e estabiliza: rotular mais ajuda
  pouco na média, mas reduziria o IC do teste, que é o critério que falhou.
- **Calibração:** ECE 0,02 (dev) e 0,04 (teste), dentro do critério ≤ 0,05.

## Critérios de aceite

| Critério | Valor | Situação |
|---|---|---|
| Precisão de A, limite inferior do IC (teste) ≥ 0,90 | 0,57 | **falhou** (só 5 aceitas) |
| κ ponderado do ordinal ≥ 0,6 | 0,69 | passou |
| ECE ≤ 0,05 | 0,039 | passou |
| Precisão de A ≥ 0,85 por fonte (n ≥ 30) | 1,00 | passou |
| Ganho sobre trivial e só-fonte (IC exclui zero) | +0,42 / +0,47 (limite inferior) | passou |

O critério que falhou não é por erro do modelo (as 5 aceitas estão certas), mas por **tamanho**: com
15 Z0 no teste fixo e cobertura de 33%, o IC é largo demais. Saídas da seção 16.3 (S3): aumentar o
teste para 25–30% dos grupos ou rotular o teste inteiro com dupla anotação.

## Erros

Nenhum falso positivo no dataset sem zoom (dev, fora-da-amostra). Os 16 falsos negativos são fotos de
face inteira mandadas para com zoom, em quatro padrões: vista da porta com a face pequena no quadro,
vista de cima (teto), cenas noturnas e imagens com rotação artificial.

## Limitações

- **Rótulos provisórios** por anotador automático: κ entre humanos ainda não existe. A rotulagem
  humana substitui `labels/zoom_rotulos.csv` e o pipeline roda de novo sem mudar código.
- `geometria_suficiente` (P1) não foi rotulada: o dataset sem zoom é definido só pelo zoom, como manda
  a seção 15.0; a visão "sem zoom e geometria suficiente" fica para quando houver esse rótulo.
- Regra por detector de castanhas (Z-7) não testada: o detector é da fase F4.

## Próximos passos

1. Rotulagem humana com o guia v0 (2 anotadores nas 100 imagens de calibração), κ e adjudicação.
2. Revisar as 138 imagens em revisão (ficam fora dos dois datasets até lá).
3. Aumentar o teste rotulado para fechar o critério de precisão de A.
4. Rotular `geometria_suficiente` no dataset sem zoom e o conjunto 4 + 1.
