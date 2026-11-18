# Guia de anotação do zoom — v0 (S2)

Escrito **antes** de rotular qualquer imagem. Versão registrada em cada rótulo (`versao_guia = v0`).

## Definições

- **Face**: painel retangular de um contêiner — lateral, porta (traseira), frente ou teto.
- **Canto**: castanha (*corner casting*) ou encontro de duas arestas estruturais da face,
  **dentro do quadro e identificável**.
- **Contêiner de referência**: o contêiner que contém a avaria mais visível. Com vários
  contêineres no quadro, é o mais proeminente (maior área, em foco).

## Árvore de decisão

1. Alguma face do contêiner de referência tem **os 4 cantos dentro do quadro**, identificáveis? → **Z0** (sem zoom)
2. Senão: aparece **ao menos 1 canto ou 2 arestas estruturais** da face (face cortada, mas a estrutura é reconhecível)? → **Z1** (zoom parcial)
3. Senão (só trecho de chapa, textura ou a avaria, sem canto) → **Z2** (detalhe)
4. Não dá para decidir (imagem ilegível, não é contêiner, ambígua) → **IND** (indeterminado → revisão)

## Casos de fronteira (regra escrita)

| Caso | Regra |
|---|---|
| Vários contêineres empilhados | Avalie só o contêiner de referência |
| Canto oculto por pessoa, veículo ou objeto | Conta como visível só se a posição do canto é inequívoca; senão desce para Z1 |
| Ângulo extremo (face muito oblíqua) | Rotule o zoom pelo que está no quadro; obliquidade é outro atributo |
| Imagem esticada para 640 × 640 | Rotule normalmente e marque `proporcao_distorcida` |
| Preenchimento de rotação ou cutout | Rotule o que se vê; um canto escondido pelo preenchimento não conta. O artefato já está marcado no manifesto |
| Render ou foto de catálogo (fundo branco) | Rotule o zoom normalmente; a fonte fica registrada |
| Interior do contêiner | Use o painel visível como face; sem cantos → Z1/Z2 |

## Rótulos provisórios desta versão

Nesta execução os rótulos foram produzidos por **anotador automático** (Claude, inspeção visual
das pranchas numeradas em 256 × 256 px), seguindo esta árvore. Eles servem para implementar e
validar o pipeline de ponta a ponta, **não substituem a anotação humana** da seção 2.6
(dois anotadores, κ, adjudicação). O arquivo `labels/zoom_rotulos.csv` aceita rótulos humanos no
mesmo formato (`anotador` diferente); o pipeline usa o rótulo adjudicado quando existir.

Colunas: `image_id, zoom_status (Z0|Z1|Z2|IND), anotador, rodada, versao_guia, observacao`.
