# Análise da execução de 18/nov — o que o tratamento revelou

Dataset: `container-damage-detection-lt-6ylwp` (5.761 imagens, 640×640, caixas de
rust/dent/hole; split do Roboflow 4033/1151/577). Números completos em `RELATORIO.md`.

## 1. O dataset já vem com augmentation embutida (achado principal)

As imagens foram exportadas de outra versão do Roboflow **com augmentation** e reimportadas
(nomes com `.rf.<hash>`). Por isso:

- **5.761 imagens correspondem a só 1.845 fotos de origem** (~1.949 fotos distintas depois de
  separar colisões de nome por similaridade). **66% do dataset é redundante.**
- **39% das imagens têm rotação/cisalhamento artificial** (preenchimento preto ou cinza constante
  nos cantos) e 4,7% têm *cutout* (retângulos colados). Há também flips e ruído sal-e-pimenta.
- 506 imagens `aug_*` são pré-aumentadas, com até 30 cópias da mesma foto.

**Consequência para a planificação:** rotação, cisalhamento e perspectiva artificiais alteram
justamente a geometria que o planificador vai estimar, e o preenchimento não é uma borda
retangular (o recorte da seção 4 não resolve). Essas cópias não devem entrar na rotulagem de
planificabilidade, nos 4 + 1 pontos nem no teste real. Proposta: usar só a cópia limpa de cada
foto (`is_representative` já prioriza a cópia sem artefato) e tratar as aumentadas como
*augmentation* disponível só no treino. Restam **1.520 representantes limpos** (1.268 no `dev`,
252 no teste fixo).

## 2. O split original do Roboflow vaza

- **889 fotos de origem (3.960 imagens, 69%) aparecem em mais de um split.** 820 das 1.151
  imagens de validação e 391 das 577 de teste têm uma cópia da mesma foto no treino.
- 104 contêineres (código ISO 6346) aparecem em mais de um split.
- Qualquer métrica medida no split original estaria inflada (Barz & Denzler, 2020).

**Decisão aplicada (E5.4):** novo split com `StratifiedGroupKFold` por `group_id` (nome de origem
+ cluster de duplicata + contêiner), 6 folds, fold 0 = **teste fixo** (960 imagens, 274 grupos).
`check_no_leak` garante que nenhum grupo atravessa folds. O split original fica registrado em
`split_orig` para comparação.

## 3. Bordas, colagens e qualidade

- Bordas pretas retangulares (letterbox) são raras: 17 imagens; 5 com `safe_to_crop`.
- 488 imagens têm faixa escura **gradual** na borda — conteúdo real (sombra, noite); não recortar.
- Nenhuma colagem detectada depois de exigir conteúdo claro dos dois lados da faixa (a primeira
  versão do detector dava falso positivo em cenas escuras; registrado como estratégia que falhou).
- A qualidade JPEG estimada é ~75 em todo o dataset: é a recompressão do Roboflow, não da câmera.
  Não serve para distinguir fontes.
- Fontes diferem bastante em nitidez e ruído (ex.: `cam_fixa` e `frame_video` com ruído 3× maior
  que `camera_dsc`). Isso entra no baseline trivial (E0) para checar atalho.

## 4. Limitações desta execução

- **Embeddings:** usei o embedding que o próprio Roboflow fornece (768-d, tipo CLIP), não DINOv2/SSCD
  como o guia recomenda. Serve para E3/E5; a extração DINOv2 (E7) precisa de GPU (Colab).
- **Fonte (`source_id`)** é inferida pelo padrão do nome; não há metadado de origem no Roboflow.
- **Limiares** (pHash 6, cosseno 0,95, cosseno intra-nome 0,80, detectores de preenchimento) são
  iniciais. Calibrar com `pares_revisao_dedup.csv` (coluna `julgamento`).
- **Não há rótulo de planificabilidade nem de zoom**: o dataset tem caixas de avaria. A triagem
  (E8/E9) só pode ser treinada depois da rotulagem dos atributos da seção 2.

## 5. Próximos passos (cronograma da seção 12)

1. Validar com o orientador a decisão de usar só as cópias limpas (item 1).
2. Rodada de calibração: as 100 primeiras linhas de `amostra_rotulagem.csv`
   (`calibracao_kappa = True`) com 2 anotadores; calcular κ por atributo.
3. Julgar `pares_revisao_dedup.csv` para calibrar os limiares de E3.
4. Medir a fração Z0/Z1/Z2 na amostra (marco da semana 2).
5. E7: extrair DINOv2 no Colab (GPU) sobre os representantes limpos.
