# Classificador de Cartas Numéricas de Uno com CNN

Este projeto implementa uma Rede Neural Convolucional (CNN) usando PyTorch para reconhecimento automático de cartas numeradas do jogo Uno, classificando-as conforme cor e número. O objetivo é identificar cartas como `blue_3`, `red_7`, `yellow_0`, etc., a partir de imagens.

---

## Sobre o Modelo

O modelo implementado neste projeto é uma Rede Neural Convolucional (CNN) desenvolvida para reconhecimento automático de cartas numéricas do jogo Uno a partir de imagens. O objetivo principal é classificar corretamente cada carta de acordo com sua cor (azul, verde, vermelho, amarelo) e número (de 0 a 9), totalizando 40 classes diferentes.

### Arquitetura

A arquitetura do modelo é simples e eficiente, composta por duas camadas convolucionais seguidas de pooling, além de duas camadas totalmente conectadas (fully connected). Ela foi projetada para trabalhar com imagens coloridas (RGB) redimensionadas para 64x64 pixels, equilibrando desempenho e precisão para o conjunto de dados disponível.

- **Entradas:** Imagens de cartas Uno em diferentes cores e números.
- **Saídas:** Uma das 40 classes (ex: `blue_3`, `red_7`, `yellow_0`, etc.).
- **Camadas principais:**
    - 2 camadas convolucionais com ReLU e MaxPooling (extração de características visuais).
    - 2 camadas lineares (fully connected) para classificação final.

### Fluxo de Treinamento e Avaliação

1. **Pré-processamento:**  
   As imagens são redimensionadas e normalizadas para garantir consistência durante o treinamento e avaliação.
2. **Treinamento:**  
   O modelo é treinado utilizando o conjunto de dados de treino, com validação durante as épocas para evitar overfitting.
3. **Avaliação:**  
   Após o treinamento, a acurácia do modelo é medida utilizando um conjunto de teste independente.
4. **Inferência:**  
   É possível usar o modelo treinado para prever a classe de uma nova imagem de carta Uno.

### Aplicações e Resultados

Este modelo pode ser utilizado em aplicações como jogos digitais, sistemas de auxílio para pessoas com deficiência visual, ou projetos de visão computacional que envolvam reconhecimento de objetos em imagens. Com um conjunto de dados bem balanceado e imagens de boa qualidade, a CNN é capaz de atingir uma acurácia elevada na identificação das cartas.

---

## Dificuldades Encontradas

A principal dificuldade durante o desenvolvimento deste projeto foi relacionada à obtenção das imagens das cartas de Uno. Foi encontrado apenas um dataset disponível na internet, o que limitou a quantidade e a diversidade de dados para o treinamento do modelo. Isso ressalta a importância de bases de dados abertas e diversificadas para projetos de visão computacional. Caso novos datasets sejam disponibilizados no futuro, o modelo poderá ser expandido e aprimorado.

---

## Requisitos

- Python 3.8+
- torch >= 2.0.0
- torchvision >= 0.15.0
- Pillow >= 9.0.0

Instale as dependências com:
```bash
pip install -r requirements.txt
```

Ou manualmente:
```bash
pip install torch>=2.0.0 torchvision>=0.15.0 Pillow>=9.0.0
```

---

## Estrutura do Projeto

```
.
├── train_model.py        # Script de treinamento do modelo
├── cnn_model.py          # Definição da arquitetura da CNN
├── evaluate_model.py     # Script para avaliar a acurácia do modelo
├── inference.py          # Script para testar uma imagem manualmente
├── classes.txt           # Lista de classes (rótulos)
├── modelo_treinado.pth   # Arquivo do modelo treinado (gerado após treinamento)
└── dataset/
    ├── train/            # Imagens de treino (2119 arquivos, ~129MB)
    ├── val/              # Imagens de validação (161 arquivos, ~9.5MB)
    └── test/             # Imagens de teste (139 arquivos, ~10.3MB)
```

---

## Como Treinar o Modelo

Execute:
```bash
python train_model.py
```
O modelo será treinado com as imagens de `dataset/train` e validado em `dataset/val`. Ao final, o arquivo `modelo_treinado.pth` será salvo com os pesos do modelo treinado.

---

## Como Avaliar o Modelo

Após o treinamento, avalie o desempenho no conjunto de teste:
```bash
python avaliation_acuracia.py
```
O script irá mostrar a acurácia do modelo em `dataset/test`.

---

## Como Fazer Inferência Manual

Para testar uma imagem manualmente, edite o caminho da imagem em `inference.py`:
```python
image_path = "./avaliation/test1.jpg"  # Altere para sua imagem
```
E execute:
```bash
python inference.py
```
O script exibirá a classe prevista para a imagem fornecida.

---

## Sobre o Dataset

A pasta `dataset/` deve conter:
- `train/`: 2119 imagens (~129MB)
- `val/`: 161 imagens (~9.5MB)
- `test/`: 139 imagens (~10.3MB)

Cada subpasta representa uma classe listada em `classes.txt`.

---

## Sobre as Classes

O arquivo `classes.txt` contém todas as classes de cartas (cor+número) reconhecidas pelo modelo, por exemplo:

```
blue_0
blue_1
...
yellow_9
```

---

## Observações

- As transformações aplicadas em treino, validação, teste e inferência são idênticas (resize 64x64, normalização, etc).
- O modelo é salvo e carregado pelo arquivo `modelo_treinado.pth`.
- As classes devem estar sincronizadas com as pastas dentro de `dataset/` e o arquivo `classes.txt`.
- Scripts estão organizados para serem simples de usar, modificar e entender.

---
