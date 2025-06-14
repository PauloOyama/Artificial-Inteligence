# BERT Sentiment Analysis - IMDB

Este diretório contém scripts para análise de sentimentos no dataset IMDB utilizando o modelo BERT com PyTorch.

## Pré-requisitos

- Python 3.8 ou superior
- (Opcional, mas recomendado) GPU com CUDA para acelerar o treinamento
- [gdown](https://github.com/wkentaro/gdown) para baixar arquivos do Google Drive via terminal

### Instale as dependências

No terminal, execute:

```bash
pip install -r requirements_bert_gpu.txt
```

Se precisar baixar o `gdown`:

```bash
pip install gdown
```

## Dataset

O dataset utilizado é o [IMDB Dataset of 50K Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/), que deve estar disponível em:

```
../data/IMDB Dataset.csv
```

em relação a este diretório.

## Baixando o Modelo Treinado

Se preferir utilizar um modelo já treinado, baixe-o do Google Drive usando o link abaixo

```
https://drive.google.com/drive/folders/1tbam26k_E2WA2-mSkLuJCNHWjfGgprME?usp=drive_link 

```


Coloque o arquivo baixado no mesmo diretório dos scripts.

## Como Executar

### 1. Treinamento e Avaliação com `bert_gpu.py`

Este script realiza:
- Carregamento e pré-processamento dos dados
- Tokenização com BERT
- Treinamento do modelo
- Avaliação e geração de métricas
- Salvamento do modelo e gráficos

Execute:

```bash
python bert_gpu.py
```

### 2. Treinamento e Avaliação com `bert_gpu_dois.py`

Este script é uma variação do anterior, com pequenas diferenças na divisão dos dados e fluxo de treinamento.

Execute:

```bash
python bert_gpu_dois.py
```

## Saídas

- Modelos treinados: `bert_imdb_sentiment_model.pt` ou `bert_imdb_sentiment_model_3.pt`
- Gráficos de acurácia, perda e matriz de confusão em PNG
- Métricas salvas em arquivos de texto

## Observações

- Certifique-se de que o arquivo do dataset está no caminho correto.
- Para melhores resultados, execute em uma máquina com GPU.
- Para usar o modelo pré-treinado, basta baixar e colocar o arquivo no diretório dos scripts.
