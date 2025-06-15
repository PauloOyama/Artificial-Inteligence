# Naive Bayes - Análise de Sentimentos IMDB

Esta pasta contém experimentos de análise de sentimentos no dataset IMDB utilizando o classificador Naive Bayes com scikit-learn.

## Pré-requisitos

- Python 3.8+
- Jupyter Notebook

### Instale as dependências

No terminal, execute:

```bash
pip install -r requirements.txt
```

## Como rodar os notebooks

1. Abra o terminal na pasta `Naive Bayes`.
2. Inicie o Jupyter Notebook:

```bash
jupyter notebook
```

3. Abra e execute os notebooks:
   - `analysis.ipynb`: Análise exploratória e resultados.
   - `sklearn.ipynb`: Implementação do Naive Bayes com scikit-learn.

## Arquivos

- `analysis.ipynb`: Análise dos dados e visualização de métricas.
- `sklearn.ipynb`: Treinamento e avaliação do Naive Bayes.
- `requirements.txt`: Dependências necessárias.
- `results.json`, `matrix_confusao.png`, `Number of interactions.png`: Resultados e gráficos gerados.

---
## Naive Bayes Pipeline

### Instalação de dependências

Recomenda-se o uso de um ambiente virtual (por exemplo, com `venv`):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

Instale as dependências necessárias:

```bash
pip install -r requirements_pipeline.txt
```

### Execução

Para rodar o pipeline de Naive Bayes:

```bash
python naive_bayes_pipeline.py
```

O script irá:
- Carregar o dataset IMDB.
- Realizar o pré-processamento dos textos.
- Treinar um classificador Naive Bayes.
- Exibir métricas de acurácia, recall e f1-score no terminal.
- Gerar e salvar a matriz de confusão como `confusion_matrix.png` na pasta.
