<<<<<<< HEAD
version https://git-lfs.github.com/spec/v1
oid sha256:7f6def448baddf25026454f03f93207afe43e56a38bf9a26fedcc09cb8e5c970
size 1519
=======
# LSTM Sentiment Analysis

Este diretório contém um script para análise de sentimentos em reviews do IMDB utilizando uma rede neural LSTM implementada em PyTorch.

## Pré-requisitos

- Python 3.9
- PyTorch
- scikit-learn
- pandas
- numpy
- matplotlib

Você pode instalar as dependências principais com:

```bash
pip install torch scikit-learn pandas numpy matplotlib
```

## Como usar

1. **Coloque o dataset** `IMDB Dataset.csv` na pasta `../data/` (um nível acima deste diretório), ou ajuste o caminho no script `lstm.py` conforme necessário.
2. **Execute o script:**

```bash
python lstm.py
```

O script irá:
- Treinar um modelo LSTM para classificação de sentimentos (positivo/negativo).
- Avaliar o modelo em um conjunto de teste.
- Gerar e salvar gráficos de curva de aprendizado e matriz de confusão.

## Saídas geradas

- `confusion_matrix.png`: Imagem da matriz de confusão com os valores numéricos.
- `confusion_matrix.txt`: Matriz de confusão em formato texto.
- Gráficos de loss e acurácia por época (se implementado no script).
- Métricas de avaliação impressas no terminal (acurácia, precisão, recall, F1-score).

## Observações

- O script já fixa a seed para reprodutibilidade.
- Parâmetros como tamanho do batch, embedding, épocas e tamanho do vocabulário podem ser ajustados no início do script.
- O modelo é simples e serve como baseline para comparação com abordagens mais avançadas (ex: BERT).

---

Dúvidas ou sugestões? Abra uma issue ou entre em contato!
>>>>>>> 9e7d5718bda85efbc2b0aa5346e0385b022bb86f
