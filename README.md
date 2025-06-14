# Inteligência Artificial - Análise de Sentimentos IMDB

Este projeto reúne diferentes abordagens de aprendizado de máquina e deep learning para análise de sentimentos no dataset IMDB, explorando desde modelos clássicos até arquiteturas modernas de redes neurais.

## Estrutura do Projeto

```
Artificial-Inteligence/
│
├── bert/           # Modelos BERT para análise de sentimentos (PyTorch)
│
├── LSTM/           # Modelos LSTM para análise de sentimentos (SentimentLSTM)
│
├── Naive Bayes/    # Abordagem clássica com Naive Bayes (Jupyter)
│
└── data/           # Dataset IMDB (arquivo CSV)
```

## Descrição das Pastas

- **bert/**
  Scripts para treinamento, avaliação e inferência usando o modelo BERT (PyTorch). Inclui exemplos de uso, requisitos e instruções para baixar modelos pré-treinados.

- **rnn/**
  

- **LSTM/**
  Implementação de redes neurais recorrentes (LSTM) para análise de sentimentos. Contém scripts para treinamento, avaliação e geração de métricas.

- **Naive Bayes/**
  Notebooks Jupyter com experimentos utilizando o classificador Naive Bayes, além de análises exploratórias e visualização de resultados.

- **data/**
  Contém o arquivo `IMDB Dataset.csv` com 50.000 avaliações de filmes rotuladas como positivas ou negativas.

## Como começar

1. **Escolha a abordagem:**
   - Para modelos clássicos, acesse `Naive Bayes/`.
   - Para deep learning, explore `rnn/`, `LSTM/` ou `bert/`.

2. **Instale as dependências** de cada pasta conforme instruções nos respectivos READMEs.

3. **Execute os scripts ou notebooks** para treinar, avaliar e visualizar os resultados.

## Sobre o Projeto

O objetivo é comparar diferentes técnicas de processamento de linguagem natural (NLP) para análise de sentimentos, desde métodos tradicionais até modelos de última geração, promovendo aprendizado prático e análise crítica dos resultados.

---