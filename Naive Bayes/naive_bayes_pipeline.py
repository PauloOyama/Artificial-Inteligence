# %%
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import string
import os

# %%
# Baixar recursos do NLTK se necessário
nltk.download('punkt')
nltk.download('stopwords')

# %%
# Carregar o dataset IMDB
DATA_PATH = os.path.join('..', 'data', 'IMDB Dataset.csv')
df = pd.read_csv(DATA_PATH)

# %%
# Amostrar 1000 exemplos para execução rápida
#np.random.seed(42)
#df = df.sample(1000, random_state=42).reset_index(drop=True)

# %%
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text.lower())
    stopwords_set = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stopwords_set]
    stemmer = PorterStemmer()
    tokens_stemmed = [stemmer.stem(token) for token in filtered_tokens]
    return tokens_stemmed

# %%
# Construir vocabulário simples (Bag of Words)
def build_vocab(texts, max_features=2000):
    freq = {}
    for text in texts:
        for token in preprocess_text(text):
            freq[token] = freq.get(token, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    vocab = {token: idx for idx, (token, _) in enumerate(sorted_tokens[:max_features])}
    return vocab

# %%
def vectorize(texts, vocab):
    X = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, text in enumerate(texts):
        tokens = preprocess_text(text)
        for token in tokens:
            idx = vocab.get(token)
            if idx is not None:
                X[i, idx] += 1
    return X

# %%
# Converter rótulos para 0 (negativo) e 1 (positivo)
df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})

# %%
# Split train/test
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['sentiment'])

# %%
# Construir vocabulário e vetorização
vocab = build_vocab(train_df['review'])
X_train = vectorize(train_df['review'], vocab)
X_test = vectorize(test_df['review'], vocab)
y_train = train_df['sentiment'].values
y_test = test_df['sentiment'].values

# %%
# Treinar o classificador Naive Bayes
clf = GaussianNB()
clf.fit(X_train, y_train)

# %%
# Avaliação
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Accuracy: {acc:.6f}')
print(f'Precision: {prec:.6f}')
print(f'Recall: {rec:.6f}')
print(f'F1-score: {f1:.6f}')

# %%
# Matriz de confusão com números
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(5,5))
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_nb.png')
plt.show()
plt.close()

# (Opcional) Salvar matriz como texto
np.savetxt('confusion_matrix_nb.txt', cm, fmt='%d')
