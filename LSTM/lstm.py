import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import re
import random

# Fixar seeds para reprodutibilidade
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

# Parâmetros
BATCH_SIZE = 64
EMBED_DIM = 128
HIDDEN_DIM = 128
EPOCHS = 8
MAXLEN = 120
VOCAB_SIZE = 10000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pré-processamento simples
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

def tokenize(text):
    return text.split()

# Dataset personalizado
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, maxlen):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.maxlen = maxlen
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        tokens = tokenize(clean_text(self.texts[idx]))
        ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        if len(ids) < self.maxlen:
            ids += [self.vocab['<PAD>']] * (self.maxlen - len(ids))
        else:
            ids = ids[:self.maxlen]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

# Construir vocabulário
def build_vocab(texts, vocab_size=VOCAB_SIZE):
    freq = {}
    for text in texts:
        for token in tokenize(clean_text(text)):
            freq[token] = freq.get(token, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for i, (token, _) in enumerate(sorted_tokens[:vocab_size-2]):
        vocab[token] = i + 2
    return vocab

# Modelo LSTM
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=0.3, num_layers=2)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        out = h_n[-1]
        out = self.fc(out)
        return self.sigmoid(out).squeeze(1)

# Carregar o dataset Twitter
# Substitua o caminho abaixo pelo caminho correto do seu arquivo CSV
DATASET_PATH = '..\\data\\TwitterRenamed.csv'  # Compatível com Windows
# DATASET_PATH = '../data/TwitterRenamed.csv'    # Descomente para sistemas Unix/Linux/Mac

df = pd.read_csv(DATASET_PATH)

# Split
print('Split Traint/Test')
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['sentiment'])
df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=42, stratify=df_train['sentiment'])

# Vocabulário
total_texts = pd.concat([df_train['review'], df_val['review']])
vocab = build_vocab(total_texts.tolist())

print('Building Vocab')
# Datasets e DataLoaders
train_ds = TextDataset(df_train['review'].tolist(), df_train['sentiment'].tolist(), vocab, MAXLEN)
val_ds = TextDataset(df_val['review'].tolist(), df_val['sentiment'].tolist(), vocab, MAXLEN)
test_ds = TextDataset(df_test['review'].tolist(), df_test['sentiment'].tolist(), vocab, MAXLEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# Modelo, loss, otimizador
model = SentimentLSTM(len(vocab), EMBED_DIM, HIDDEN_DIM).to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Treinamento
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = (out > 0.5).float()
        total_correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            preds = (out > 0.5).float()
            total_correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / total, total_correct / total
print('Starting Training ')
train_losses, val_losses, train_accs, val_accs = [], [], [], []
for epoch in range(EPOCHS):
    tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = eval_epoch(model, val_loader, criterion)
    train_losses.append(tr_loss)
    val_losses.append(val_loss)
    train_accs.append(tr_acc)
    val_accs.append(val_acc)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {tr_acc:.4f} | Val Acc: {val_acc:.4f}")

# Avaliação final
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            preds = (out > 0.5).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)
    return np.array(y_true), np.array(y_pred)

print('Evaluating')
y_true, y_pred = evaluate(model, test_loader)

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Acurácia: {acc:.4f}")
print(f"Precisão: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")

cm = confusion_matrix(y_true, y_pred)
print("Matriz de Confusão:")
print(cm)

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(5,5))
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('Matriz de Confusão')
plt.savefig('confusion_matrix.png')
plt.close()

np.savetxt('confusion_matrix.txt', cm, fmt='%d')

# Salvar métricas e gráficos
os.makedirs('metricas', exist_ok=True)
np.savetxt('metricas/confusion_matrix.csv', cm, delimiter=',', fmt='%d')
with open('metricas/metrics.txt', 'w') as f:
    f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\n")

plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')
plt.savefig('metricas/loss_curve.png')

plt.figure()
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')
plt.savefig('metricas/accuracy_curve.png')

plt.figure()
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
plt.savefig('metricas/confusion_matrix.png')

# Salvar modelo
torch.save(model.state_dict(), 'metricas/sentiment_lstm.pt')
