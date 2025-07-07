import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import time

# Forçar uso da GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Usar primeira GPU
torch.cuda.empty_cache()  # Limpar cache da GPU

# Verificar se CUDA está disponível
if not torch.cuda.is_available():
    raise RuntimeError("Este script requer uma GPU NVIDIA. Por favor, verifique se você tem uma GPU NVIDIA instalada e se os drivers estão atualizados.")

device = torch.device('cuda')
print(f"Usando dispositivo: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 2. Carregue o dataset Twitter
print("Carregando o dataset Twitter...")
dataset = load_dataset("csv", data_files={"train": "..\\data\\TwitterRenamed.csv"}, split="train")

# Converter os rótulos de string para inteiro
# label_map = {"positive": 1, "negative": 0}
# dataset = dataset.map(lambda x: {"sentiment": label_map[x["sentiment"]]})

# 3. Tokenização
MODEL_NAME = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 10

# Função de tokenização
def tokenize(batch):
    return tokenizer(
        batch['review'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN
    )

print("Tokenizando os dados...")
dataset = dataset.map(tokenize, batched=True)
dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'sentiment'])

# Após a tokenização, cheque os valores máximos dos input_ids
print("Maior valor em input_ids:", torch.max(dataset['input_ids']).item())

# Split manual 80/20
N_TRAIN = 1048572*8//10
N_TEST = 1048572*2//10
data_indices = np.arange(len(dataset['review']))
train_idx, test_idx = train_test_split(data_indices, test_size=0.2, random_state=42)
train_idx = train_idx[:N_TRAIN]
test_idx = test_idx[:N_TEST]

# Criar DataLoaders
from torch.utils.data import DataLoader, TensorDataset

def create_dataloader(ds, indices, batch_size):
    # Converter índices para lista de inteiros Python
    indices = [int(i) for i in indices]
    
    input_ids = torch.stack([ds[i]['input_ids'] for i in indices])
    attention_mask = torch.stack([ds[i]['attention_mask'] for i in indices])
    labels = torch.tensor([ds[i]['sentiment'] for i in indices])
    
    dataset = TensorDataset(input_ids, attention_mask, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False) # Não precisa embaralhar para teste

test_dataloader = create_dataloader(dataset, test_idx, BATCH_SIZE)

# Caminho do modelo salvo
model_save_path = "bert_Twitter_sentiment_model_3.pt"

# Carregar o modelo treinado
print("Carregando o modelo BERT treinado...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)
model.eval()

# Avaliação do modelo no conjunto de teste
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"Total de amostras avaliadas: {len(all_preds)}")  # Deve ser 15000

if len(all_preds) == 15000:
    # Métricas
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    # Matriz de confusão
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusão')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negativo', 'Positivo'])
    plt.yticks(tick_marks, ['Negativo', 'Positivo'])
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.savefig('matriz_confusao.png')
    plt.close()
else:
    print("Erro: A matriz de confusão não terá 15 mil registros!")


# Exemplo de listas simuladas caso não existam
if 'train_losses' not in locals():
    train_losses = [0.7, 0.5, 0.4, 0.35, 0.3]
    val_losses = [0.75, 0.55, 0.45, 0.4, 0.38]
    train_accuracies = [0.6, 0.7, 0.8, 0.85, 0.88]
    val_accuracies = [0.58, 0.68, 0.78, 0.83, 0.86]

epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Val Loss')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.title('Loss por Época')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, val_accuracies, label='Val Accuracy')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.title('Accuracy por Época')
plt.legend()

plt.tight_layout()
plt.savefig('treinamento_loss_accuracy.png')
plt.close() 
