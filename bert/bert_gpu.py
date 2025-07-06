<<<<<<< HEAD
version https://git-lfs.github.com/spec/v1
oid sha256:eb5aa2ce5afe2115b22e1df15a6654cc7fff7adcbcd33572772eb94e04c157fb
size 10389
=======
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import time # Importar o módulo time

# Forçar uso da GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Usar primeira GPU
torch.cuda.empty_cache()  # Limpar cache da GPU

# Verificar se CUDA está disponível
if not torch.cuda.is_available():
    raise RuntimeError("Este script requer uma GPU NVIDIA. Por favor, verifique se você tem uma GPU NVIDIA instalada e se os drivers estão atualizados.")

device = torch.device('cuda')
print(f"Usando dispositivo: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 2. Carregue o dataset IMDB
print("Carregando o dataset IMDB...")
dataset = load_dataset("csv", data_files={"train": "..\\data\\IMDB Dataset.csv"}, split="train")

# Converter os rótulos de string para inteiro
label_map = {"positive": 1, "negative": 0}
dataset = dataset.map(lambda x: {"sentiment": label_map[x["sentiment"]]})

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
N_TRAIN = 35000
N_TEST = 15000
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
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_dataloader = create_dataloader(dataset, train_idx, BATCH_SIZE)
test_dataloader = create_dataloader(dataset, test_idx, BATCH_SIZE)

# 5. Modelo BERT para classificação
print("Carregando o modelo BERT...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model = model.to(device)

# 6. Configuração do treinamento
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = CrossEntropyLoss()

# Listas para armazenar métricas
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# 7. Treinamento
print("Treinando o modelo...")
model.train()
for epoch in range(EPOCHS):
    start_time = time.time()  # Início da época
    total_train_loss = 0
    total_train_correct = 0
    total_train_samples = 0

    for batch in train_dataloader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        total_train_correct += (preds == labels).sum().item()
        total_train_samples += labels.size(0)

    avg_train_loss = total_train_loss / len(train_dataloader)
    train_accuracy = total_train_correct / total_train_samples
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    # Avaliação após cada época
    model.eval()
    total_val_loss = 0
    total_val_correct = 0
    total_val_samples = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_val_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            total_val_correct += (preds == labels).sum().item()
            total_val_samples += labels.size(0)
    
    avg_val_loss = total_val_loss / len(test_dataloader)
    val_accuracy = total_val_correct / total_val_samples
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)
    model.train() # Voltar para modo de treinamento

    end_time = time.time()  # Fim da época
    epoch_time = end_time - start_time
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Tempo: {epoch_time:.2f}s")

# Salvar o modelo
model_save_path = "bert_imdb_sentiment_model_3.pt"
torch.save(model.state_dict(), model_save_path)
print(f"Modelo salvo em: {model_save_path}")

# 8. Avaliação Final
print("Avaliando o modelo...")
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

print(f"Total de amostras de teste utilizadas para avaliação: {len(all_labels)}")

# Calcular métricas
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
final_accuracy = accuracy_score(all_labels, all_preds)

print(f"\nTest Accuracy Final: {final_accuracy:.4f}")
print(f"Test Precision: {prec:.4f}")
print(f"Test Recall: {rec:.4f}")
print(f"Test F1-score: {f1:.4f}")
print("Confusion Matrix:\n", cm)

# Visualização da matriz de confusão
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['neg', 'pos'])
plt.yticks(tick_marks, ['neg', 'pos'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.tight_layout()
plt.savefig("confusion_matrix.png") # Salvar a matriz de confusão
plt.show()

# Plotar curvas de aprendizado
epochs_range = range(3)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("learning_curves.png") # Salvar as curvas de aprendizado
plt.show()

# 9. Exemplo de inferência
sample_text = "This movie was fantastic! I loved it."
inputs = tokenizer(sample_text, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN)
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
print(f"\nExemplo de inferência: '{sample_text}' => Label: {pred} (0=neg, 1=pos)")

# Exemplo de carregamento do modelo salvo
print("\nCarregando o modelo salvo para inferência...")
loaded_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
loaded_model.load_state_dict(torch.load(model_save_path))
loaded_model.to(device)
loaded_model.eval()

sample_text_loaded = "This movie was terrible! I hated it."
inputs_loaded = tokenizer(sample_text_loaded, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN)
inputs_loaded = {k: v.to(device) for k, v in inputs_loaded.items()}

with torch.no_grad():
    outputs_loaded = loaded_model(**inputs_loaded)
    pred_loaded = torch.argmax(outputs_loaded.logits, dim=1).item()

print(f"Exemplo de inferência (modelo carregado): '{sample_text_loaded}' => Label: {pred_loaded} (0=neg, 1=pos)")

# Salvar métricas em arquivo de texto
metrics_file_path = "training_metrics.txt"
with open(metrics_file_path, "w") as f:
    f.write(f"Final Test Accuracy: {final_accuracy:.4f}\n")
    f.write(f"Test Precision: {prec:.4f}\n")
    f.write(f"Test Recall: {rec:.4f}\n")
    f.write(f"Test F1-score: {f1:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(f"{cm}\n\n")
    
    f.write("Losses over Epochs:\n")
    for i, (tl, vl) in enumerate(zip(train_losses, val_losses)):
        f.write(f"Epoch {i+1}: Train Loss = {tl:.4f}, Val Loss = {vl:.4f}\n")
    
    f.write("\nAccuracies over Epochs:\n")
    for i, (ta, va) in enumerate(zip(train_accuracies, val_accuracies)):
        f.write(f"Epoch {i+1}: Train Acc = {ta:.4f}, Val Acc = {va:.4f}\n")

print(f"Métricas salvas em: {metrics_file_path}") 
>>>>>>> 9e7d5718bda85efbc2b0aa5346e0385b022bb86f
