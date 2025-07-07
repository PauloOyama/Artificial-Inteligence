import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs disponíveis:", gpus)
if not gpus:
    print("\n[ATENÇÃO] Nenhuma GPU foi reconhecida pelo TensorFlow.\n" 
          "Verifique se os drivers NVIDIA, CUDA Toolkit e cuDNN compatíveis estão instalados e configurados corretamente.\n" 
          "Consulte https://www.tensorflow.org/install/source#gpu para detalhes de compatibilidade.\n")
else:
    print(f"[OK] {len(gpus)} GPU(s) reconhecida(s) pelo TensorFlow.")

from transformers import TFBertForSequenceClassification, BertTokenizerFast
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# 2. Carregue o dataset Twitter
print("Carregando o dataset Twitter...")
dataset = load_dataset("csv", data_files={"train": "../data/Twitter Dataset.csv"}, split="train")

# Converter os rótulos de string para inteiro
label_map = {"positive": 1, "negative": 0}
dataset = dataset.map(lambda x: {"sentiment": label_map[x["sentiment"]]})

# 3. Tokenização
MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 3

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
dataset.set_format('tensorflow', columns=['input_ids', 'attention_mask', 'sentiment'])

# Após a tokenização, cheque os valores máximos dos input_ids
print("Maior valor em input_ids:", np.max(np.array(dataset['input_ids'])))

# Split manual 80/20
N_TRAIN = 4000
N_TEST = 1000
data_indices = np.arange(len(dataset['review']))
train_idx, test_idx = train_test_split(data_indices, test_size=0.2, random_state=42)
train_idx = train_idx[:N_TRAIN]
test_idx = test_idx[:N_TEST]

def to_tf_dataset(ds, indices):
    return tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': np.array(ds['input_ids'])[indices],
            'attention_mask': np.array(ds['attention_mask'])[indices]
        },
        np.array(ds['sentiment'])[indices]
    )).batch(BATCH_SIZE)

train_dataset = to_tf_dataset(dataset, train_idx)
test_dataset = to_tf_dataset(dataset, test_idx)

# 5. Modelo BERT para classificação
print("Carregando o modelo BERT...")
model = TFBertForSequenceClassification.from_pretrained(MODEL_NAME)

# 6. Compilação do modelo
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# 7. Treinamento
print("Treinando o modelo...")
history = model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset)

# 8. Avaliação
print("Avaliando o modelo...")
results = model.evaluate(test_dataset)
print(f"\nTest Loss: {results[0]:.4f} | Test Accuracy: {results[1]:.4f}")

# Calcular precisão manualmente
# Obter previsões para o conjunto de teste
y_true = np.array(dataset['sentiment'])[test_idx]
y_pred_logits = model.predict(test_dataset).logits
y_pred = np.argmax(y_pred_logits, axis=1)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
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
plt.show()

# 9. Exemplo de inferência
sample_text = "This movie was fantastic! I loved it."
inputs = tokenizer(sample_text, return_tensors='tf', padding='max_length', truncation=True, max_length=MAX_LEN)
preds = model(inputs)[0]
label = tf.argmax(preds, axis=1).numpy()[0]
print(f"\nExemplo de inferência: '{sample_text}' => Label: {label} (0=neg, 1=pos)") 
