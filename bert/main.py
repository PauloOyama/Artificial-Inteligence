import tensorflow as tf
# import numpy as np### math computations
# import matplotlib.pyplot as plt### plotting bar chart
# import sklearn### machine learning library
# import cv2## image processing
# from sklearn.metrics import confusion_matrix, roc_curve### metrics
# import seaborn as sns### visualizations
import datetime # For Datetime Functions
import pathlib # handling files and paths on your operating system
import io # dealing with various types of I/O
import os 
import re # for Regular Expressions
import string
import time
import pandas as pd
from numpy import random
from datasets import Dataset, DatasetDict,NamedSplit
from transformers import (BertTokenizerFast,TFBertTokenizer,BertTokenizer,RobertaTokenizerFast,
                          DataCollatorWithPadding,TFRobertaForSequenceClassification,TFBertForSequenceClassification,
                          TFBertModel,create_optimizer)
import matplotlib.pyplot as plt
import sys

BATCH_SIZE= 8 
NUM_EPOCHS= 10 
SAMPLE_SIZE = 50000

print(sys.argv)
if len(sys.argv)> 4 : 
    print("Too Much Paramenters")
    sys.exit()
elif len(sys.argv) < 4 :
    print("Too Few Paramenters")
    print('python <file>.py <BATCH_SIZE> <NUM_EPOCHS> <SAMPLE_SIZE>')
    sys.exit()
else:
    if (int(sys.argv[1]) % 8) != 0 :
        print("Batch Size needs to be power of 8")
        sys.exit()
    elif (int(sys.argv[2])) == 0 :
        print("Number of Epochs needs to be greater than 0")
        sys.exit()
    elif (int(sys.argv[3]) ) == 0 :
        print("Number of Sample Size needs to be greater than 0")
        sys.exit()

BATCH_SIZE = int(sys.argv[1])
NUM_EPOCHS = int(sys.argv[2])
SAMPLE_SIZE = int(sys.argv[3])


##FUNCTION##
def preprocess_function(examples):
  return tokenizer(examples["review"],padding=True,truncation=True,)

def swap_positions(dataset):
  return {'input_ids':dataset['input_ids'],
          'token_type_ids':dataset['token_type_ids'],
          'attention_mask':dataset['attention_mask'],},dataset['sentiment']

## MAIN ###
print("INITIATING WITH...")
print("BATCH SIZE = ",BATCH_SIZE)
print("NUM_EPOCHS = ",NUM_EPOCHS)
print("SAMPLE_SIZE = ",SAMPLE_SIZE)

print("Reading Dataset")
df = pd.read_csv('../data/IMDB Dataset.csv')


df = df.sample(SAMPLE_SIZE)
new_values = {"negative":0,"positive":1 }
df['sentiment'] = df['sentiment'].replace(new_values)

dataset = Dataset.from_pandas(df, split='train',preserve_index=False)
dataset = dataset.train_test_split(test_size=0.7)

print("Loading Model")
model_id="bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_id)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

print("Tokenizing")
tf_train_dataset = tokenized_dataset["train"].to_tf_dataset(
    columns=['input_ids', 'token_type_ids', 'attention_mask', 'sentiment'],
    shuffle=True,
    batch_size=BATCH_SIZE,
)

tf_val_dataset = tokenized_dataset["test"].to_tf_dataset(
    columns=['input_ids', 'token_type_ids', 'attention_mask', 'sentiment'],
    shuffle=True,
    batch_size=BATCH_SIZE,
)

tf_train_dataset=tf_train_dataset.map(swap_positions).prefetch(tf.data.AUTOTUNE)
tf_val_dataset=tf_val_dataset.map(swap_positions).prefetch(tf.data.AUTOTUNE)

model=TFBertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=1)


batches_per_epoch = len(tokenized_dataset["train"]) // BATCH_SIZE
total_train_steps = int(batches_per_epoch * NUM_EPOCHS)

optimizer, schedule = create_optimizer(init_lr=2e-5,num_warmup_steps=0, num_train_steps=total_train_steps)

print("Compile Model")
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=optimizer,
    metrics=['accuracy'],)

print("Fit Model")
history=model.fit(
    tf_train_dataset,
    validation_data=tf_val_dataset,
    epochs=NUM_EPOCHS)

print("Saving Model")
model.save_pretrained("./Bert_50000", from_pt=True) 


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("Model Accuracy")
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("Model Loss")
plt.show()