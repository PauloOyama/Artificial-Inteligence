import tensorflow as tf### models
import numpy as np### math computations
import matplotlib.pyplot as plt### plotting bar chart
import sklearn### machine learning library
import cv2## image processing
from sklearn.metrics import confusion_matrix, roc_curve### metrics
import seaborn as sns### visualizations
import datetime # For Datetime Functions
import pathlib # handling files and paths on your operating system
import io # dealing with various types of I/O
import os 
import re # for Regular Expressions
import string
import time
from numpy import random
import gensim.downloader as api # to download pre-trained model datasets and word embeddings from Gensim's repository
import tensorflow_datasets as tfds # Tf Datasets
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (Dense,Flatten,InputLayer,BatchNormalization,
                                     Dropout,Input,LayerNormalization)
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy,SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy,TopKCategoricalAccuracy,CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from datasets import load_dataset
from transformers import (BertTokenizerFast,TFBertTokenizer,BertTokenizer,RobertaTokenizerFast,
                          DataCollatorWithPadding,TFRobertaForSequenceClassification,TFBertForSequenceClassification,
                          TFBertModel,create_optimizer)

BATCH_SIZE=8

# Load the amazon review dataset
df = pd.read_csv('../data/IMDB Dataset.csv')

df = df.sample(5000)

dataset = Dataset.from_pandas(df, split='train')
dataset = dataset.train_test_split(test_size=0.3)