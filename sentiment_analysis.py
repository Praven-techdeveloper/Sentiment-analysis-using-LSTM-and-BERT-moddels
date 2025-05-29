import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

# =====================
# Install the required libraries if not already installed.
# Uncomment and run the following lines if needed:
# !pip install tensorflow transformers datasets torch matplotlib seaborn
# =====================

# ---------------------
# 1. Import Necessary Libraries
# ---------------------
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from datasets import load_dataset

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ---------------------
# 2. Load and Prepare the IMDB Dataset
# ---------------------
# Load the IMDB dataset from Hugging Face
dataset = load_dataset("imdb")
df_full = pd.DataFrame(dataset["train"])

# Create a dummy "movie_name" column. (The original dataset only has "text" and "label")
df_full["movie_name"] = "Movie " + (df_full.index + 1).astype(str)

# Keep only the desired columns: movie_name, text (review), and label (sentiment)
df_full = df_full[["movie_name", "text", "label"]]

# For demonstration, sample a subset (e.g., 2000 examples) to keep training time reasonable.
df = df_full.sample(n=2000, random_state=42).reset_index(drop=True)

print("Dataset sample:")
print(df.head())

# ---------------------
# 3. LSTM Model Implementation with TensorFlow
# ---------------------
# Data preprocessing for LSTM model
MAX_VOCAB = 10000
SEQ_LEN = 200

tokenizer_tf = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer_tf.fit_on_texts(df["text"])
sequences = tokenizer_tf.texts_to_sequences(df["text"])
padded_sequences = pad_sequences(sequences, maxlen=SEQ_LEN, padding="post", truncating="post")
labels_tf = np.array(df["label"])  # 0 for negative, 1 for positive

# Build the LSTM model
lstm_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(MAX_VOCAB, 128, input_length=SEQ_LEN),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

lstm_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("\nTraining LSTM Model...")
lstm_model.fit(padded_sequences, labels_tf, epochs=3, batch_size=32, verbose=1)
print("LSTM Model Training Complete!\n")

# ---------------------
# 4. BERT Model Implementation with PyTorch
# ---------------------
# Create a PyTorch Dataset for the BERT model
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = str(self.texts[index])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(), 
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[index], dtype=torch.long)
        }

# Initialize the BERT tokenizer and dataset
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_LENGTH = 128  # maximum sequence length for BERT

dataset_bert = IMDBDataset(
    texts=df["text"].tolist(),
    labels=df["label"].tolist(),
    tokenizer=bert_tokenizer,
    max_length=MAX_LENGTH
)

# Create a DataLoader for the BERT dataset
BATCH_SIZE = 16
dataloader_bert = DataLoader(dataset_bert, batch_size=BATCH_SIZE, shuffle=True)

# Set up BERT model for sequence classification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model_bert = model_bert.to(device)

# Set up the optimizer and learning rate scheduler
optimizer = AdamW(model_bert.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(dataloader_bert) * 3  # training for 3 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop for BERT
print("Training BERT Model...")
model_bert.train()
for epoch in range(3):
    print(f"Epoch {epoch+1}/3")
    total_loss = 0
    for batch in tqdm(dataloader_bert, desc="BERT Training"):
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch = batch["labels"].to(device)
        
        outputs = model_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_batch
        )
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    avg_loss = total_loss / len(dataloader_bert)
    print(f"Average Loss: {avg_loss:.4f}")
    
print("BERT Model Training Complete!\n")

# ---------------------
# 5. Evaluate and Visualize Sentiment Predictions from Both Models
# ---------------------
# LSTM predictions (using TensorFlow model)
lstm_preds = lstm_model.predict(padded_sequences).flatten()

# BERT predictions (using PyTorch model)
model_bert.eval()
bert_preds = []
with torch.no_grad():
    # We use a DataLoader over the whole dataset
    dataloader_eval = DataLoader(dataset_bert, batch_size=BATCH_SIZE)
    for batch in dataloader_eval:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model_bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # Apply softmax to get probabilities; take probability for positive sentiment (index 1)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        bert_preds.extend(probs)
bert_preds = np.array(bert_preds)

# Visualization: Histogram comparing predicted sentiment scores (0 for negative, 1 for positive)
plt.figure(figsize=(10, 5))
sns.histplot(lstm_preds, color="blue", kde=True, label="LSTM Predictions", bins=30, stat="density", alpha=0.6)
sns.histplot(bert_preds, color="green", kde=True, label="BERT Predictions", bins=30, stat="density", alpha=0.6)
plt.xlabel("Sentiment Score (0=Negative, 1=Positive)")
plt.ylabel("Density")
plt.title("Sentiment Analysis Prediction Distribution")
plt.legend()
plt.show()
