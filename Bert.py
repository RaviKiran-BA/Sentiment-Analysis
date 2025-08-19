# ------------------------------------------------------------------
#   distilbert_sentiment.py
#   DistilBERT fine‑tuning (CPU only) - FIXED VERSION
# ------------------------------------------------------------------
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------- 0️⃣ Settings -------------------------
DATA_PATH   = "E:\\Sentiment_Analysis_Dataset.csv"
MODEL_DIR   = "distilbert_sentiment_cpu"
SEED        = 42
MAX_LEN     = 128            # most sentences fit comfortably
EPOCHS      = 3
BATCH_SIZE  = 16             # smaller batch for CPU training
LEARNING_RATE = 2e-5         # slightly lower learning rate
set_seed(SEED)

# Force CPU – even if a GPU is present
DEVICE = torch.device("cpu")
print(f"Running on → {DEVICE}")

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------- 1️⃣ Load data -----------------------
print(f"Loading data from: {DATA_PATH}")
try:
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
except UnicodeDecodeError:
    try:
        df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, encoding="latin1")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check if we have the required columns (adjusted for your CSV structure)
if "text" not in df.columns:
    raise ValueError("CSV must contain 'text' column")
if "sentiment" not in df.columns:
    raise ValueError("CSV must contain 'sentiment' column")

# Remove rows with missing text or sentiment
df = df.dropna(subset=['text', 'sentiment'])
print(f"After removing NaN: {df.shape}")

# Map sentiment labels to numerical values
sentiment_counts = df['sentiment'].value_counts()
print(f"Original sentiment distribution: {sentiment_counts}")

# Create mapping based on what we find in the data
unique_sentiments = df['sentiment'].unique()
print(f"Unique sentiments: {unique_sentiments}")

# Flexible mapping for different sentiment formats
label2id = {}
if 'negative' in unique_sentiments:
    label2id = {"negative": 0, "neutral": 1, "positive": 2}
elif 'neg' in unique_sentiments:
    label2id = {"neg": 0, "neutral": 1, "pos": 2}
elif any(str(sentiment).lower() == 'negative' for sentiment in unique_sentiments):
    # Handle case variations
    for sentiment in unique_sentiments:
        if str(sentiment).lower() == 'negative':
            label2id[sentiment] = 0
        elif str(sentiment).lower() == 'neutral':
            label2id[sentiment] = 1
        elif str(sentiment).lower() == 'positive':
            label2id[sentiment] = 2
else:
    # Create mapping for whatever labels exist (alphabetical order)
    sorted_sentiments = sorted(unique_sentiments)
    label2id = {sentiment: idx for idx, sentiment in enumerate(sorted_sentiments)}

print(f"Label mapping: {label2id}")
id2label = {v: k for k, v in label2id.items()}
print(f"ID to label mapping: {id2label}")

# Convert to lists for processing
X = df["text"].astype(str).tolist()
y_mapped = df["sentiment"].map(label2id)

# Remove any unmapped labels (NaN values)
valid_mask = ~pd.isna(y_mapped)
X = [X[i] for i in range(len(X)) if valid_mask.iloc[i]]
y = y_mapped.dropna().astype(int).tolist()

print(f"Final dataset size: {len(X)}")
print(f"Class distribution: {pd.Series(y).value_counts().sort_index().to_dict()}")

# Ensure we have at least 3 classes
if len(set(y)) < 2:
    raise ValueError(f"Need at least 2 classes for classification, found: {len(set(y))}")

# same stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

print(f"\nTrain: {len(X_train)}   Test: {len(X_test)}")
print("Train class distribution:", pd.Series(y_train).value_counts().sort_index().to_dict())
print("Test class distribution:", pd.Series(y_test).value_counts().sort_index().to_dict())

# -------------------------- 2️⃣ Tokeniser -----------------------
print("\nLoading DistilBERT tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def encode_batch(texts):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )

# Test tokenization
print("Testing tokenization...")
sample_text = X_train[0] if X_train else "This is a test"
sample_encoding = tokenizer(sample_text, return_tensors="pt", max_length=MAX_LEN, truncation=True, padding="max_length")
print(f"Sample text length: {len(sample_text)}")
print(f"Tokenized length: {sample_encoding['input_ids'].shape}")

# -------------------------- 3️⃣ Dataset class -------------------
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

print("\nCreating datasets...")
train_dataset = SentimentDataset(X_train, y_train)
val_dataset   = SentimentDataset(X_test,  y_test)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Test dataset
sample_item = train_dataset[0]
print(f"Sample item keys: {sample_item.keys()}")
print(f"Sample input_ids shape: {sample_item['input_ids'].shape}")
print(f"Sample label: {sample_item['labels']}")

# -------------------------- 4️⃣ Model ---------------------------
print("\nLoading DistilBERT model...")
num_labels = len(set(y))
print(f"Number of labels: {num_labels}")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels,
    problem_type="single_label_classification"
).to(DEVICE)

print(f"Model loaded on device: {next(model.parameters()).device}")

# -------------------------- 5️⃣ TrainingArguments -------------
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir=f"{MODEL_DIR}/logs",
    logging_steps=50,
    seed=SEED,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_macro_f1",
    greater_is_better=True,
    use_cpu=True,
    dataloader_num_workers=0,  # Important for CPU training
)

# -------------------------- 6️⃣ Metrics -------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    macro_f1 = f1_score(labels, preds, average="macro")
    accuracy = accuracy_score(labels, preds)
    
    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy
    }

# -------------------------- 7️⃣ Trainer ------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# -------------------------- 8️⃣ Train ---------------------------
print("\n=== Training DistilBERT (CPU) ===")
print("This may take a while on CPU...")

try:
    trainer.train()
    print("Training completed successfully!")
except Exception as e:
    print(f"Training error: {e}")
    # Continue with evaluation if training partially completed

# -------------------------- 9️⃣ Final evaluation ----------------
print("\n=== Evaluating model ===")
try:
    metrics = trainer.evaluate()
    print(f"\n=== Test set results ===")
    print(f"Accuracy: {metrics.get('eval_accuracy', 'N/A'):.4f}")
    print(f"Macro F1: {metrics.get('eval_macro_f1', 'N/A'):.4f}")
except Exception as e:
    print(f"Evaluation error: {e}")

# Manual evaluation for detailed report
print("\n=== Generating detailed classification report ===")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Print classification report
target_names = [id2label[i] for i in sorted(id2label.keys())]
print("\n=== Test set classification report ===")
print(classification_report(all_labels, all_preds, target_names=target_names))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("DistilBERT Sentiment Analysis - Confusion Matrix")
plt.tight_layout()
plt.show()

# -------------------------- 10️⃣ Save model & tokenizer ----------
print(f"\n=== Saving model ===")
try:
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    
    # Save the label mappings
    import json
    with open(f"{MODEL_DIR}/label_mapping.json", "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f)
    
    print(f"Model, tokenizer, and mappings saved to → {MODEL_DIR}")
except Exception as e:
    print(f"Error saving model: {e}")

# -------------------------- 11️⃣ Inference helper -------------
def load_sentiment_model(model_dir=MODEL_DIR):
    """Loads the fine‑tuned DistilBERT model (CPU only)."""
    try:
        import json
        
        # Load label mappings
        with open(f"{model_dir}/label_mapping.json", "r") as f:
            mappings = json.load(f)
        
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        model.to(torch.device("cpu"))
        
        return tokenizer, model, mappings["id2label"]
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

# -------------------------- 12️⃣ Prediction function -------------
def predict_sentiment_batch(texts, tokenizer_model_tuple=None):
    """Predict sentiment for a batch of texts"""
    if tokenizer_model_tuple is None:
        tokenizer_model_tuple = load_sentiment_model()
    
    tokenizer, model, id2label_map = tokenizer_model_tuple
    if model is None:
        return ["unknown"] * len(texts)
    
    predictions = []
    model.eval()
    
    with torch.no_grad():
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                predictions.append("neutral")
                continue
                
            encoding = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            
            outputs = model(**encoding)
            logits = outputs.logits
            pred_id = int(torch.argmax(logits, dim=-1).cpu().numpy()[0])
            predictions.append(id2label_map[str(pred_id)])
    
    return predictions

def predict_sentiment(text: str, tokenizer_model_tuple=None) -> str:
    """Return sentiment prediction for a single text."""
    return predict_sentiment_batch([text], tokenizer_model_tuple)[0]

# -------------------------- 13️⃣ Interactive loop ---------------
if __name__ == "__main__":
    print("\n=== Loading model for inference ===")
    model_tuple = load_sentiment_model()
    
    if model_tuple[0] is not None:
        print("Model loaded successfully!")
        print("\n=== Testing with your examples ===")
        
        test_cases = [
            "i am sad",
            "i am good", 
            "i am happy",
            "i am ok",
            "I love this movie!",
            "This is terrible",
            "It's okay I guess"
        ]
        
        for text in test_cases:
            prediction = predict_sentiment(text, model_tuple)
            print(f"'{text}' → {prediction}")
        
        print("\n=== Interactive mode ===")
        print("Enter sentences to predict their sentiment (blank to quit)")
        
        while True:
            s = input("\nEnter a sentence: ").strip()
            if not s:
                print("Goodbye!")
                break
            
            try:
                prediction = predict_sentiment(s, model_tuple)
                print(f"Predicted sentiment → {prediction}")
            except Exception as e:
                print(f"Prediction error: {e}")
    else:
        print("Could not load model. Please ensure training completed successfully.")