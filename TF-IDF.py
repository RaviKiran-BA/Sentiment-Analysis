import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH   = "Sentiment_Analysis_Dataset.csv"   # <- change
MODEL_PATH  = "tfidf_linsvc.pkl"
SEED        = 42
np.random.seed(SEED)

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
    _USE_SPACY = True
    print("Using spaCy for text processing")
except Exception:                       # fallback to NLTK if spaCy not installed
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")
    
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    _EN_STOP = set(stopwords.words("english"))
    _lemmatizer = WordNetLemmatizer()
    _USE_SPACY = False
    print("Using NLTK for text processing")

class TextCleaner(BaseEstimator, TransformerMixin):
    NEG_WORDS = {"not", "no", "never", "n't", "dont", "don't", "wont", "won't", "cant", "can't"}

    def __init__(self, use_spacy=False):
        self.use_spacy = use_spacy

    def _clean_spacy(self, doc: str) -> str:
        tokens = []
        negate = False
        for token in _nlp(doc.lower()):
            txt = token.text
            if any(neg in txt for neg in self.NEG_WORDS):
                negate = True
                continue
            if token.is_stop or token.is_punct or len(txt) < 2:
                continue
            lemma = token.lemma_
            if negate:
                lemma = f"NEG_{lemma}"
                negate = False
            tokens.append(lemma)
        return " ".join(tokens)

    def _clean_nltk(self, doc: str) -> str:
        tokens = []
        negate = False
        words = word_tokenize(doc.lower())
        for w in words:
            if any(neg in w for neg in self.NEG_WORDS):
                negate = True
                continue
            if not w.isalnum() or w in _EN_STOP or len(w) < 2:
                continue
            lemma = _lemmatizer.lemmatize(w)
            if negate:
                lemma = f"NEG_{lemma}"
                negate = False
            tokens.append(lemma)
        return " ".join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.use_spacy:
            return [self._clean_spacy(str(t)) for t in X]
        else:
            return [self._clean_nltk(str(t)) for t in X]

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

if "text" not in df.columns:
    raise ValueError("CSV must contain 'text' column")
if "sentiment" not in df.columns:
    raise ValueError("CSV must contain 'sentiment' column")

# Remove rows with missing text or sentiment
df = df.dropna(subset=['text', 'sentiment'])
print(f"After removing NaN: {df.shape}")

# Map sentiment labels to numerical values
# Handle different possible sentiment formats
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
elif 0 in unique_sentiments:  # already numeric
    label2id = {0: 0, 1: 1, 2: 2}  # or adjust based on your encoding
else:
    # Create mapping for whatever labels exist
    sorted_sentiments = sorted(unique_sentiments)
    label2id = {sentiment: idx for idx, sentiment in enumerate(sorted_sentiments)}
    
print(f"Label mapping: {label2id}")

X = df["text"].astype(str).values
y = df["sentiment"].map(label2id).values

# Remove any unmapped labels (NaN values)
valid_mask = ~pd.isna(y)
X = X[valid_mask]
y = y[valid_mask].astype(int)

print(f"Final dataset size: {len(X)}")
print(f"Class distribution: {pd.Series(y).value_counts().sort_index().to_dict()}")

# stratified train / test (20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")
print("Train class distribution:", pd.Series(y_train).value_counts().sort_index().to_dict())
print("Test  class distribution:",  pd.Series(y_test).value_counts().sort_index().to_dict())

word_vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1,2),
    max_features=20000,  # Reduced for better performance
    min_df=2,           # Require at least 2 occurrences
    max_df=0.95,
    sublinear_tf=True,
    norm="l2",
)

char_vec = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3,5),
    max_features=10000,  # Added max_features for char n-grams
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    norm="l2",
)

features = FeatureUnion([("word", word_vec), ("char", char_vec)])

# LinearSVC with built‑in class‑weight balancing
clf = LinearSVC(
    class_weight="balanced", 
    C=1.0,  # Start with default C value
    random_state=SEED,
    max_iter=2000  # Increase max iterations
)

pipeline = Pipeline([
    ("cleaner", TextCleaner(use_spacy=_USE_SPACY)),
    ("features", features),
    ("clf", clf)
])

print("\n=== Running 5-fold cross-validation ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

cv_f1 = cross_val_score(
    pipeline, X_train, y_train,  # Use training data for CV
    cv=cv,
    scoring="f1_macro",            # macro‑F1 is the right metric for imbalanced data
    n_jobs=1                        # CPU only (no joblib parallelism on Windows)
)

print("\n=== CV macro‑F1 per fold ===")
print(cv_f1)
print(f"Mean ± std = {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

print("\n=== Training final model ===")
pipeline.fit(X_train, y_train)

print("\n=== Evaluating on test set ===")
y_pred = pipeline.predict(X_test)

# Create id2label mapping
id2label = {v: k for k, v in label2id.items()}
target_names = [id2label[i] for i in sorted(id2label.keys())]

print("\n=== Test set classification report ===")
print(classification_report(
    y_test, y_pred,
    target_names=target_names
))
print("Macro‑F1 on test :", f1_score(y_test, y_pred, average="macro"))

# confusion matrix (nice heatmap)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("TF‑IDF + LinearSVC – Confusion matrix")
plt.tight_layout()
plt.show()

joblib.dump(pipeline, MODEL_PATH)
joblib.dump(id2label, "id2label_mapping.pkl")  # Save the mapping too
print(f"\nPipeline saved to → {MODEL_PATH}")
print(f"Label mapping saved to → id2label_mapping.pkl")

def load_model():
    """Load the trained pipeline and label mapping"""
    try:
        pipeline = joblib.load(MODEL_PATH)
        id2label_map = joblib.load("id2label_mapping.pkl")
        return pipeline, id2label_map
    except FileNotFoundError:
        print("Model files not found. Please train the model first.")
        return None, None

def predict_sentiment(text: str, pipeline=None, id2label_map=None) -> str:
    """
    Fast CPU inference – just call this function.
    Returns sentiment label (e.g., 'negative', 'neutral', 'positive')
    """
    if pipeline is None or id2label_map is None:
        pipeline, id2label_map = load_model()
        if pipeline is None:
            return "unknown"
    
    if not isinstance(text, str) or not text.strip():
        return "neutral"
    
    try:
        pred_id = pipeline.predict([text])[0]
        return id2label_map[pred_id]
    except Exception as e:
        print(f"Prediction error: {e}")
        return "unknown"

if __name__ == "__main__":
    # Load the trained model for inference
    trained_pipeline, trained_id2label = load_model()
    
    if trained_pipeline is not None:
        print("\n=== Interactive Sentiment Prediction ===")
        print("Enter sentences to predict their sentiment (blank to quit)")
        
        while True:
            s = input("\nEnter a sentence: ").strip()
            if not s:
                print("Goodbye!")
                break
            
            sentiment = predict_sentiment(s, trained_pipeline, trained_id2label)
            print(f"Predicted sentiment → {sentiment}")
            
            # Get prediction probabilities if available
            try:
                # For LinearSVC, we can get decision function scores
                decision_scores = trained_pipeline.decision_function([s])[0]
                if len(decision_scores.shape) > 0 and len(decision_scores) > 1:
                    print("Decision scores:", {trained_id2label[i]: score 
                                           for i, score in enumerate(decision_scores)})
            except:
                pass  # Skip if not available
    else:
        print("Please ensure the model is trained first!")

# Test with the specific example you mentioned
print("\n=== Testing the problematic example ===")
test_text = "i am sad"
if 'trained_pipeline' in locals() and trained_pipeline is not None:
    result = predict_sentiment(test_text, trained_pipeline, trained_id2label)
    print(f"Text: '{test_text}' → Predicted sentiment: {result}")
else:
    print("Model not loaded. Train the model first.")
