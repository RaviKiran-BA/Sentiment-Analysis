# Sentiment Analysis

# TF-IDF with LinearSVC Pipeline - 68.70% Accuracy

# DistilBERT - 79.29% Accuracy

## Workflow of the Project:  

1. Dataset Collection.
2. Data Preprocessing:  
   - Reads dataset with encoding fallbacks (utf-8, ISO-8859-1, latin1).  
   - Drops missing values.  
   - Automatically maps labels → IDs: {"negative": 0, "neutral": 1, "positive": 2} (or dataset-specific).  
   - Splits data into 80% train / 20% test (stratified).  
   - Tokenizes text with DistilBERT tokenizer: max_length=128 and Padding & truncation applied.  
3. Dataloader:  
   - Implements a custom SentimentDataset: Returns input_ids, attention_mask, labels.  
   - Wraps train/test splits into HuggingFace Trainer-compatible datasets.  
4. Model Creation:  
   - Loads DistilBERT base uncased: DistilBertForSequenceClassification, problem_type="single_label_classification" and num_labels = number of unique classes.  
5. Training:  
   Arguments:  
   - Epochs: 3  
   - Batch size: 16 (CPU-friendly)  
   - Learning rate: 2e-5  
   - Evaluation + checkpoint saving at every epoch  
   - Early stopping (patience=2)  
   Optimizer & Scheduler handled internally by HuggingFace Trainer.  
   Metrics logged:  
   - Accuracy  
   - Macro-F1  
6. Evaluation:  
   - Evaluates on the held-out test set.  
   - Prints:  
   Overall Accuracy  
   Macro-F1 Score  
   Detailed classification_report (precision/recall/F1 per class).  
   - Generates confusion matrix heatmap with Seaborn.
7. Model Saving: Saves to distilbert_sentiment_cpu/.
8. Output: Console logs: Training & evaluation metrics.
9. Usage: python sentiment_analysis.py  
