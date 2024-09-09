**Name**: Ravi Kiran B A\
**Company**: CODTECH IT SOLUTIONS\
**ID**: CT08DS2280\
**Domain**: Artificial Intelligence\
**Duration**: August to October 2024

## Overview of the project

Project: Sentiment Analysis Using NLP

## Objective
The goal of this project is to build and evaluate a sentiment analysis model using Natural Language Processing (NLP) techniques on the "Sentiment_Analysis_Dataset". The model is trained to classify text into three sentiment categories: Negative, Neutral, and Positive. The project involves preprocessing text data, building a classification pipeline with TF-IDF vectorization and Logistic Regression, and evaluating the model’s performance through cross-validation, hyperparameter tuning, and various visualizations.

## Key Activities
1. **Data Preparation**:
    - Loaded and inspected the sentiment analysis dataset.
    - Preprocessed text data by:
        - Tokenizing
        - Lemmatizing
        - Removing stopwords

2. **Model Building**:
    - Created a machine learning pipeline using:
        - TF-IDF Vectorizer
        - Logistic Regression
    - Conducted cross-validation to assess initial performance.

3. **Hyperparameter Tuning**:
    - Used GridSearchCV to optimize hyperparameters.

4. **Model Evaluation**:
    - Evaluated using:
        - Accuracy
        - Classification report
        - Confusion matrix
    - Plotted:
        - Confusion matrix
        - Sentiment distribution of predictions

5. **Model Saving**:
    - Saved the trained model and TF-IDF Vectorizer using `pickle`.

6. **User Interface**:
    - Implemented a function to predict sentiment for user-provided text input.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: 
    - `pandas` for data manipulation
    - `numpy` for numerical operations
    - `nltk` for natural language processing
    - `matplotlib` and `seaborn` for visualization
    - `scikit-learn` for machine learning
    - `pickle` for saving and loading models
      
## Output
### Confusion Matrix
![Confusion_Matrix](https://github.com/user-attachments/assets/b5071992-43f6-4b22-94c0-177ee94ffd27)

### Sentiment Distribution in Predictions
![Sentiment_Distribution_in_Predictions](https://github.com/user-attachments/assets/b3694afb-8648-41f0-896b-002e45e56993)

## Conclusion
This project demonstrates the application of NLP techniques for sentiment analysis. The model achieved good performance on the validation set, and various evaluation metrics were used to assess its effectiveness. The visualizations provide insights into the model's predictions and its ability to classify text accurately. The trained model and vectorizer have been saved for future use or further experimentation.
