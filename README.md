
# Fake News Detection System

A machine learning project that classifies news headlines as Fake News or Real News using Natural Language Processing (NLP) and TF-IDF.

## Features
- Headline-based fake/real news classification
- Machine learning model using Logistic Regression
- TF-IDF text feature extraction
- Related news search using Google News RSS
- Official-source search for additional evidence
- Streamlit web application

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit
- Requests

## Model Evaluation
The headline model achieved approximately 93.80% accuracy and a 0.9850 ROC-AUC score on its held-out test set.

These metrics reflect performance on the dataset used for evaluation. They do not guarantee accuracy on current or unseen news.

## Important Limitation
The model's prediction is not proof that a news claim is true or false. Its scores are model outputs, not verified probabilities. Always check reliable sources and whether the evidence directly supports the claim.

## How to Run

1. Install Python.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   python -m streamlit run app.py
   ```

## Model Files
The trained model and TF-IDF vectorizer are required to run the app. Keep these files in the same project folder as `app.py`.

## Future Improvements
- Evaluate performance on independent, recent news
- Improve source verification and evidence matching
- Add more robust model evaluation
- Deploy the application online

## Disclaimer
This project is for educational purposes and should not be used as the sole basis for determining whether news is genuine.
