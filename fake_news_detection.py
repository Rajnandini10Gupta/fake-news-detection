
import re
import warnings
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

warnings.filterwarnings("ignore")

print("=" * 60)
print("FAKE NEWS DETECTION - MODEL TRAINING")
print("=" * 60)

# 1. Load the datasets
print("\nLoading datasets...")

fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

fake["label"] = 0
real["label"] = 1

# 2. Prepare the text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

for df in (fake, real):
    for col in ("title", "text"):
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' was not found."
            )
        df[col] = df[col].fillna("").astype(str)

fake["combined_text"] = (
    fake["title"] + " " + fake["text"]
).map(clean_text)

real["combined_text"] = (
    real["title"] + " " + real["text"]
).map(clean_text)

data = pd.concat(
    [
        fake[["title", "text", "combined_text", "label"]],
        real[["title", "text", "combined_text", "label"]]
    ],
    ignore_index=True
)

# Remove empty articles and exact duplicate text
data = data[data["combined_text"].str.len() > 0]
data = data.drop_duplicates(
    subset=["combined_text"]
).reset_index(drop=True)

# 3. Remove identical text appearing under conflicting labels
label_counts = data.groupby("combined_text")["label"].nunique()
conflicting_texts = label_counts[label_counts > 1].index

data = data[
    ~data["combined_text"].isin(conflicting_texts)
].reset_index(drop=True)

print(f"\nConflicting articles removed: {len(conflicting_texts)}")
print(f"Articles available: {len(data)}")
print("\nLabel counts (0 = Fake, 1 = Real):")
print(data["label"].value_counts().sort_index())

if data["label"].nunique() != 2:
    raise ValueError("Both fake and real articles are required.")

# 4. Split the data
# A stratified split keeps both classes represented in each set.
X_train, X_test, y_train, y_test = train_test_split(
    data["combined_text"],
    data["label"],
    test_size=0.20,
    random_state=42,
    stratify=data["label"]
)

print(f"\nTraining articles: {len(X_train)}")
print(f"Testing articles: {len(X_test)}")

# 5. Convert text into numerical features
print("\nConverting text with TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6. Train the model
print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)

# 7. Evaluate the model on held-out articles
predictions = model.predict(X_test_tfidf)
probabilities = model.predict_proba(X_test_tfidf)[:, 1]

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\nAccuracy: {accuracy_score(y_test, predictions):.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Fake News", "Real News"],
        zero_division=0
    )
)

print("Confusion matrix:")
print("Rows = actual labels; columns = predicted labels")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print(
    "\nROC-AUC:",
    f"{roc_auc_score(y_test, probabilities):.4f}"
)

# 8. Save the model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel saved: fake_news_model.pkl")
print("Vectorizer saved: tfidf_vectorizer.pkl")

print("\nTraining completed.")
print(
    "IMPORTANT: High test accuracy does not prove that "
    "current news can be fact-checked reliably."
)
