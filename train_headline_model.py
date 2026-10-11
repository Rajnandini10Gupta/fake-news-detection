
import re
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

print("=" * 60)
print("HEADLINE-ONLY FAKE NEWS MODEL")
print("=" * 60)

# 1. Load the datasets
print("\nLoading datasets...")

fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

if "title" not in fake.columns or "title" not in real.columns:
    raise ValueError("Both datasets must contain a 'title' column.")

fake = fake[["title"]].copy()
real = real[["title"]].copy()

fake["label"] = 0
real["label"] = 1

# 2. Clean headlines
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

fake["title"] = fake["title"].fillna("").astype(str)
real["title"] = real["title"].fillna("").astype(str)

data = pd.concat([fake, real], ignore_index=True)
data["clean_title"] = data["title"].map(clean_text)

# Remove empty headlines
data = data[data["clean_title"].str.len() > 0].copy()

# Remove identical headlines that have conflicting labels
label_counts = data.groupby("clean_title")["label"].nunique()
conflicts = label_counts[label_counts > 1].index

print(f"Conflicting headlines removed: {len(conflicts)}")

data = data[
    ~data["clean_title"].isin(conflicts)
].copy()

# Remove duplicate headlines
data = data.drop_duplicates(
    subset=["clean_title"]
).reset_index(drop=True)

print(f"Total headlines: {len(data)}")
print("\nClass counts (0 = Fake, 1 = Real):")
print(data["label"].value_counts().sort_index())

if data["label"].nunique() != 2:
    raise ValueError("Both fake and real headlines are required.")

# 3. Split headlines into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data["clean_title"],
    data["label"],
    test_size=0.20,
    random_state=42,
    stratify=data["label"]
)

print(f"\nTraining headlines: {len(X_train)}")
print(f"Testing headlines: {len(X_test)}")

# 4. Convert headlines into numerical features
print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train the headline-only model
print("\nTraining model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)

# 6. Evaluate on the held-out test set
predictions = model.predict(X_test_tfidf)
probabilities = model.predict_proba(X_test_tfidf)[:, 1]

print("\n" + "=" * 60)
print("HEADLINE MODEL EVALUATION")
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

print("\nConfusion matrix:")
print("Rows = actual; columns = predicted")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print(f"\nROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")

# 7. Save separate files; preserve the existing model
joblib.dump(model, "headline_news_model.pkl")
joblib.dump(vectorizer, "headline_tfidf_vectorizer.pkl")

print("\nSaved headline_news_model.pkl")
print("Saved headline_tfidf_vectorizer.pkl")
print("\nHeadline model training completed.")
print(
    "Reminder: dataset test scores do not establish the truth "
    "of a new or current headline."
)
