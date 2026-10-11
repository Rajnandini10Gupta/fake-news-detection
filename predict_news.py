
import re
import joblib

# Load the trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


print("=" * 45)
print("FAKE NEWS DETECTION")
print("=" * 45)

headline = input("\nEnter the news headline: ").strip()
article_body = input(
    "Paste the article body (or press Enter to skip): "
).strip()

if not headline and not article_body:
    print("\nPlease enter a headline or article body.")
else:
    # Match the training format: headline + article body
    combined_text = f"{headline} {article_body}".strip()
    cleaned_news = clean_text(combined_text)

    if not cleaned_news:
        print("\nThe entered text contains no usable English letters.")
    else:
        news_vector = vectorizer.transform([cleaned_news])
        prediction = model.predict(news_vector)[0]
        probabilities = model.predict_proba(news_vector)[0]
        confidence = probabilities.max() * 100

        print("\n" + "=" * 45)

        if prediction == 0:
            print("MODEL PREDICTION: FAKE NEWS")
        else:
            print("MODEL PREDICTION: REAL NEWS")

        print(f"Model confidence score: {confidence:.2f}%")

        print("\nClass scores:")
        for label, probability in zip(
            model.classes_, probabilities
        ):
            name = "Fake News" if label == 0 else "Real News"
            print(f"{name}: {probability * 100:.2f}%")

        print("\nImportant:")
        print("These scores reflect the model's output,")
        print("not the actual probability that a claim is true.")
        print("Verify important claims using reliable sources.")
