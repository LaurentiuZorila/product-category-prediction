import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ------------------------
# Functie pentru curatarea textului
# ------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main():
    print("Loading dataset...")

    df = pd.read_csv("data/products.csv")

    # -----------------------------------
    # Curatare text
    # -----------------------------------
    df["clean_title"] = df["Product Title"].astype(str).apply(clean_text)

    # Curatam whitespace din numele coloanelor
    df.columns = df.columns.str.strip()

    # Remove rows where Category Label is missing
    df = df.dropna(subset=["Category Label", "Product Title"])

    # -----------------------------------
    # Pregatire X si y
    # -----------------------------------
    X = df["clean_title"]
    y = df["Category Label"]

    # -----------------------------------
    # Vectorizare TF-IDF
    # -----------------------------------
    print("Vectorizing text...")
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(X)

    # -----------------------------------
    # Train/Test split
    # -----------------------------------
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------------
    # Modele de comparat
    # -----------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200),
        "Linear SVC": LinearSVC()
    }

    best_model = None
    best_acc = 0

    # -----------------------------------
    # Antrenare si evaluare
    # -----------------------------------
    print("Training models...\n")

    for name, model in models.items():
        print(f"Training: {name}")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, zero_division=0))

        if acc > best_acc:
            best_acc = acc
            best_model = model

    print("\n====================================")
    print(f"Best model selected with accuracy: {best_acc:.4f}")
    print("====================================")

    # -----------------------------------
    # Salvare model + TF-IDF
    # -----------------------------------
    print("Saving model...")

    with open("models/final_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    print("Model and vectorizer saved successfully!")


if __name__ == "__main__":
    main()
