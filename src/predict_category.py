import pickle
import re


# ----------------------------
# Functie pentru curatarea textului
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main():
    print("Loading model...")

    # Incarcam modelul final
    with open("models/final_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Incarcam vectorizatorul TF-IDF
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

    print("Model loaded. Ready to predict!\n")

    while True:
        title = input("Enter product title (or type 'exit' to stop): ")

        if title.lower() == "exit":
            print("Exiting prediction system.")
            break

        clean = clean_text(title)
        vector = tfidf.transform([clean])
        prediction = model.predict(vector)[0]

        print(f"Predicted category: {prediction}\n")


if __name__ == "__main__":
    main()
