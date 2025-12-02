# password_classifier.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

LABELS = {0: "Weak", 1: "Medium", 2: "Strong"}

def load_dataset(csv_path: str = "data.csv") -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")

    df = pd.read_csv(path, on_bad_lines="skip")
    df = df.dropna(subset=["password", "strength"])
    df = df[df["password"].apply(lambda x: isinstance(x, str) and x.strip() != "")]
    df["strength"] = df["strength"].astype(int)
    return df

def train_model(csv_path="data.csv", max_iter=1000, verbose=False):
    df = load_dataset(csv_path)

    # Split once: train + test
    X_train, X_test, y_train, y_test = train_test_split(
        df["password"],
        df["strength"],
        test_size=0.20,
        random_state=42,
        stratify=df["strength"]
    )

    # Build TF-IDF
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=max_iter, n_jobs=1)
    model.fit(X_train_vec, y_train)

    # Evaluate on real test set
    if verbose:
        y_pred = model.predict(X_test_vec)
        print(classification_report(y_test, y_pred, target_names=LABELS.values()))

    return model, vectorizer

'''def predict_strength(password: str, model, vectorizer):
    vec = vectorizer.transform([password])
    pred = int(model.predict(vec)[0])
    return LABELS.get(pred, str(pred))'''


# train_mlp.py and save
import joblib

def main():
    model, vectorizer = train_model(csv_path="data.csv", verbose=True)
    joblib.dump(model, "password_model.pkl")
    joblib.dump(vectorizer, "password_vectorizer.pkl")
    print("Saved model + vectorizer.")

if __name__ == "__main__":
    main()

#need to load model here that was trained in "train_mlp" and test with user input password
#put thru brute force function from "brute_force.py" 
#"IF" mlp says its "WEAK" 
#||
#"IF" brute force says can be cracked under X amount of time
#THEN give to feedback model to give user suggestions

