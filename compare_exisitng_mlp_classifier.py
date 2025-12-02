from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# --------------------------
# Load HF model
# --------------------------
tokenizer = AutoTokenizer.from_pretrained("DunnBC22/codebert-base-Password_Strength_Classifier")
hf_model = AutoModelForSequenceClassification.from_pretrained(
    "DunnBC22/codebert-base-Password_Strength_Classifier"
)
hf_model.eval()

LABELS = ["Weak", "Medium", "Strong"]

# --------------------------
# Load & prepare dataset
# --------------------------
df = pd.read_csv("data.csv", on_bad_lines='skip')

df = df.dropna(subset=["password", "strength"])
df = df.sample(n=10000, random_state=42)
df["strength"] = df["strength"].astype(int)

passwords = df["password"].tolist()
labels = df["strength"].tolist()

# --------------------------
# TF-IDF + Logistic Regression
# --------------------------
tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
X = tfidf.fit_transform(passwords)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\n=== TF-IDF + Logistic Regression Results ===")
print("Overall Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# --------------------------
# HuggingFace Model Accuracy
# --------------------------
def hf_predict(password):
    tokens = tokenizer(password, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = hf_model(**tokens)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return pred


hf_preds = [hf_predict(pw) for pw in passwords]

print("\n=== HuggingFace CodeBERT Model Results ===")
print("Overall Accuracy:", accuracy_score(labels, hf_preds))
print(classification_report(labels, hf_preds))


#test it and compare with ours. if its better use this one