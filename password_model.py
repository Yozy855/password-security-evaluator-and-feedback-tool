import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# --- Step 1: Create a small mock dataset ---
"""data = 
{
    'password': [
        'password123', 'P@ssw0rd!', '123456', 'letmein', 'sunshine',
        'Qwerty2024', 'Myp@ssword2025', 'abcABC123!', 'h3lloworld',
        'Xy!94zaQ', '!!Aa123Bb', 'k!M9x#12ghT'
    ],
    'strength': [
        0, 0, 0, 0, 0,   # weak
        1, 1, 1,          # medium
        2, 2, 2, 2        # strong
    ]
}"""


# --- Step 1: Load dataset ---
import pandas as pd

df = pd.read_csv("data.csv", on_bad_lines='skip')

# Remove any rows where the password is missing or not a string
df = df.dropna(subset=['password'])
df = df[df['password'].apply(lambda x: isinstance(x, str) and x.strip() != '')]

print(df.head())
print(df['strength'].value_counts())

# --- Step 2: Convert passwords into numerical features ---
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,4))
X = vectorizer.fit_transform(df['password'])
y = df['strength']

# --- Step 3: Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 4: Train a simple model ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- Step 5: Evaluate the model ---
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# --- Step 6: Try predictions ---
def predict_strength(pw):
    vec = vectorizer.transform([pw])
    pred = model.predict(vec)[0]
    label = {0: "Weak", 1: "Medium", 2: "Strong"}[pred]
    print(f"Password: {pw}  →  Predicted Strength: {label}")

print("\n--- Test Predictions ---")
predict_strength("password123")
predict_strength("Summer2025!")
predict_strength("Tg!93xQ#zA")
