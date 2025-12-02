# main.py
import joblib
#from password_classifier import predict_strength
from brute_force import analyze_password, format_time
#from feedback_model import generate_feedback
from sft_feedback_llm import generate_feedback

CRACK_THRESHOLD = 60 * 60 * 24 * 7  # 1 week

model = joblib.load("password_model.pkl")
vectorizer = joblib.load("password_vectorizer.pkl")


def classify_password(password: str) -> str:
    '''inputs = tokenizer(password, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        label_id = torch.argmax(logits).item()'''
    X = vectorizer.transform([password])
    label_id = model.predict(X)[0]

    labels = ["weak", "medium", "strong"]  # must match your training
    return labels[label_id]


def main():
    est_crack_time = False
    detected_in_list = False

    pwd = input("Enter password: ")

    ml_label = classify_password(pwd)
    time_seconds = analyze_password(pwd)

    print(f"ML Strength: {ml_label}")

    if time_seconds is None:
        print("Detected in common-passwords or names list!!")
        detected_in_list = True
    else:
        print(f"Pswd crack time: {format_time(time_seconds)}")
        if time_seconds < CRACK_THRESHOLD:
            print("Pswd crack time is less than threshold (a week)")
            est_crack_time = True

    if ml_label == "weak" or est_crack_time or detected_in_list:
        print("\nPassword flagged as weak, common, or easy to crack. (Generating feedback...)")

        print(generate_feedback(pwd)) 
    else:
        print("\nPassword is acceptable.")


#load feedback model


if __name__ == "__main__":
    main()
