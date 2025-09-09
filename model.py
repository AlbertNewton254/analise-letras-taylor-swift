import re
import spacy
import joblib
import sys

model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

word_regex = r'[^a-zA-Z]'
break_regex = r'[\r\n]'

nlp = spacy.load("en_core_web_sm")

def preprocess(x):
    x = re.sub(word_regex, ' ', x)
    x = re.sub(break_regex, ' ', x)
    x = x.lower()
    doc = nlp(x)
    tokens = [token.lemma_ for token in doc 
              if not token.is_stop and not token.is_punct]
    xProcessed = ' '.join(tokens)
    return xProcessed

def predict_single(text):
    processed = preprocess(text)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)
    return pred[0]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])  # Take all arguments as input text
    else:
        user_input = input("Enter a lyric or phrase to classify: ")
    result = predict_single(user_input)
    print(f"Predicted class: {result}")