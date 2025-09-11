import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model and vectorizer
model = pickle.load(open("artifacts/model.pkl", "rb"))
vectorizer = pickle.load(open("artifacts/vectorizer.pkl", "rb"))

ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()  # convert text to lowercase
    text = nltk.word_tokenize(text)  # tokenize the text

    # remove special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Streamlit UI
st.title("📩 SMS Spam Classifier")

st.write("Enter an SMS below to check if it's Spam or Not Spam.")

# Input from user
input_sms = st.text_area("Enter SMS message here:")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = vectorizer.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Show Result
        if result == 1:
            st.error("🚨 This message is **Spam**!")
        else:
            st.success("✅ This message is **Not Spam**.")
