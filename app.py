import streamlit as st
import pickle
import nltk
import string
import pandas as pd
import matplotlib.pyplot as plt

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model and vectorizer
model = pickle.load(open("artifacts/model.pkl", "rb"))
vectorizer = pickle.load(open("artifacts/vectorizer.pkl", "rb"))

ps = PorterStemmer()

# ----------------- Styling -----------------
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1581090700227-4c4dcbd6d1e0");
            background-attachment: fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()

# ----------------- Preprocess Function -----------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# ----------------- Main App -----------------
st.title("📩 SMS Spam Classifier")
st.markdown("### Detect spam messages with Machine Learning")

# Project description
st.markdown(
    """
    **Project Overview**  
    This SMS Spam Classifier is built using **Natural Language Processing (NLP)** and a 
    **Multinomial Naive Bayes model**.  

    - **Purpose**: To automatically detect spam SMS messages and protect users from scams.  
    - **Dataset**: The model was trained on the popular **SMS Spam Collection dataset**.  
    - **Pipeline**:  
        1. Preprocessing (lowercasing, tokenization, stopword removal, stemming)  
        2. Feature extraction using **TF-IDF Vectorizer**  
        3. Classification using **Multinomial Naive Bayes**  

    ---
    """
)

# Input Section
input_sms = st.text_area("✉️ Enter SMS message here:")

if st.button('🔍 Predict'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = vectorizer.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 This message is **Spam**!")
        else:
            st.success("✅ This message is **Not Spam**.")

# ----------------- Dataset Section -----------------
st.markdown("## 📊 Dataset Preview & Visualization")

try:
    df = pd.read_csv("notebook/data/spam.csv", encoding="ISO-8859-1")[["v1", "v2"]]
    df = df.rename(columns={"v1": "label", "v2": "message"})

    st.write("### Sample Data")
    st.dataframe(df.sample(10))  # show random 10 rows

    # Label distribution
    st.write("### Spam vs Ham Distribution")
    label_counts = df["label"].value_counts()

    fig, ax = plt.subplots()
    ax.pie(label_counts, labels=label_counts.index, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

except FileNotFoundError:
    st.warning("⚠️ Dataset file not found. Please place it in `notebook/data/spam.csv`.")

# ----------------- Footer -----------------
st.markdown(
    """
    ---
    💡 *Developed by Binuda Dewhan*  
    🚀 Deployed using **Streamlit**  
    """
)
