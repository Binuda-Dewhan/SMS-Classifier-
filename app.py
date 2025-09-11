import streamlit as st
import pickle
import nltk
import string
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ----------------- NLTK setup -----------------
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ----------------- Load model + vectorizer -----------------
model = pickle.load(open("artifacts/model.pkl", "rb"))
vectorizer = pickle.load(open("artifacts/vectorizer.pkl", "rb"))

ps = PorterStemmer()

# ----------------- Preprocess Function -----------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# ----------------- App Layout -----------------
st.title("📩 SMS Spam Classifier")

# ✅ Tabs (Rearranged)
tab1, tab2, tab3 = st.tabs(["ℹ️ About Project", "🔍 Classifier", "📊 Dataset"])

# ----------------- Tab 1: About Project -----------------
with tab1:
    st.header("About This Project")
    st.image(
        "https://www.nyckel.com/assets/images/functions/emailspam.webp",
        caption="Machine Learning for Spam Detection",
        use_container_width=True
    )

    st.markdown(
        """
        **Project Overview**  
        This SMS Spam Classifier is built using **Natural Language Processing (NLP)** and a 
        **Multinomial Naive Bayes model**.  

        - **Purpose**: Automatically detect spam SMS messages to protect users.  
        - **Dataset**: SMS Spam Collection dataset.  
        - **Pipeline**:  
            1. Text preprocessing (tokenization, stopword removal, stemming)  
            2. Feature extraction with **TF-IDF**  
            3. Classification using **Multinomial Naive Bayes**  

        ---
        💡 *Developed by Binuda Dewhan*  
        🚀 Deployed using **Streamlit Community Cloud**
        """
    )

# ----------------- Tab 2: Classifier -----------------
with tab2:
    st.header("Detect Spam Messages")

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

# ----------------- Tab 3: Dataset -----------------
with tab3:
    st.header("Dataset Preview & Visualization")
    try:
        df = pd.read_csv("notebook/data/spam.csv", encoding="ISO-8859-1")[["v1", "v2"]]
        df = df.rename(columns={"v1": "label", "v2": "message"})

        st.write("### Sample Data")
        st.dataframe(df.sample(10))

        st.write("### Spam vs Ham Distribution")
        label_counts = df["label"].value_counts()

        fig, ax = plt.subplots()
        ax.pie(label_counts, labels=label_counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

    except FileNotFoundError:
        st.warning("⚠️ Dataset file not found. Please place it in `notebook/data/spam.csv`.")
