# 📩 SMS Spam Classifier  

🔗 **[Live Demo on Streamlit](https://spamsms-classifier.streamlit.app/)**  

A Machine Learning project to classify SMS messages as **Spam** or **Not Spam** using NLP and a Multinomial Naive Bayes model.

This project is a **Machine Learning based SMS Spam Classifier** that can automatically detect whether a given SMS is **Spam** or **Ham (Not Spam)**.  
It is built using **Natural Language Processing (NLP)** techniques and deployed with **Streamlit** for an interactive user experience.  

---

## 📌 Purpose of the Project
The main objective of this project is to:
- Detect spam SMS messages to protect users from scams, phishing, and unnecessary ads.  
- Demonstrate a complete **end-to-end ML workflow**: data ingestion → preprocessing → model training → deployment.  
- Provide a simple, user-friendly web app for real-time SMS classification.  

---

## 🛠️ Technologies Used
- **Python** (3.9+)  
- **Streamlit** (for the web app)  
- **Scikit-learn** (ML model: Multinomial Naive Bayes)  
- **NLTK** (text preprocessing: tokenization, stopwords, stemming)  
- **Pandas & NumPy** (data handling)  
- **Matplotlib** (data visualization)  

---

## 🔬 Methods & Approach
1. **Data Ingestion**  
   - Load dataset (`spam.csv`).  
   - Perform cleaning and preprocessing.  

2. **Text Preprocessing**  
   - Lowercasing  
   - Tokenization  
   - Stopword removal  
   - Stemming (Porter Stemmer)  

3. **Feature Engineering**  
   - TF-IDF Vectorization  

4. **Model Training**  
   - Multinomial Naive Bayes classifier  
   - Trained and stored as `model.pkl`  

5. **Deployment**  
   - Streamlit app (`app.py`)  
   - Tabs for **About Project**, **Classifier**, and **Dataset Visualization**  

---

## ⚙️ How to Run Locally  

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/sms-spam-classifier.git
cd sms-spam-classifier
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Mac/Linux
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Run Model Training Pipeline (if needed)
If you want to retrain the model:  
```bash
python -m src.components.data_ingestion
```

This will:
- Load the dataset  
- Preprocess the data  
- Train the model  
- Save `model.pkl` and `vectorizer.pkl` inside the `artifacts/` folder  

### 5. Run the Streamlit App
```bash
streamlit run app.py
```

---

## 🚀 Deployment
The project is deployed on **Streamlit Community Cloud**.  
You can also deploy it to:  
- **Heroku**  
- **Azure Web Apps**  
- **AWS Elastic Beanstalk**  

---

## 📂 Project Structure
```
├── app.py                # Streamlit app entry point
├── artifacts/            # Saved model & vectorizer
│   ├── model.pkl
│   ├── vectorizer.pkl
├── notebook/             # Dataset & experiments
│   ├── data/spam.csv
├── src/                  # ML pipeline components
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── model_trainer.py
│   │   └── preprocessing.py
├── requirements.txt
├── README.md
```

---

## ✨ Features
- 📊 Dataset visualization (spam vs ham distribution)  
- 🔍 Real-time SMS classification  
- 📦 Retrainable pipeline  
- 🌐 Easy deployment via Streamlit  

---

## 👤 Author
**Binuda Dewhan**  
🚀 Passionate about Data Science, Machine Learning & AI  

---
