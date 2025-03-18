import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# NLTK setup
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
port_stem = PorterStemmer()

# Function to clean and preprocess text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [port_stem.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

# Load the trained model and vectorizer
model = pickle.load(open('trained_model.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Streamlit UI
st.title("📝 Twitter Sentiment Analysis")
st.subheader("Enter a tweet to analyze its sentiment")

# User input
user_tweet = st.text_area("Type your tweet here...")

if st.button("Analyze Sentiment"):
    if user_tweet.strip() != "":
        processed_tweet = preprocess_text(user_tweet)
        vectorized_tweet = vectorizer.transform([processed_tweet])
        prediction = model.predict(vectorized_tweet)

        # Display result
        sentiment = "Positive 😊" if prediction[0] == 1 else "Negative 😞"
        st.success(f"Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter a valid tweet!")
