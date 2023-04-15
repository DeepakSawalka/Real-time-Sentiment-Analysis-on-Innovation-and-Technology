import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import streamlit as st
import warnings
import re
import emoji
import pandas as pd

warnings.filterwarnings('ignore')


st.markdown("<h1 style='text-align: center;'>Real-Time Sentiment Analysis</h1>", unsafe_allow_html=True)


# Initialize a TfidfVectorizer object
vect = TfidfVectorizer()

# Load your trained model
model = joblib.load('lr_model.joblib')

# Load the vocabulary and idf values from the trained TfidfVectorizer
vect= joblib.load('vectorizer.joblib')


def preprocess_text(text):
    # Remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])

    # Tokenize text
    tokens = re.findall(r'\b[a-zA-Z]+\b', text)

    # Remove stopwords
    stopword = nltk.corpus.stopwords.words('english')
    text = [word for word in tokens if word not in stopword]
    
    # Remove words starting with 'https'
    text = [word for word in text if not word.startswith('https')]

    # Lemmatizing
    wn = WordNetLemmatizer()
    text = [wn.lemmatize(word) for word in text]

    # Join text
    text = ' '.join(text)

    return text

def predict_sentiment(text):
    # Preprocess input text data
    processed_text = preprocess_text(text)

    # Apply same vectorization as used for training the model
    csv_vect = vect.transform([processed_text])

    # Predict sentiment using the trained model
    sentiment = model.predict(csv_vect)[0]

    if sentiment >= 0.5:
        return'Positive'
    elif sentiment <= -0.5:
        return 'Negative'
    else:
        return 'Neutral'

    return sentiment

with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        # Clean the input text
        cleaned_text = preprocess_text(text)
        
        # Vectorize the cleaned text using the TfidfVectorizer
        text_vector = vect.transform([cleaned_text])

        # Use the trained model to predict the sentiment
        sentiment = model.predict(text_vector)[0]
        if sentiment >= 0.5:
            emoji_icon = emoji.emojize(":smiling_face_with_smiling_eyes:")
            st.write('Sentiment: ', emoji_icon, 'Positive')
        elif sentiment <= -0.5:
            emoji_icon = emoji.emojize(":disappointed_face:")
            st.write('Sentiment: ', emoji_icon, 'Negative')
        else:
            emoji_icon = emoji.emojize(":neutral_face:")
            st.write('Sentiment: ', emoji_icon, 'Neutral')

    pre = st.text_input('Clean Text: ')
    if pre:
        clean_text = preprocess_text(pre) 
        st.write(clean_text)


with st.expander('Analyze CSV'):
     upl = st.file_uploader('Upload file')

     if upl:
        df = pd.read_csv(upl, names=['Tweets'])
        df['Tweets'] = df['Tweets'].apply(preprocess_text)
        # Predict sentiment
        df['sentiment'] = df['Tweets'].apply(predict_sentiment)
        st.write(df.head(10))

        @st.cache_data
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv',
        )

    
