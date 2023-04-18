import streamlit as st
from snscrape.modules import twitter
import os
from PIL import Image
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import warnings
import re
import emoji
import pandas as pd
import matplotlib.pyplot as plt
import cleantext
from streamlit_option_menu import option_menu

warnings.filterwarnings('ignore')

def set_favicon():
    favicon = Image.open("logo.png")
    st.set_page_config(page_title="Tweet App", page_icon=favicon)

set_favicon()

# Initialize a TfidfVectorizer object
vect = TfidfVectorizer()

# Load your trained model
model = joblib.load('lr_model.joblib')

# Load the vocabulary and idf values from the trained TfidfVectorizer
vect= joblib.load('vectorizer.joblib')


@st.cache_data
def predict_sentiment(text):


    # Apply same vectorization as used for training the model
    csv_vect = vect.transform([text])

    # Predict sentiment using the trained model
    sentiment = model.predict(csv_vect)[0]
    
    if sentiment >= 0.5:
        return'Positive'
    elif sentiment <= -0.5:
        return 'Negative'
    else:
        return 'Neutral'

    return sentiment


with st.sidebar:
    selected=option_menu(
        menu_title = None,
        options=["Home","Tweets Live Analysis", "Text Analysis", "CSV Analysis","Visualization","Project Report"],
        icons=['house-door-fill','twitter','card-text','file-earmark-post-fill','bar-chart-fill','file-earmark-word'],
        menu_icon='cast',
        default_index=0,)
    
       
if selected == "Tweets Live Analysis":

    # Function to get tweets based on keyword or hashtag
    def get_tweets(query):
        tweets = []
        for tweet in twitter.TwitterSearchScraper(query + ' lang:en').get_items():
            tweet_text = tweet.content
            cleaned_tweet = cleantext.clean(tweet_text, clean_all=False, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True)
            sentiment = predict_sentiment(cleaned_tweet)
            tweets.append({'Text': tweet_text, 'Cleaned Text': cleaned_tweet, 'Sentiment': sentiment})
            if len(tweets) == 20:
                break
        return tweets

    # Set up Streamlit app
    st.title('Twitter Search')
    query = st.text_input('Enter a keyword or hashtag')
    if query:
        st.write(f'Showing results for "{query}"')
        tweets = get_tweets(query)
        df = pd.DataFrame(tweets)
        st.table(df)

        # create a  bar chart of sentiment label counts
        labels = ['Positive', 'Negative', 'Neutral']
        counts = [len(df[df['Sentiment'] == 'Positive']),
          len(df[df['Sentiment'] == 'Negative']),
          len(df[df['Sentiment'] == 'Neutral'])]

# create the plot
        fig, ax = plt.subplots()
        bars= ax.bar(labels, counts, color=['green', 'red', 'blue'])
        ax.set_title('Sentiment Label Counts')
        ax.set_xlabel('Sentiment Label')
        ax.set_ylabel('Count')

        # add count labels on top of each bar
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{counts[i]}", ha='center', va='bottom', color='black')

    # display the plot in Streamlit
        st.pyplot(fig)

        
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
        
        
if selected == "Text Analysis":
    with st.expander('Analyze Text'):
        text = st.text_input('Text here: ')
        if text:
        # Clean the input text
            cleaned_text = cleantext.clean(text,clean_all= False, extra_spaces=True ,
                                 stopwords=True ,lowercase=True ,numbers=True , punct=True)
        
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
            st.write(cleaned_text)
    # code for text analysis

    
if selected == "CSV Analysis":
    # Load the Jupyter Notebook
    with st.expander('Analyze CSV'):
     upl = st.file_uploader('Upload file')

     if upl:
        df = pd.read_csv(upl, names=['Tweets'])
        df['Tweets'] = df['Tweets'].apply(lambda x: cleantext.clean(x,clean_all= False, extra_spaces=True ,
                                 stopwords=True ,lowercase=True ,numbers=True , punct=True))
        # Predict sentiment
        df['sentiment'] = df['Tweets'].apply(predict_sentiment)
        st.write(df.head(10))

        # create a  bar chart of sentiment label counts
        labels = ['Positive', 'Negative', 'Neutral']
        counts = [len(df[df['sentiment'] == 'Positive']),
          len(df[df['sentiment'] == 'Negative']),
          len(df[df['sentiment'] == 'Neutral'])]

        # create the plot
        fig, ax = plt.subplots()
        bars= ax.bar(labels, counts, color=['green', 'red', 'blue'])
        ax.set_title('Sentiment Label Counts')
        ax.set_xlabel('Sentiment Label')
        ax.set_ylabel('Count')

        # add count labels on top of each bar
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{counts[i]}", ha='center', va='bottom', color='black')

    # display the plot in Streamlit
        st.pyplot(fig)

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

    

if selected == "Visualization":
    
    # Define the report ID and workspace ID
        select=option_menu(
        menu_title = None,
        options=["Exploratory Data Analysis","Performance Analysis"],
        icons=['clipboard-data','robot'],
        menu_icon='cast',
        default_index=0,
        orientation='horizontal',)
        st.write("")
        st.write("")
    
        if select == "Exploratory Data Analysis":
            report_id = '9e39918f-c187-4300-9474-08cf2913d490'
            page_name = 'EDA'
            embed_url = f"https://app.powerbi.com/reportEmbed?reportId={report_id}&pageName={page_name}&autoAuth=true&ctid=a8eec281-aaa3-4dae-ac9b-9a398b9215e7&filterPaneEnabled=false&navContentPaneEnabled=false"
            iframe = f"<iframe src='{embed_url}' width='120%' height='450' style='border:none;'></iframe>"
            st.markdown(iframe, unsafe_allow_html=True)

        if select == "Performance Analysis":
            report_id = 'e6334fbb-d737-4f7a-aed5-fbc051ee9f27'
            page_name = 'Performance'
            embed_url = f"https://app.powerbi.com/reportEmbed?reportId={report_id}&pageName={page_name}&autoAuth=true&ctid=a8eec281-aaa3-4dae-ac9b-9a398b9215e7&filterPaneEnabled=false&navContentPaneEnabled=false"
            iframe = f"<iframe src='{embed_url}' width='120%' height='450' style='border:none;'></iframe>"
            st.markdown(iframe, unsafe_allow_html=True)
            
    

    
if selected == "Project Report":
    
    st.write("<h1 style='text-align: center;'>Project Report</h1>", unsafe_allow_html=True)
    # Load and display the project report
    # Define the folder path where the images are stored
    image_folder = "images"

# Get the list of image file names in the folder
    image_files = os.listdir(image_folder)

# Sort the file names alphabetically
    image_files.sort()

# Loop through the list of image file names and display each image
    for file_name in image_files:
    # Load the image using PIL
        image = Image.open(os.path.join(image_folder, file_name))

    # Resize the image to fit the Streamlit page width
        width, height = image.size
        new_width = int(width * 0.8)
        new_height = int(height * 0.8)
        resized_image = image.resize((new_width, new_height))

    # Display the image in the center of the Streamlit page
        st.image(resized_image, use_column_width=True)


if selected == "Home":
    # Define the title and image
    title = "👋 Welcome to the World of Tweets"
    image_url = "twitter.png"

# Create a layout with two columns
    col1, col2 = st.columns([20, 4])

# Add the title to the first column
    with col1:
        st.title(title)

# Add the image to the second column
    with col2:
        st.image(image_url, width=100)

    st.write("")
    st.write("<h4><i>'Twitter is a great place to tell the world what you're thinking before you've had a chance to think about it.' - Chris Pirillo</i></h4>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
   
    st.markdown("""
    <style>
        h5 {
            text-align: center;
        }
    
    </style>
    """, unsafe_allow_html=True)

    st.write("<h5>Our platform offers an online solution for analyzing the sentiments of your Tweets. Our machine learning model is utilized to perform the analysis, providing accurate predictions of the overall sentiment of your Tweets. By utilizing this tool, users can gain valuable insights into the sentiment of their Tweets and make data-driven decisions based on the results. Whether you're a business looking to understand customer feedback or an individual looking to gain insights into your personal social media presence, our platform can help you achieve your goals</h5>", unsafe_allow_html=True)
    st.markdown("""
            <h1 style='text-align: center;'>😊 ☹️ 😶</h1>
            """, unsafe_allow_html=True)


