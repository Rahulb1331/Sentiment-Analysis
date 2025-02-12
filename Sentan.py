import streamlit as st
import pdfplumber
import pytesseract
from PIL import Image
import nltk
nltk.download('punkt_tab')
from nltk.sentiment import SentimentIntensityAnalyzer
import io
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import pandas as pd
from collections import Counter

# Download VADER lexicon (only needed once)
nltk.download('vader_lexicon')
nltk.download('punkt_tab')

# Installing Tesseract-OCR for streamlit
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()


# Function to extract text from PDFs
@st.cache_data()
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()


# Function to extract text from images using OCR
@st.cache_data()
def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file)
    text = pytesseract.image_to_string(image)
    return text.strip()


# Function to extract text from text files
@st.cache_data()
def extract_text_from_txt(uploaded_file):
    return uploaded_file.getvalue().decode("utf-8").strip()


# Function to analyze sentiment
@st.cache_data()
def analyze_sentiment(text):
    if not text:
        return None
    sentiment_scores = sia.polarity_scores(text)
    if sentiment_scores['compound'] >= 0.05:
        sentiment = "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, sentiment_scores


# Function to generate a word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)


# Function to plot sentiment scores
def plot_sentiment_scores(scores):
    df = pd.DataFrame(scores, index=[0])
    df = df.melt(var_name='Metric', value_name='Score')
    plt.figure(figsize=(6, 4))
    sns.barplot(x='Metric', y='Score', data=df, palette='coolwarm')
    plt.xticks(rotation=45)
    plt.title("Sentiment Scores")
    st.pyplot(plt)


# Function to extract top sentiment words
def get_top_sentiment_words(text):
    words = nltk.word_tokenize(text)
    word_scores = {word: sia.polarity_scores(word)['compound'] for word in words if word.isalnum()}
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    top_positive = [w[0] for w in sorted_words if w[1] > 0][:5]
    top_negative = [w[0] for w in sorted_words if w[1] < 0][-5:]
    top_neutral = [w[0] for w in sorted_words if w[1] == 0][:5]
    return top_positive, top_negative, top_neutral


# Streamlit UI
st.title("Sentiment Analysis Web App")
st.write("Upload a **PDF, Image, or Text file** to analyze its sentiment.")

uploaded_file = st.file_uploader("Upload File", type=["pdf", "png", "jpg", "jpeg", "txt"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    text = ""

    if "pdf" in file_type:
        text = extract_text_from_pdf(uploaded_file)
    elif "image" in file_type:
        text = extract_text_from_image(uploaded_file)
    elif "text" in file_type:
        text = extract_text_from_txt(uploaded_file)

    if text:
        st.subheader("Extracted Text:")
        st.text_area("", text, height=200)

        st.subheader("Sentiment Analysis:")
        sentiment, scores = analyze_sentiment(text)
        st.write(f"**Overall Sentiment:** {sentiment}")
        st.json(scores)

        st.subheader("Word Cloud:")
        generate_wordcloud(text)

        st.subheader("Sentiment Score Breakdown:")
        plot_sentiment_scores(scores)

        st.subheader("Top Words for each Sentiment:")
        top_positive, top_negative, top_neutral = get_top_sentiment_words(text)
        st.write("**Top Positive Words:**", ', '.join(top_positive))
        st.write("**Top Negative Words:**", ', '.join(top_negative))
        st.write("**Top Neutral Words:**", ', '.join(top_neutral))
    else:
        st.error("Could not extract any text. Try another file.")
