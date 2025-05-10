# app.py

import nltk
import streamlit as st

# Download the necessary corpora manually (only if not already downloaded)
def download_nltk_corpora():
    try:
        nltk.download('punkt', quiet=True)  # Tokenizer
        nltk.download('averaged_perceptron_tagger', quiet=True)  # POS Tagger
        nltk.download('wordnet', quiet=True)  # WordNet
        nltk.download('omw-1.4', quiet=True)  # OMW WordNet
    except Exception as e:
        st.error(f"Error downloading NLTK corpora: {e}")

download_nltk_corpora()

# Download TextBlob corpora as well (to avoid MissingCorpusError)
from textblob import download_corpora
try:
    download_corpora.download_all()
except Exception as e:
    st.error(f"Error downloading TextBlob corpora: {e}")

# Continue with app imports
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
import pandas as pd
import plotly.express as px
import re

st.set_page_config(page_title="Sentiment & Emotion Analyzer", layout="wide")

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Load Custom Emotions from CSV (optional)
custom_emotions = {}
try:
    emotion_df = pd.read_csv("custom_emotions.csv")
    for _, row in emotion_df.iterrows():
        emotion = row['Emotion'].strip().lower()
        keywords = [kw.strip().lower() for kw in row['Keywords'].split(',')]
        custom_emotions[emotion] = keywords
except FileNotFoundError:
    st.warning("Custom emotions file (custom_emotions.csv) not found. Continuing without custom emotions.")

st.title("💬 Sentiment & Emotion Analyzer")
st.write("### Enter a message below to analyze its sentiment and emotions:")

# Text input
user_message = st.text_area("Message:", height=150)

if st.button("Analyze"):
    if user_message.strip() == "":
        st.warning("⚠️ Please enter a valid message.")
    else:
        # Analyze sentiment
        sentiment = analyzer.polarity_scores(user_message)
        if sentiment['compound'] >= 0.05:
            sentiment_label = 'Positive'
        elif sentiment['compound'] <= -0.05:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'

        # Analyze emotions using NRCLex
        try:
            emotion = NRCLex(user_message)
            emotion_scores = emotion.raw_emotion_scores
        except Exception as e:
            st.error(f"Emotion detection failed: {e}")
            emotion_scores = {}

        # Custom Emotion Detection
        detected_emotions = {}
        total_words = len(user_message.split())
        total_words = max(total_words, 1)

        if custom_emotions:
            for emo, keywords in custom_emotions.items():
                for word in keywords:
                    pattern = r'\b' + re.escape(word.lower()) + r'\b'
                    matches = re.findall(pattern, user_message.lower())
                    count = len(matches)
                    if count > 0:
                        detected_emotions[emo] = detected_emotions.get(emo, 0) + count

        # Combine NRCLex + Custom Detected Emotions
        for emo, count in detected_emotions.items():
            emotion_scores[emo] = emotion_scores.get(emo, 0) + count

        # Normalize emotion scores
        emotion_scores_normalized = {
            emo: round(score / total_words, 4) for emo, score in emotion_scores.items()
        }

        # Sort emotions
        sorted_emotions = sorted(emotion_scores_normalized.items(), key=lambda x: x[1], reverse=True)

        # Display results
        st.subheader("🔍 Analysis Results")
        st.write(f"**Message:** {user_message}")
        st.write(f"**Sentiment:** **{sentiment_label}** ({sentiment})")

        # Visualize sentiment
        sentiment_data = {
            'Positive': sentiment['pos'],
            'Neutral': sentiment['neu'],
            'Negative': sentiment['neg']
        }
        sentiment_df = pd.DataFrame(sentiment_data.items(), columns=["Sentiment", "Score"])
        fig_sentiment = px.pie(sentiment_df, values='Score', names='Sentiment',
                                title='Sentiment Distribution',
                                color_discrete_sequence=px.colors.sequential.RdBu)

        # Visualize emotions
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Sentiment Distribution")
            st.plotly_chart(fig_sentiment, use_container_width=True)

        with col2:
            st.subheader("📈 Emotion Intensities")
            if sorted_emotions:
                emotion_df = pd.DataFrame(sorted_emotions, columns=["Emotion", "Score"])
                fig_emotions = px.bar(emotion_df, x='Emotion', y='Score',
                                       title='Emotion Intensities',
                                       text='Score',
                                       color='Score',
                                       color_continuous_scale='Blues')
                st.plotly_chart(fig_emotions, use_container_width=True)
            else:
                st.info("No emotions detected.")

        # Display emotion scores as list
        st.subheader("📋 Detailed Emotion Scores")
        if not sorted_emotions:
            st.write("- No emotions detected.")
        else:
            for emo, score in sorted_emotions:
                st.write(f"- **{emo.capitalize()}**: {score:.4f}")
