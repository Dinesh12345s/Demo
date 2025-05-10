import streamlit as st
import pandas as pd
import re
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex

# Initialize Sentiment Analyzer globally
sentiment_analyzer = SentimentIntensityAnalyzer()

# Load custom emotions from CSV
custom_emotions = {}
try:
    emotion_df = pd.read_csv("custom_emotions.csv")
    for _, row in emotion_df.iterrows():
        emotion = row['Emotion'].strip().lower()
        keywords = [kw.strip().lower() for kw in row['Keywords'].split(',')]
        custom_emotions[emotion] = keywords
except FileNotFoundError:
    st.error("Custom emotions file (custom_emotions.csv) not found. Please ensure it exists in the directory.")

def analyze_sentiment(text: str):
    """
    Returns sentiment label and sentiment score dict from VADER.
    """
    sentiment = sentiment_analyzer.polarity_scores(text)
    if sentiment['compound'] >= 0.05:
        label = 'Positive'
    elif sentiment['compound'] <= -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'
    return label, sentiment

def extract_emotion_scores(text: str):
    """
    Returns NRCLex emotion scores as a dict.
    Example: {'joy': 2, 'fear': 1}
    """
    emotion = NRCLex(text)
    return emotion.raw_emotion_scores

def extract_custom_emotions(text: str, custom_emotions: dict):
    """
    Returns custom detected emotions based on keyword matches.
    Example: {'gratitude': 3}
    """
    detected_emotions = {}
    for emo, keywords in custom_emotions.items():
        for word in keywords:
            pattern = r'\b' + re.escape(word.lower()) + r'\b'
            matches = re.findall(pattern, text.lower())
            count = len(matches)
            if count > 0:
                detected_emotions[emo] = detected_emotions.get(emo, 0) + count
    return detected_emotions

def combine_emotions(text: str, custom_emotions: dict = None):
    """
    Combines NRCLex and custom detected emotions into normalized scores.
    Returns sorted list of (emotion, normalized_score)
    """
    # 1. Emotion analysis with NRCLex
    nrc_scores = extract_emotion_scores(text)
    
    # 2. Custom Emotion Detection based on CSV file
    custom_scores = extract_custom_emotions(text, custom_emotions) if custom_emotions else {}

    # 3. Combine the results
    combined = nrc_scores.copy()
    for emo, count in custom_scores.items():
        combined[emo] = combined.get(emo, 0) + count

    # 4. Normalize by word count
    total_words = len(text.split())
    total_words = max(total_words, 1)

    # 5. Normalize and sort by score
    normalized_scores = {
        emo: round(score / total_words, 4) for emo, score in combined.items()
    }

    return sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)

# Streamlit UI setup
st.set_page_config(page_title="Sentiment & Emotion Analyzer", layout="wide")

st.title("💬 Sentiment & Emotion Analyzer")
st.write("### Enter a message below to analyze its sentiment and emotions:")

# Text input
user_message = st.text_area("Message:", height=150)

if st.button("Analyze"):
    if user_message.strip() == "":
        st.warning("⚠️ Please enter a valid message.")
    else:
        # Sentiment analysis using VADER
        sentiment_label, sentiment = analyze_sentiment(user_message)

        # Emotion analysis combining NRC and custom emotions
        sorted_emotions = combine_emotions(user_message, custom_emotions)

        # Visualizations and results display
        st.subheader("🔍 Analysis Results")
        st.write(f"**Message:** {user_message}")
        st.write(f"**Sentiment:** **{sentiment_label}** ({sentiment})")

        # Visualize sentiment (Pie Chart)
        sentiment_data = {
            'Positive': sentiment['pos'],
            'Neutral': sentiment['neu'],
            'Negative': sentiment['neg']
        }
        sentiment_df = pd.DataFrame(sentiment_data.items(), columns=["Sentiment", "Score"])
        fig_sentiment = px.pie(sentiment_df, values='Score', names='Sentiment',
                               title='Sentiment Distribution', color_discrete_sequence=px.colors.sequential.RdBu)

        # Visualize emotions (Bar Chart)
        if sorted_emotions:
            emotion_df = pd.DataFrame(sorted_emotions, columns=["Emotion", "Score"])
            fig_emotions = px.bar(emotion_df, x='Emotion', y='Score',
                                  title='Emotion Intensities', text='Score',
                                  color='Score', color_continuous_scale='Blues')
        else:
            emotion_df = None

        # Display side-by-side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Sentiment Distribution")
            st.plotly_chart(fig_sentiment, use_container_width=True)

        with col2:
            st.subheader("📈 Emotion Intensities")
            if emotion_df is not None:
                st.plotly_chart(fig_emotions, use_container_width=True)
            else:
                st.info("No emotions detected.")

        # Also display emotion scores as list
        st.subheader("📋 Detailed Emotion Scores")
        if not sorted_emotions:
            st.write("- No emotions detected.")
        else:
            for emo, score in sorted_emotions:
                st.write(f"- **{emo.capitalize()}**: {score:.4f}")
