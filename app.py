import streamlit as st
import pandas as pd
import plotly.express as px
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Sentiment & Emotion Analyzer", layout="wide")

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Load Custom Emotions from CSV
custom_emotions = {}
try:
    emotion_df = pd.read_csv("custom_emotions.csv")
    for _, row in emotion_df.iterrows():
        emotion = row['Emotion'].strip().lower()
        keywords = [kw.strip().lower() for kw in row['Keywords'].split(',')]
        custom_emotions[emotion] = keywords
except FileNotFoundError:
    st.error("Custom emotions file (custom_emotions.csv) not found. Please ensure it exists in the directory.")

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

        # Custom Emotion Detection
        detected_emotions = {}
        total_words = len(user_message.split())
        total_words = max(total_words, 1)

        for emo, keywords in custom_emotions.items():
            for word in keywords:
                pattern = r'\b' + re.escape(word.lower()) + r'\b'
                matches = re.findall(pattern, user_message.lower())
                count = len(matches)
                if count > 0:
                    detected_emotions[emo] = detected_emotions.get(emo, 0) + count

        # Normalize and Sort Emotion Scores
        emotion_scores_normalized = {emo: round(score / total_words, 4) for emo, score in detected_emotions.items()}
        sorted_emotions = sorted(emotion_scores_normalized.items(), key=lambda x: x[1], reverse=True)

        # Display results
        st.subheader("🔍 Analysis Results")
        st.write(f"**Message:** {user_message}")
        st.write(f"**Sentiment:** **{sentiment_label}** ({sentiment})")

        # Side-by-Side Layout
        col1, col2 = st.columns(2)

        with col1:
            # Visualize Sentiment (Pie Chart)
            sentiment_data = {'Positive': sentiment['pos'], 'Neutral': sentiment['neu'], 'Negative': sentiment['neg']}
            sentiment_df = pd.DataFrame(sentiment_data.items(), columns=["Sentiment", "Score"])
            fig_sentiment = px.pie(sentiment_df, values='Score', names='Sentiment', title='Sentiment Distribution', color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_sentiment, use_container_width=True)

        with col2:
            # Visualize Emotions (Bar Chart)
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
