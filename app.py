# streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
from collections import Counter
import io

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Sentiment & Emotion Analyzer", layout="wide")

st.title("Sentiment & Emotion Analyzer")

def process_file(df):
    """Process DataFrame to extract sentiment and emotion data."""
    sentiments = []
    emotions = []
    results_text = ""

    for _, row in df.iterrows():
        text = row["message"]
        if pd.isna(text) or str(text).strip() == "":
            continue
        text = str(text).strip()

        sentiment = analyzer.polarity_scores(text)
        emotion = NRCLex(text)

        if sentiment['compound'] >= 0.05:
            sentiment_label = 'Positive'
        elif sentiment['compound'] <= -0.05:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'
        sentiments.append(sentiment_label)

        for emo, _ in emotion.top_emotions:
            emotions.append(emo)

        result = (
            f"Message: {text}\n"
            f"Sentiment: {sentiment_label} ({sentiment})\n"
            f"Emotions: {emotion.top_emotions}\n"
            f"{'-'*60}\n"
        )
        results_text += result

    return sentiments, emotions, results_text

def plot_pie_chart(data_list, title):
    """Plot and return pie chart."""
    counts = Counter(data_list)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', startangle=140)
    ax.set_title(title)
    st.pyplot(fig)

def plot_bar_chart(data_list, title, xlabel):
    """Plot and return bar chart."""
    counts = Counter(data_list)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.bar(counts.keys(), counts.values(), color='lightcoral')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# File upload
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        file_name = uploaded_file.name.lower()
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()

        # Normalize column names
        df.columns = [col.lower() for col in df.columns]

        if "message" not in df.columns:
            st.error("The file must contain a 'message' column (case-insensitive).")
            st.stop()

        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        st.write(f"*Total Messages:* {len(df)}")

        # Process file
        sentiments, emotions, results_text = process_file(df)

        # Display pie and bar charts
        st.subheader("Sentiment Distribution")
        plot_pie_chart(sentiments, "Sentiment Pie Chart")
        plot_bar_chart(sentiments, "Sentiment Bar Chart", "Sentiment")

        st.subheader("Emotion Distribution")
        plot_pie_chart(emotions, "Emotion Pie Chart")
        plot_bar_chart(emotions, "Emotion Bar Chart", "Emotion")

        # Display text output in expandable section
        with st.expander("See Full Analysis Text"):
            st.text(results_text)

        # Option to download the result text
        result_bytes = results_text.encode('utf-8')
        st.download_button(
            label="Download Analysis Text",
            data=result_bytes,
            file_name='analysis_results.txt',
            mime='text/plain'
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")