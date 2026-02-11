import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="NLP Sentiment & Topic Dashboard", layout="wide")

st.title("Airline Tweets NLP Analysis Dashboard")

# Load metrics
with open('output/sentiment_metrics.json') as f:
    metrics = json.load(f)

st.header("Sentiment Model Performance")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", round(metrics["accuracy"], 2))
col2.metric("Precision", round(metrics["precision_macro"], 2))
col3.metric("Recall", round(metrics["recall_macro"], 2))
col4.metric("F1 Score", round(metrics["f1_score_macro"], 2))

# Sentiment distribution
st.header("Sentiment Distribution")

pred_df = pd.read_csv('output/sentiment_predictions.csv')
sentiment_counts = pred_df['predicted_sentiment'].value_counts()

st.bar_chart(sentiment_counts)

# Topics
st.header("Discovered Topics")

with open('output/topics.json') as f:
    topics = json.load(f)

for topic, words in topics.items():
    st.subheader(topic)
    st.write(", ".join(words))

# LDA Visualization
st.header("LDA Topic Visualization")

with open('output/lda_visualization.html', 'r', encoding='utf-8') as f:
    html_data = f.read()

st.components.v1.html(html_data, width=1300, height=800)