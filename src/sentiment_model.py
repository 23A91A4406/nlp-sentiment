import pandas as pd
import json
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load original dataset (for labels)
df_original = pd.read_csv('data/Tweets.csv')

# Load preprocessed data
df_clean = pd.read_csv('output/preprocessed_data.csv')

# Merge to get sentiment labels
df = pd.merge(df_clean, df_original[['tweet_id', 'airline_sentiment']], on='tweet_id')
df = df.dropna(subset=['cleaned_text'])
df = df[df['cleaned_text'].str.strip() != ""]

X = df['cleaned_text']
y = df['airline_sentiment']

# Train test split
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, df['tweet_id'], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save vectorizer
with open('output/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Save model
with open('output/sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Save predictions
pred_df = pd.DataFrame({
    'tweet_id': id_test,
    'predicted_sentiment': y_pred
})
pred_df.to_csv('output/sentiment_predictions.csv', index=False)

# Metrics
metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision_macro": float(precision_score(y_test, y_pred, average='macro')),
    "recall_macro": float(recall_score(y_test, y_pred, average='macro')),
    "f1_score_macro": float(f1_score(y_test, y_pred, average='macro'))
}

with open('output/sentiment_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Sentiment model training completed successfully")
print("Metrics:", metrics)