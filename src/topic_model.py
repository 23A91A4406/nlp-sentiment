import pandas as pd
import pickle
import json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import pyLDAvis
import pyLDAvis.lda_model

# Load preprocessed data
df = pd.read_csv('output/preprocessed_data.csv')

# Remove empty rows if any
df = df.dropna(subset=['cleaned_text'])
df = df[df['cleaned_text'].str.strip() != ""]

texts = df['cleaned_text']

# Convert text to count vectors
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)

# Train LDA model
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

# Save LDA model
with open('output/lda_model.pkl', 'wb') as f:
    pickle.dump(lda, f)

# Extract top words per topic
feature_names = vectorizer.get_feature_names_out()
topics = {}

for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[-10:]]
    topics[f"topic_{topic_idx}"] = top_words

# Save topics
with open('output/topics.json', 'w') as f:
    json.dump(topics, f, indent=4)

# Create visualization
vis = pyLDAvis.lda_model.prepare(lda, X, vectorizer)
pyLDAvis.save_html(vis, 'output/lda_visualization.html')

print("Topic modeling completed successfully")