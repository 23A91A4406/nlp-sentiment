import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load dataset
df = pd.read_csv('data/Tweets.csv')

print("Columns in dataset:", df.columns)

# Keep required columns
df = df[['tweet_id', 'text']]

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(words)

df['cleaned_text'] = df['text'].apply(clean_text)

# Save output
df[['tweet_id', 'cleaned_text']].to_csv('output/preprocessed_data.csv', index=False)

print("Preprocessing completed successfully")