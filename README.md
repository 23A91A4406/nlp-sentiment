# Social Media Sentiment & Topic Analysis Dashboard
## Overview
This project implements a complete Natural Language Processing (NLP) pipeline to analyze sentiment and extract topics from social media text data. It processes raw tweets, performs text preprocessing, trains machine learning models for sentiment classification and topic modeling, and presents the results in an interactive Streamlit dashboard.

The application is fully containerized using Docker to ensure reproducibility and ease of deployment.

## Objectives
* Clean and preprocess raw social media text data
* Extract features using TF-IDF
* Train a sentiment classification model
* Train a topic model using Latent Dirichlet Allocation (LDA)
* Save all trained models and evaluation artifacts
* Visualize results in an interactive dashboard
* Run the complete application inside Docker containers

## Dataset
The project uses the Twitter US Airline Sentiment dataset, which contains tweets related to airline services along with sentiment labels such as positive, negative, and neutral.

## Features
* Text preprocessing (lowercasing, URL removal, stopword removal, lemmatization)
* TF-IDF vectorization
* Sentiment classification using Logistic Regression
* Model evaluation with accuracy, precision, recall, and F1-score
* Topic modeling using LDA
* Interactive topic visualization using pyLDAvis
* Streamlit-based interactive dashboard
* Docker and Docker Compose based containerization


## Output Artifacts
All generated artifacts are stored in the output/ directory:

* preprocessed_data.csv
* tfidf_vectorizer.pkl
* sentiment_model.pkl
* sentiment_metrics.json
* sentiment_predictions.csv
* lda_model.pkl
* topics.json
* lda_visualization.html

## Project Structure
```bash 
├── data/
├── output/
├── src/
│   ├── preprocess.py
│   ├── sentiment_model.py
│   └── topic_model.py
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Running the Application

### Prerequisites
* Docker
* Docker Compose


## Build and Run
```bash
docker-compose up --build
```
After the application starts, open browser and visit http://localhost:8501