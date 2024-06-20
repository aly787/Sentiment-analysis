import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("twitter_training.csv")

# Download stopwords
nltk.download('stopwords')
stop_words = set(nltk_stopwords.words('english'))

def cleaning(text):
    try:
        token = text.split()
        tokens = [word.lower() for word in token if word.lower() not in stop_words]
        tokens = [word for word in tokens if word.isalpha()]
        return ' '.join(tokens)
    except AttributeError:
        return ''

# Clean the tweets
data['cleaned_text'] = data.iloc[:, 3].apply(cleaning)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=2000)
features = vectorizer.fit_transform(data['cleaned_text'])

# Splitting data for training
X_train, X_test, y_train, y_test = train_test_split(features, data.iloc[:, 2], test_size=0.2)

# Training the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Function to predict sentiment
def predict_sentiment(new_statement):
    cleaned = cleaning(new_statement)
    cleaned_features = vectorizer.transform([cleaned])
    prediction = model.predict(cleaned_features)[0]
    return prediction

# Example usage:
new_statement = "I, for one, welcome our new insect overlords."
predicted_sentiment = predict_sentiment(new_statement)
print(f"Predicted sentiment for '{new_statement}': {predicted_sentiment}")
