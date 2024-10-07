import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import requests
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load the datasets
train = pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv')
test = pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/test.csv')

# Combine datasets
combined_data = pd.concat([train, test], ignore_index=True, sort=True)

# Clean the tweets
def remove_pattern(text, pattern):
    r = re.findall(pattern, text)
    for i in r:
        text = re.sub(i, "", text)
    return text

combined_data['Cleaned_Tweets'] = np.vectorize(remove_pattern)(combined_data['tweet'], r"@[\w]*")
combined_data['Cleaned_Tweets'] = combined_data['Cleaned_Tweets'].str.replace("[^a-zA-Z#]", " ")
combined_data['Cleaned_Tweets'] = combined_data['Cleaned_Tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

# Tokenization and stemming
tokenized_tweets = combined_data['Cleaned_Tweets'].apply(lambda x: x.split())
from nltk import PorterStemmer
ps = PorterStemmer()
tokenized_tweets = tokenized_tweets.apply(lambda x: [ps.stem(i) for i in x])
combined_data['Clean_Tweets'] = tokenized_tweets.apply(lambda x: ' '.join(x))

# WordCloud Visualization
positive_words = ' '.join(combined_data['Clean_Tweets'][combined_data['label'] == 0])
negative_words = ' '.join(combined_data['Clean_Tweets'][combined_data['label'] == 1])

mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))
wc = WordCloud(background_color='black', height=1500, width=4000, mask=mask).generate(positive_words)
plt.figure(figsize=(10, 20))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# Feature Extraction using Bag of Words and TF-IDF
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words="english")
bow = bow_vectorizer.fit_transform(combined_data['Clean_Tweets'])

tfidf = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(combined_data['Clean_Tweets'])

# Split the data into training and validation sets
x_train_bow, x_valid_bow, y_train_bow, y_valid_bow = train_test_split(bow[:len(train)], train['label'], test_size=0.3, random_state=2)
x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf = train_test_split(tfidf_matrix[:len(train)], train['label'], test_size=0.3, random_state=17)

# Logistic Regression model
log_reg = LogisticRegression(random_state=0, solver='lbfgs')

# Train and evaluate using Bag of Words
log_reg.fit(x_train_bow, y_train_bow)
predict_bow = log_reg.predict(x_valid_bow)

# Evaluate the Bag of Words model
f1_bow = f1_score(y_valid_bow, predict_bow)
precision_bow = precision_score(y_valid_bow, predict_bow, average='weighted')
recall_bow = recall_score(y_valid_bow, predict_bow, average='weighted')
accuracy_bow = accuracy_score(y_valid_bow, predict_bow)

print(f"Bag of Words Metrics:")
print(f"Precision: {precision_bow:.2f}")
print(f"Recall: {recall_bow:.2f}")
print(f"Accuracy: {accuracy_bow:.2f}")
print(f"F1 Score: {f1_bow:.2f}")

# Train and evaluate using TF-IDF
log_reg.fit(x_train_tfidf, y_train_tfidf)
predict_tfidf = log_reg.predict(x_valid_tfidf)

# Evaluate the TF-IDF model
f1_tfidf = f1_score(y_valid_tfidf, predict_tfidf)
precision_tfidf = precision_score(y_valid_tfidf, predict_tfidf, average='weighted')
recall_tfidf = recall_score(y_valid_tfidf, predict_tfidf, average='weighted')
accuracy_tfidf = accuracy_score(y_valid_tfidf, predict_tfidf)

print(f"\nTF-IDF Metrics:")
print(f"Precision: {precision_tfidf:.2f}")
print(f"Recall: {recall_tfidf:.2f}")
print(f"Accuracy: {accuracy_tfidf:.2f}")
print(f"F1 Score: {f1_tfidf:.2f}")

# Final sentiment prediction based on majority class
unique, counts = np.unique(predict_tfidf, return_counts=True)
majority_sentiment = unique[np.argmax(counts)]

# Map the numeric label to sentiment text
sentiment_mapping = {0: 'Positive', 1: 'Negative'}  # Assuming 0 is Positive, 1 is Negative
final_sentiment = sentiment_mapping.get(majority_sentiment, "Neutral")

print(f"\nFinal Sentiment Prediction (Based on majority class): {final_sentiment}")
