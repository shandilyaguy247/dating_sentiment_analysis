import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

# Define the directory
directory = "datasets/"

# Load the sentences
sentences = pd.read_csv(directory + 'datasetSentences.txt', sep='\t')

# Load the dictionary
dictionary = pd.read_csv(directory + 'dictionary.txt', sep='|', names=['phrase', 'phrase_ids'])

# Load the sentiment labels
sentiments = pd.read_csv(directory + 'sentiment_labels.txt', sep='|')
sentiments.columns = ['phrase_ids', 'sentiment values']

# Merge the dictionary with the sentiments
dictionary = pd.merge(dictionary, sentiments, on='phrase_ids', how='inner')

def label_sentiment(row):
    if row['sentiment values'] < 0.510058:
        return 'negative'
    elif row['sentiment values'] < 0.526534:
        return 'neutral'
    else:
        return 'positive'

# Label sentiments for each phrase
dictionary['sentiment'] = dictionary.apply(label_sentiment, axis=1)

# Load the splits
splits = pd.read_csv(directory + 'datasetSplit.txt', sep=',')
splits.columns = ['sentence_index', 'splitset_label']

# Merge the sentences DataFrame with the dictionary DataFrame on phrase/sentence
final_data = pd.merge(sentences, dictionary, left_on='sentence', right_on='phrase', how='inner')

# Then merge the final_data DataFrame with the splits DataFrame
final_data = pd.merge(final_data, splits, on='sentence_index', how='inner')

# Print the data with sentiment labels
print(final_data.head())

vectorizer = TfidfVectorizer(use_idf=True)
X = vectorizer.fit_transform(final_data['sentence'])

# Convert sentiment labels into numerical form
encoder = LabelEncoder()
y = encoder.fit_transform(final_data['sentiment'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a simple neural network model
model = keras.models.Sequential([
    keras.layers.Dense(100, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train.toarray(), y_train, epochs=10, batch_size=32)

# Evaluate the model
predictions = model.predict(X_test.toarray())
predictions = np.argmax(predictions, axis=1)
print(classification_report(y_test, predictions))

# Print the confusion matrix
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)

# Use the trained model to predict sentiment of new texts
conversation = [
    "Did it hurt?",
    "Did what hurt?",
    "When you fell from Heaven",
    "That pickup line has been done to death",
    "Oh, you got me! I promise, no more cheesy pickup lines if I can buy you a beer tonight?",
    "Where at?",
    "Harry's is awesome. I know the bartender and she makes some killer cocktails!",
    "Harry's it is",
    "Pick me up at 9?",
    "You bet, see you at 9 :)",
    ":D",
    "I love you",
    "I love you",
]

X_new = vectorizer.transform(conversation)
new_predictions = model.predict(X_new.toarray())
new_predictions = np.argmax(new_predictions, axis=1)

# Map numeric predictions back to sentiment labels
sentiment_mapping = {i: sentiment for i, sentiment in enumerate(encoder.classes_)}
predictions = [sentiment_mapping[pred] for pred in new_predictions]

# Map sentiment labels back to their numerical form
inverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}
numerical_predictions = [inverse_sentiment_mapping[pred] for pred in predictions]

# Define weights for the sentences: higher weight for recent sentences
weights = np.arange(1, len(numerical_predictions) + 1)

# Normalize the weights
weights = weights / np.sum(weights)

# Calculate the weighted mean sentiment
mean_sentiment = np.average(numerical_predictions, weights=weights)

# Assign overall sentiment based on the mean
if mean_sentiment < -2/3:
    overall_sentiment = "negative"
elif mean_sentiment < 1/3:
    overall_sentiment = "neutral"
else:
    overall_sentiment = "positive"

print("The overall sentiment of the conversation is:", overall_sentiment)

# Print the individual sentiment of each line along with the weight
for i in range(len(conversation)):
    print(f"Sentence: {conversation[i]}")
    print(f"Sentiment: {predictions[i]}")
    print(f"Weight: {weights[i]}")
    print()

# Count the occurrences of each sentiment and weight them
sentiment_counts = Counter({sentiment: numerical_predictions.count(inverse_sentiment_mapping[sentiment]) * weight for sentiment, weight in zip(predictions, weights)})

# Create a bar plot for the sentiment counts
plt.figure(figsize=(10, 5))
sentiments = ["negative", "neutral", "positive"]
counts = [sentiment_counts.get(sentiment, 0) for sentiment in sentiments]
colors = ['red', 'grey', 'green']
plt.bar(sentiments, counts, color=colors)
plt.xlabel('Sentiments')
plt.ylabel('Weighted Count')
plt.title('Weighted distribution of sentiments in the conversation')
plt.show()

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
