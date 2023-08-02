# dating_sentiment_analysis
### Contextual Sentiment Analysis for Conversations on Dating Apps

Sentiment Analysis with Neural Networks and SVM

This repository contains Python scripts for training sentiment analysis models using a Neural Network and a Support Vector Machine (SVM). Both scripts use the same dataset and perform similar preprocessing steps. The major difference lies in the machine learning model used for sentiment analysis.

The dataset used in this project is Stanford's Sentiment Treebank, which is a standard dataset for sentiment analysis that includes sentence-level sentiment labels.

### Technologies used

- **Python**: The scripts are written in Python, which is a versatile high-level programming language widely used for data science and machine learning tasks.

- **Pandas**: This is a powerful data manipulation library in Python. It is used here to load, merge, and manipulate the dataset.

- **Scikit-learn**: This is a machine learning library in Python. In this project, it is used for TF-IDF vectorization, data splitting, training the SVM model, and evaluation metrics.

- **Tensorflow**: This is a comprehensive library for machine learning and deep learning. It is used here for defining, compiling, training, and using the neural network model.

- **Keras**: This is a high-level neural networks API, written in Python and capable of running on top of TensorFlow. It is used here for the neural network model definition and training.

- **Seaborn and Matplotlib**: These are data visualization libraries in Python. They are used for plotting confusion matrix and sentiment distribution.

### Scripts

#### Neural Network

The script for the neural network model is `neural_network.py`. It starts with loading the necessary libraries and defining the directory for the dataset.

The sentences, dictionary, and sentiment labels are loaded from respective .txt files. These datasets are merged to create a final dataset with sentences and their corresponding sentiment labels.

Sentiment labels are assigned as 'negative', 'neutral', or 'positive' based on their sentiment values.

The sentences are then vectorized using TF-IDF vectorizer, and the sentiment labels are encoded into numerical values.

The dataset is split into training and testing sets, and a simple neural network model is defined using TensorFlow's Keras. The model is compiled and trained on the training data.

The trained model is then evaluated on the test data, and a classification report and confusion matrix are printed.

The trained model is used to predict the sentiment of a conversation, and the overall sentiment of the conversation is calculated using a weighted mean.

Finally, a bar plot is created for the weighted count of sentiments, and a heatmap is created for the confusion matrix.

#### SVM

The script for the SVM model is `svm.py`. It follows the same steps as the neural network script for data loading, merging, and preprocessing.

However, instead of a neural network, a Support Vector Machine (SVM) model with a linear kernel and balanced class weights is used for sentiment analysis. The SVM model is trained on the training data and evaluated on the test data.

Like in the neural network script, the trained SVM model is used to predict the sentiment of a conversation, and the overall sentiment of the conversation is calculated using a weighted mean.

A bar plot is created for the weighted count of sentiments, and a heatmap is created for the confusion matrix.

### Contribution

Feel free to fork this project, make changes according to your needs and propose any improvements via a Pull Request.
