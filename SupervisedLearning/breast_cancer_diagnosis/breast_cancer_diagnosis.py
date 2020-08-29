# In this program, we will create a K-nearest Neighbors model to predict the diagnosis label
# for breast cancer tumors.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Reading the file.

def file_handle():
    cancerdf = pd.read_csv('breast_cancer.csv')
    return cancerdf

# Checking the distribution of dioagnoses in the current dataset.

def diagnosis_distribution():
    
    cancerdf = file_handle()
    target = cancerdf[['id', 'diagnosis']].groupby('diagnosis')['id'].count()
    
    return target

# Creating two datasets, one including the features and the other including the labels.

def features_labels():
    
    cancerdf = file_handle()
    features_cols = cancerdf.columns[1:-1]
    labels_cols = cancerdf.columns[-1]
    the_data = cancerdf[features_cols]
    the_labels = cancerdf[labels_cols]
    
    return the_data, the_labels

# Creating the training and test datasets.

def train_test():
    
    X, y = features_labels()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    
    return (X_train, X_test, y_train, y_test)

# Creating a KNN model and training it with our training dataset.

def knn_model():
    X_train, X_test, y_train, y_test = train_test()
    
    model = KNeighborsClassifier(n_neighbors = 1)
    model.fit(X_train, y_train)
    
    return model

# Predicting the label for the mean values of the columns as an individual input.

def mean_prediction():
    
    cancerdf = file_handle()
    means = cancerdf.mean()[:-1].values
    
    knn = knn_model()
    
    result = knn.predict([means])
    
    return result

# Predicting the labels for the test dataset.

def test_prediction():
    
    X_train, X_test, y_train, y_test = train_test()
    knn = knn_model()
    
    result = knn.predict(X_test)
    
    return result

# Assessing the accuracy of our model using test dataset actual and predicted labels.

def model_accuracy():
    
    X_train, X_test, y_train, y_test = train_test()
    knn = knn_model()
    
    result = knn.score(X_test, y_test)
    
    return result

# Drawing a plot to check the accuracy of the model for benign and malignant diagnoses.

def accuracy_plot():

    X_train, X_test, y_train, y_test = train_test()

    mal_train_X = X_train[y_train=='M']
    mal_train_y = y_train[y_train=='M']
    ben_train_X = X_train[y_train=='B']
    ben_train_y = y_train[y_train=='B']

    mal_test_X = X_test[y_test=='M']
    mal_test_y = y_test[y_test=='M']
    ben_test_X = X_test[y_test=='B']
    ben_test_y = y_test[y_test=='B']

    knn = knn_model()

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]


    plt.figure()

    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
                     ha='center', color='w', fontsize=11)

    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'])
    plt.title('Training and Test Accuracies for Malignant and Benign Diagnoses')

    plt.savefig('BarPlot.png')

    plt.show()

file_handle()
diagnosis_distribution()
features_labels()
train_test()
knn_model()
mean_prediction()
test_prediction()
model_accuracy()
accuracy_plot()


print('\nThanks for reviewing')

# Thanks for reviewing
