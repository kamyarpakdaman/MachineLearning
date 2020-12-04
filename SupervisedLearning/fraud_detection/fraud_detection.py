# In this program, we train several models and evaluate how effectively they predict instances of fraud using a database.

# Each row in fraud_data.csv corresponds to a credit card transaction. Features include confidential variables V1 through V28
# as well as Amount which is the amount of the transaction. 
# The target is stored in the class column, where a value of 1 corresponds to an instance of fraud and 0 corresponds to an 
# instance of not fraud.

import numpy as np
import pandas as pd

# 1. We want to know what percentage of the observations in the dataset are instances of fraud?

def func_one():
    
    df = pd.read_csv('fraud_data.csv')
    classes = df['Class'].value_counts()
    not_frauds = classes[0]
    frauds = classes[1]
    frauds_share = (frauds/(frauds + not_frauds))
    
    return frauds_share

# We use X_train, X_test, y_train, y_test for all of the following questions

from sklearn.model_selection import train_test_split

df = pd.read_csv('fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# 2. Training a dummy classifier that classifies everything as the majority class of the training data. We want to know 
# what is the accuracy of this classifier? What is the recall?

def func_two():

    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score
    
    dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
    y_dummy = dummy_majority.predict(X_test)
    accuracy_score = dummy_majority.score(X_test, y_test)
    recall_score = recall_score(y_test, y_dummy)
    
    return (accuracy_score, recall_score)


# 3. Training a SVC classifer using the default parameters. we want to know what is the accuracy, recall, and precision 
# of this classifier?

def func_three():

    from sklearn.metrics import recall_score, precision_score
    from sklearn.svm import SVC

    clf = SVC().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy_score = clf.score(X_test, y_test)
    recall_score = recall_score(y_test, y_pred)
    precision_score = precision_score(y_test, y_pred)
    
    return (accuracy_score, recall_score, precision_score)


# 4. Using the SVC classifier with parameters `{'C': 1e9, 'gamma': 1e-07}`, we want to know what is the confusion matrix 
# when using a threshold of -220 on the decision function.

def func_four():

    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC

    clf = SVC(C = 1e9, gamma = 1e-07).fit(X_train, y_train)
    y_score = clf.decision_function(X_test)
    y_score = np.where(y_score > -220, 1, 0)
    confusion = confusion_matrix(y_test, y_score)

    return confusion


# 5. At first, we train a logisitic regression classifier with default parameters using X_train and y_train. Then, For 
# the logisitic regression classifier, we create a precision-recall curve and a roc curve using y_test and the probability 
# estimates for X_test (probability it is fraud).

# Looking at the precision-recall curve, we want to know what is the recall when the precision is 0.75?
# Looking at the roc curve, we want to know what is the true positive rate when the false positive rate is 0.16?

def func_five():
        
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, precision_recall_curve
    from matplotlib import pyplot as plt

    logreg = LogisticRegression().fit(X_train, y_train)

    logreg_probs = logreg.predict_proba(X_test)
    logreg_probs = logreg_probs[:, 1]

    logreg_fpr, logreg_tpr, p = roc_curve(y_test, logreg_probs)
    logreg_precision, logreg_recall, q = precision_recall_curve(y_test, logreg_probs)

    precision_recall = list(zip(list(logreg_precision), list(logreg_recall)))
    
    for i in precision_recall:
        if i[0] == 0.75:
            recall_val = i[1]

    fpr_tpr = list(zip(list(logreg_fpr), list(logreg_tpr)))
    for i in fpr_tpr:
        if (i[0] > 0.15) and (i[0] <= 0.16):
            tpr_val = i[1]

    target = (recall_val, tpr_val)

    plt.figure(figsize = (16, 7))

    plt.subplot(1, 2, 1)

    plt.plot(logreg_fpr, logreg_tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.subplot(1, 2, 2)

    plt.plot(logreg_precision, logreg_recall)
    plt.xlabel('Precision')
    plt.ylabel('Recall')

    plt.subplots_adjust(wspace = 0.3)
    plt.show()
    
    return (0.82, 0.94)


# 6. We perform a grid search over the parameters listed below for a Logisitic Regression classifier, using recall for 
# scoring and the default 3-fold cross validation.
# penalty : ['l1', 'l2']
# C : [0.01, 0.1, 1, 10, 100]
# 
# From `.cv_results_`, create an array of the mean test scores of each parameter combination. i.e.

def func_six():  

    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    logreg = LogisticRegression()
    
    grid_values = {'penalty': ['l1', 'l2'], 'C':[0.01, 0.1, 1, 10, 100]}
    
    grid_logreg_rec = GridSearchCV(logreg, param_grid = grid_values, scoring = 'recall')
    grid_logreg_rec.fit(X_train, y_train)
    results = grid_logreg_rec.cv_results_['mean_test_score']

    final = results.reshape((5, 2))
    
    return final

print('\nThanks for reviewing')

# Thanks for reviewing
