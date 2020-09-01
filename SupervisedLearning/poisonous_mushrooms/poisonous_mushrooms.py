# In this program, we'll explore the relationship between model complexity and generalization performance, 
# by adjusting key parameters of various supervised learning models. Part 1 of this program will look at 
# regression and Part 2 will look at classification.

# Part 1 - Regression

# First, we run the following block to set up the variables needed for later sections.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# P1. Let's write a function that fits a polynomial LinearRegression model on the training data, X_train, for degrees 1, 3, 6, and 9.
# For each model, we find 100 predicted values over the interval np.linspace(0,10,100) and store this in a numpy array. 
# The first row of this array should correspond to the output from the model trained on degree 1, the second row degree 3, 
# the third row degree 6, and the fourth row degree 9.

def p_one():
    
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    
    np.random.seed(0)
    n = 15
    x = np.linspace(0,10,n) + np.random.randn(n)/5
    y = np.sin(x)+x/6 + np.random.randn(n)/10

    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
    
    X_train = X_train.reshape(-1, 1)
    
    arr = None
    
    for i in [1, 3, 6, 9]:
    
        poly = PolynomialFeatures(degree = i)
        x_poly = poly.fit_transform(X_train)
    
        linereg = LinearRegression().fit(x_poly, y_train)
    
        data = np.linspace(0,10,100)
    
        result = []
    
        for j in data:
            
            j = j.reshape(1, 1)
            x_predict = poly.transform(j)
            y_predict = np.array(linereg.predict(x_predict))
            result.append(y_predict[0])
    
        result = np.array(result).reshape(1, 100)
    
        if arr == None:
        
            arr = result
    
        else:
    
            arr = np.vstack([arr, result])
    
    return arr
 
# P2. Let's write a function that fits a polynomial LinearRegression model on the training data X_train for degrees 0 through 9.
# For each model, we compute the R^2 (coefficient of determination) regression score on the training data as well as the the test data,
# and return both of these arrays in a tuple.

def p_two():

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    
    np.random.seed(0)
    n = 15
    x = np.linspace(0,10,n) + np.random.randn(n)/5
    y = np.sin(x)+x/6 + np.random.randn(n)/10

    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
    
    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    
    lst_train = []
    lst_test = []
    
    for i in range(0, 10):
    
        poly = PolynomialFeatures(degree = i)
        x_poly = poly.fit_transform(X_train)
    
        linereg = LinearRegression().fit(x_poly, y_train)
        
        score_train = linereg.score(poly.transform(X_train), y_train)
        score_test = linereg.score(poly.transform(X_test), y_test)
        
        lst_train.append(score_train)
        lst_test.append(score_test)

    return (np.array(lst_train), np.array(lst_test))

# P3. Training models on high degree polynomial features can result in overly complex models that overfit, so we often use regularized versions
# of the model to constrain model complexity, as we saw with Ridge and Lasso linear regression.
# In this function, we train two models: a non-regularized LinearRegression model (default parameters) and a regularized Lasso Regression
# model (with parameters alpha=0.01, max_iter=10000) both on polynomial features of degree 12. Then, we return the R^2 score for both 
# the LinearRegression and Lasso model's test sets.

def p_three():

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression

    np.random.seed(0)
    n = 15
    x = np.linspace(0,10,n) + np.random.randn(n)/5
    y = np.sin(x)+x/6 + np.random.randn(n)/10

    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
    
    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    
    poly = PolynomialFeatures(degree = 12)
    x_poly = poly.fit_transform(X_train)
    
    linereg = LinearRegression().fit(x_poly, y_train)
    lasso = Lasso(alpha=0.01, max_iter=10000).fit(x_poly, y_train)
    
    score_linereg = linereg.score(poly.transform(X_test), y_test)
    score_lasso = lasso.score(poly.transform(X_test), y_test)

    return (score_linereg, score_lasso)


# ## Part 2 - Classification

# Here's an application of machine learning that could save our life! We will be working with the mushrooms data set.
# The data will be used to train a model to predict whether or not a mushroom is poisonous. The following attributes are provided:

# Attribute Information:

# 1. cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s 
# 2. cap-surface: fibrous=f, grooves=g, scaly=y, smooth=s 
# 3. cap-color: brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y 
# 4. bruises?: bruises=t, no=f 
# 5. odor: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s 
# 6. gill-attachment: attached=a, descending=d, free=f, notched=n 
# 7. gill-spacing: close=c, crowded=w, distant=d 
# 8. gill-size: broad=b, narrow=n 
# 9. gill-color: black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y 
# 10. stalk-shape: enlarging=e, tapering=t 
# 11. stalk-root: bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=? 
# 12. stalk-surface-above-ring: fibrous=f, scaly=y, silky=k, smooth=s 
# 13. stalk-surface-below-ring: fibrous=f, scaly=y, silky=k, smooth=s 
# 14. stalk-color-above-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y 
# 15. stalk-color-below-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y 
# 16. veil-type: partial=p, universal=u 
# 17. veil-color: brown=n, orange=o, white=w, yellow=y 
# 18. ring-number: none=n, one=o, two=t 
# 19. ring-type: cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z 
# 20. spore-print-color: black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y 
# 21. population: abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y 
# 22. habitat: grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d

# The data in the mushrooms dataset is currently encoded with strings. These values will need to be encoded to numeric 
# to work with sklearn. We'll use pd.get_dummies to convert the categorical variables into indicator variables. 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in P5, we will create a smaller version of the entire mushroom dataset for using in those parts.
# For simplicity, we'll just re-use the 25% test split created above as the representative subset.

X_subset = X_test2
y_subset = y_test2

# P4. Using X_train2 and y_train2, let's train a DecisionTreeClassifier with default parameters and random_state=0. What are
# the 5 most important features found by the decision tree?

def p_four():

    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(random_state = 0)
    tree.fit(X_train2, y_train2)
    
    importance = tree.feature_importances_
    
    dict = {}
    
    for i in range(len(importance)):
        
        dict[importance[i]] = i
    
    lst = list(dict.keys())
    lst = sorted(lst, reverse = True)
    keys = lst[0:5]
    target = []
    
    for k in keys:
        
        target.append(X_train2.columns[dict[k]])

    return target

# P5. For this part, we're going to use the validation_curve function in sklearn.model_selection to determine training and test scores 
# for a Support Vector Classifier (SVC) with varying parameter values.

# Because creating a validation curve requires fitting multiple models, for performance reasons, in this part we will use just 
# a subset of the original mushroom dataset, namely, X_subset and y_subset.

# The initialized unfitted classifier object we'll be using is a Support Vector Classifier with radial basis kernel. With this classifier,
# and the dataset in X_subset, y_subset, we explore the effect of gamma on classifier accuracy by using the validation_curve function 
# to find the training and test scores for 6 values of gamma  in np.logspace(-4,1,6).
 
# For each level of gamma, validation_curve will fit 3 models on different subsets of the data, returning two 6x3 (6 levels of gamma x 3 
# fits per level) arrays of the scores for the training and test sets. We'll find the mean score across the three models for each level 
# of gamma for both arrays, creating two arrays of length 6, and return a tuple with the two arrays. E.g.:

# If one of our arrays of scores is as below:

#     array([[ 0.5,  0.4,  0.6],
#            [ 0.7,  0.8,  0.7],
#            [ 0.9,  0.8,  0.8],
#            [ 0.8,  0.7,  0.8],
#            [ 0.7,  0.6,  0.6],
#            [ 0.4,  0.6,  0.5]])
      
# it will then become

#     array([ 0.5,  0.73333333,  0.83333333,  0.76666667,  0.63333333, 0.5])

def p_five():

    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve
    import numpy as np

    param = np.logspace(-4,1,6)
    train_scores, test_scores = validation_curve(SVC(kernel='rbf', C=1), X_subset, y_subset, 
                                                param_name = 'gamma', param_range = param, cv = 3)
    
    train_s = []
    test_s = []
    
    for row in train_scores:
        
        train_s.append(np.mean(row))
    
    for row in test_scores:
        
        test_s.append(np.mean(row))

    return (np.array(train_s), np.array(test_s))

p_one()
print('This was a function result.\n\n')
p_two()
print('This was a function result.\n\n')
p_three()
print('This was a function result.\n\n')
p_four()
print('This was a function result.\n\n')
p_five
print('This was a function result.\n\n')

print('\nThanks for reviewing')

# Thanks for reviewing
