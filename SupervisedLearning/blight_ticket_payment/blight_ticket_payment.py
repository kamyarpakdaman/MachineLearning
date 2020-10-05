# In this program, we create three models to predict whether a given blight ticket will be paid on time. We use GridSearchCV
# to find out which parameter values generate better results combined.

# Note that eventually there is no y_test for these models. Since this was part of a course project, the y_test was in the grader tool. D:

def blight_model():
    
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    train = pd.read_csv('train.csv', encoding='ISO-8859-1')
    test = pd.read_csv('test.csv', encoding = "ISO-8859-1")

    train = train.dropna(subset = ['compliance'])
    target = train['compliance']

    train_drop = ['payment_amount', 'payment_date', 'payment_status', 'balance_due', 
                  'collection_status', 'grafitti_status', 'compliance_detail', 'compliance']
    train.drop(train_drop, axis=1, inplace=True)

    df = pd.concat([train, test])
    df = df.set_index('ticket_id')

    common_drop = ['clean_up_cost', 'city', 'grafitti_status', 'discount_amount', 
                   'violation_street_number', 'violation_description', 'ticket_issued_date', 
                   'hearing_date', 'state', 'zip_code', 'non_us_str_code', 'country', 
                   'violator_name', 'violator_name', 'violation_zip_code', 'mailing_address_str_number', 
                   'mailing_address_str_name']
    df.drop(common_drop, axis=1, inplace=True)

    cat_columns = ['agency_name', 'inspector_name', 'violation_street_name', 'violation_code', 'disposition']
    
    for col in cat_columns:
        df[col] = df[col].astype('category') 
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    X_train = df.iloc[0:len(train),:]
    y_train = target
    X_test = df.iloc[len(train):,:]

    logreg = LogisticRegression()

    grid_values = {'C': [0.01, 0.1, 1, 10]}
    
    # Here we create the GridSearchCV.

    grid_logreg_auc = GridSearchCV(logreg, param_grid = grid_values, scoring = 'roc_auc')

    # Here we want our GridSearchCV to create the CV folds in the train and test datasets X_train and y_train, and evaluate
    # which parameters work better combined.

    grid_logreg_auc.fit(X_train, y_train)

    # These are the results of the GridSearchCV

    logreg_best_params = grid_logreg_auc.best_params_
    logreg_best_score = grid_logreg_auc.best_score_

    print('For LogisticRegression, best params are {} and best score is {}'.format(logreg_best_params, logreg_best_score))
    
    # Below, we use the GridSearchCV for predicting the target labels for X_test. After comparing the results of tht GridSearchCV
    # and the y_test, the grader tool would compute the ROC AUC.

    logreg_result = grid_logreg_auc.predict_proba(X_test)
    
    # Performing the same process two more times for two different models.

    # A DecisionTreeClassifier.

    tree = DecisionTreeClassifier()
    grid_values = {'max_depth': [6, 7, 8]}
    grid_tree_auc = GridSearchCV(tree, param_grid = grid_values, scoring = 'roc_auc')
    grid_tree_auc.fit(X_train, y_train)
    tree_best_params = grid_tree_auc.best_params_
    tree_best_score = grid_tree_auc.best_score_
    print('For DecisionTreeClassifier, best params are {} and best score is {}'.format(tree_best_params, tree_best_score))
    tree_result = grid_tree_auc.predict_proba(X_test)
    
    # A RandomForestClassifier.

    rf = RandomForestClassifier()
    grid_values = {'max_features': [5, 10, 15]}
    grid_rf_auc = GridSearchCV(rf, param_grid = grid_values, scoring = 'roc_auc')
    grid_rf_auc.fit(X_train, y_train)
    rf_best_params = grid_rf_auc.best_params_
    rf_best_score = grid_rf_auc.best_score_
    print('For RandomForestClassifier, best params are {} and best score is {}'.format(rf_best_params, rf_best_score))
    rf_result = grid_rf_auc.predict_proba(X_test)
    
    answer = pd.Series(tree_result[:,1], index=X_test.index)

    return answer

blight_model()

print('\nThanks for reviewing')

# Thanks for reviewing 
