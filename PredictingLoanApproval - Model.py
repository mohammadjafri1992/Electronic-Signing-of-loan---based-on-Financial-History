# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 01:19:37 2019

@author: Syed Jafri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time

random.seed(100)

# The reason we use seeds is so that we can replicate our results 
# later on, or if someone else wants to replicate our model

# Importing dataset

dataset = pd.read_csv('Financial-Data.csv')
dataset = dataset.drop(columns=['months_employed'])

# months_employed columns adds no value to our model

dataset['personal_account_months'] = (dataset.personal_account_m + (dataset.personal_account_y * 12))

dataset[['personal_account_m','personal_account_y','personal_account_months']].head()

dataset = dataset.drop(columns=['personal_account_m', 'personal_account_y'])

# We calculated the total number of months using two columns i.e. months, and years.
# Then we dropped those columns from our working dataset to reduce its size.

# Making sure which field is important and which is not is a very important and critical
# decision which can make or break a model. Therefore feature engineering
# should only be done by those who have domain knowledge of the problem at hand.



# Now, we are going to perform one-hot encoding and feature scaling below.

# One-hot encoding
dataset = pd.get_dummies(dataset)
dataset.columns
dataset = dataset.drop(columns=['pay_schedule_semi-monthly'])

# Since there are 3 columns which essentially mean the same thing, i.e. 3 columns
# for how the people get paid, we need to make sure we do not have all 3 in our
# dataset before we create our model because those columns are linearly dependent
# on one another and we do not want linear dependeicies in our model.

# Dropping extra columns
response = dataset['e_signed']
users = dataset['entry_id']
dataset.drop(columns=['e_signed','entry_id'])

# Splitting into Train and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, response, test_size=0.2,
                                                    random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.fit_transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2


# Model Building 

# Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0, penalty='l1')
classifier.fit(X_train, y_train)

# Predictions with Test set
y_pred = classifier.predict(X_test)


from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Linear Regression (Lasso)', accuracy, precision, recall, f1]],
               columns=['Model','Accuracy','Precision','Recall','F1 Score'])


# Support Vector Machines - SVM (linear)
from sklearn.svm import SVC
classifier = SVC(random_state=0, kernel='linear')
classifier.fit(X_train, y_train)

# Predictions with Test set
y_pred = classifier.predict(X_test)


from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results2 = pd.DataFrame([['SVM(Linear)', accuracy, precision, recall, f1]],
               columns=['Model','Accuracy','Precision','Recall','F1 Score'])


model_results = model_results.append(model_results2, ignore_index=True)


# Support Vector Machines - SVM (rbf)
from sklearn.svm import SVC
classifier = SVC(random_state=0, kernel='rbf')
classifier.fit(X_train, y_train)

# Predictions with Test set
y_pred = classifier.predict(X_test)


from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results2 = pd.DataFrame([['SVM(rbf)', accuracy, precision, recall, f1]],
               columns=['Model','Accuracy','Precision','Recall','F1 Score'])


model_results = model_results.append(model_results2, ignore_index=True)




# Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state=0, n_estimators =100,
                                    criterion='entropy')
classifier.fit(X_train, y_train)

# Predictions with Test set
y_pred = classifier.predict(X_test)


from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results2 = pd.DataFrame([['Random Forest, n=100', accuracy, precision, recall, f1]],
               columns=['Model','Accuracy','Precision','Recall','F1 Score'])


model_results = model_results.append(model_results2, ignore_index=True)


# K-Fold Cross Validation

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv = 10)

print('Random Forest Classifier Accuracy: %0.2f (+/- %0.2f)' % (accuracies.mean(), accuracies.std() * 2))

# From the results we got after running several models above, we 
# can confidently say that the best model for the job is Random Forest 
# with k-fold cross validatian as it gives us the highest model 
# accuracy.



# Parameter Tuning

# Grid Search
# Grid search performs different combinations of different parameters and 
# identifies which parameters affect which part of the model the most and 
# selects them to maximise our objective functiuon. This creates an optimized
# model.

# IN our model below, we are not giving our model infinete number of variables
# to choose from, but rather a handful of parameters to reduce the model load.
# In the production environment, we only use those parameters which are found to 
# be very effective by our model parameter tuning.


# Entropy parameters.
parameters = {"max_depth": [3, None],
              "max_features": [1, 5, 10],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10],
              "bootstrap": [True, False],
              "criterion": ["entropy"]}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=classifier, 
                           param_grid = parameters, 
                           scoring = "accuracy",
                           cv=10,
                           n_jobs = -1
                           )

# We are using estimator as classifier becauase the last model classifier we
# used was Random Forest with k-fold cross validation, therefore we want
# to further iterate on that model.


# Now we are going to fit the model
t0 = time.time()
grid_search = grid_search.fit(X_train , y_train)
t1 = time.time()
print('Total time taken: %0.2f seconds' % (t1 - t0))


rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters


# 2nd Variation - Parameters selected after seeing the results from
# the output of our first attempt in which we tested a whole bunch
# of different parameters.

parameters = {"max_depth": [None],
              "max_features": [3, 5, 7],
              'min_samples_split': [8, 12],
              'min_samples_leaf': [1, 2, 3],
              "bootstrap": [True],
              "criterion": ["entropy"]}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=classifier, 
                           param_grid = parameters, 
                           scoring = "accuracy",
                           cv=10,
                           n_jobs = -1
                           )

# We are using estimator as classifier becauase the last model classifier we
# used was Random Forest with k-fold cross validation, therefore we want
# to further iterate on that model.


# Now we are going to fit the model
t0 = time.time()
grid_search = grid_search.fit(X_train , y_train)
t1 = time.time()
print('Total time taken: %0.2f seconds' % (t1 - t0))


rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters

# Predicting Test Set
y_pred = grid_search.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results2 = pd.DataFrame([['Random Forest (n=100, GridSearchx2 + Entropy)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

model_results = model_results.append(model_results2, ignore_index = True)





# ROUND 1: Gini.
parameters = {"max_depth": [3, None],
              "max_features": [1, 5, 10],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10],
              "bootstrap": [True, False],
              "criterion": ["gini"]}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=classifier, 
                           param_grid = parameters, 
                           scoring = "accuracy",
                           cv=10,
                           n_jobs = -1
                           )

# We are using estimator as classifier becauase the last model classifier we
# used was Random Forest with k-fold cross validation, therefore we want
# to further iterate on that model.


# Now we are going to fit the model
t0 = time.time()
grid_search = grid_search.fit(X_train , y_train)
t1 = time.time()
print('Total time taken: %0.2f seconds' % (t1 - t0))


rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters


# 2nd Variation - Gini - Parameters selected after seeing the results from
# the output of our first attempt in which we tested a whole bunch
# of different parameters.

parameters = {"max_depth": [None],
              "max_features": [8, 10, 12],
              'min_samples_split': [2, 3, 4],
              'min_samples_leaf': [8,10,12],
              "bootstrap": [True],
              "criterion": ["gini"]}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=classifier, 
                           param_grid = parameters, 
                           scoring = "accuracy",
                           cv=10,
                           n_jobs = -1
                           )

# We are using estimator as classifier becauase the last model classifier we
# used was Random Forest with k-fold cross validation, therefore we want
# to further iterate on that model.


# Now we are going to fit the model
t0 = time.time()
grid_search = grid_search.fit(X_train , y_train)
t1 = time.time()
print('Total time taken: %0.2f seconds' % (t1 - t0))


rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters

# Predicting Test Set
y_pred = grid_search.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results2 = pd.DataFrame([['Random Forest (n=100, GridSearchx2 + Gini)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

model_results = model_results.append(model_results2, ignore_index = True)


# Finalizing our Model

final_results = pd.concat([y_test, users], axis=1).dropna()
final_results['predictions'] = y_pred
final_results = final_results[['entry_id', 'e_signed', 'predictions']]

























