# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

df = pd.read_csv("iris.csv")

X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:,:4], df.iloc[:,4:5], test_size=0.2, random_state=99)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(df.shape)

reg = LogisticRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)
y_train = np.array(y_train.iloc[:,0].tolist())
print(confusion_matrix(y_test, y_pred)) # confusion Matrix, Class 1,2,3 act vs predicted
print(classification_report(y_test, y_pred)) # Precision / Recall
print(accuracy_score(y_test, y_pred))# we can also add an accuracy score



""" SVM Model"""

print("SVM Model")

"""https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html"""
from sklearn.svm import SVC  # Support Vector Classifier
clf = SVC(C=100, gamma=0.01) # C is the error "weighting/scaling" and gamma is the regularization
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred)) # confusion Matrix, Class 1,2,3 act vs predicted
print(classification_report(y_test, y_pred)) # Precision / Recall
print(accuracy_score(y_test, y_pred))# we can also add an accuracy score



""" Decision Tree"""
print("Decision Tree")
from sklearn.tree import DecisionTreeClassifier
dectree = DecisionTreeClassifier(random_state=0)
dectree.fit(X_train, y_train)
y_pred = dectree.predict(X_test)
print(confusion_matrix(y_test, y_pred)) # confusion Matrix, Class 1,2,3 act vs predicted
print(classification_report(y_test, y_pred)) # Precision / Recall
print(accuracy_score(y_test, y_pred))# we can also add an accuracy score


""" Random Forest"""



""" KNN"""


