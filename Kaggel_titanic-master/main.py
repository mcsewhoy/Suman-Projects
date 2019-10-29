import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set()


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

#df_train.head()
#df_train.info()

#df_test.head()
#df_test.info()


df_train_ml = df_train.copy()
df_test_ml = df_test.copy()


df_train_ml = pd.get_dummies(df_train_ml, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)
df_train_ml.drop(['PassengerId','Name','Ticket', 'Cabin'],axis=1,inplace=True)
df_train_ml.dropna(inplace=True)

passenger_id = df_test_ml['PassengerId']
df_test_ml = pd.get_dummies(df_test_ml, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)
df_test_ml.drop(['PassengerId','Name','Ticket', 'Cabin'],axis=1,inplace=True)
#df_test_ml.head(10)

#df_train_ml.info()
#df_test_ml.info()
#df_test_ml.head(10)


from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize

#scikitlean preprocessing for data normalisation
#normalisation   <- scales to a range of 0-1
#standardscaler    <-  scales to 0 mean, unit variance, with std deviation denominator
#minmaxscaler    <- scales features to a given range (0-1)
scaler = StandardScaler()
scaler1 = MinMaxScaler()
#scaler2 = normalize()

# for df_train_ml
scaler.fit(df_train_ml.drop('Survived',axis=1))
scaled_features = scaler.transform(df_train_ml.drop('Survived',axis=1))
df_train_ml_sc = pd.DataFrame(scaled_features, columns=df_train_ml.columns[:-1])

# for df_train_ml
scaler1.fit(df_train_ml.drop('Survived',axis=1))
scaled_features1 = scaler1.transform(df_train_ml.drop('Survived',axis=1))
df_train_ml_sc1 = pd.DataFrame(scaled_features1, columns=df_train_ml.columns[:-1])

# for df_train_ml
#scaler2.fit(df_train_ml.drop('Survived',axis=1))
#scaled_features2 = scaler2.transform(df_train_ml.drop('Survived',axis=1))
#df_train_ml_sc2 = pd.DataFrame(scaled_features2, columns=df_train_ml.columns[:-1])




# for df_test_ml
df_test_ml.fillna(df_test_ml.mean(), inplace=True)
# scaler.fit(df_test_ml)
scaled_features = scaler.transform(df_test_ml)
df_test_ml_sc = pd.DataFrame(scaled_features, columns=df_test_ml.columns)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train_ml.drop('Survived',axis=1), df_train_ml['Survived'], test_size=0.30, random_state=101)
X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(df_train_ml_sc, df_train_ml['Survived'], test_size=0.30, random_state=101)
X_train_sc1, X_test_sc1, y_train_sc1, y_test_sc1 = train_test_split(df_train_ml_sc1, df_train_ml['Survived'], test_size=0.30, random_state=101)
#X_train_sc2, X_test_sc2, y_train_sc2, y_test_sc2 = train_test_split(df_train_ml_sc2, df_train_ml['Survived'], test_size=0.30, random_state=101)

# unscaled
X_train_all = df_train_ml.drop('Survived',axis=1)
y_train_all = df_train_ml['Survived']
X_test_all = df_test_ml

# scaled
X_train_all_sc = df_train_ml_sc
y_train_all_sc = df_train_ml['Survived']
X_test_all_sc = df_test_ml_sc

#print(X_test_all)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
pred_logreg = logreg.predict(X_test)
#print(confusion_matrix(y_test, pred_logreg))
#print(classification_report(y_test, pred_logreg))
print("normal model accuracy:")
print(accuracy_score(y_test, pred_logreg))
#print(logreg.get_params())

logreg = LogisticRegression()
logreg.fit(X_train_sc,y_train_sc)
pred_logreg = logreg.predict(X_test_sc)
#print(confusion_matrix(y_test_sc, pred_logreg))
#print(classification_report(y_test_sc, pred_logreg))
print("Standard scalar model:")
print(accuracy_score(y_test_sc, pred_logreg))
#print(logreg.get_params())

logreg = LogisticRegression()
logreg.fit(X_train_sc1,y_train_sc1)
pred_logreg = logreg.predict(X_test_sc1)
#print(confusion_matrix(y_test_sc1, pred_logreg))
#print(classification_report(y_test_sc1, pred_logreg))
print("MinMax model accuracy:")
print(accuracy_score(y_test_sc1, pred_logreg))
#print(logreg.get_params())






print("Gridsearch Starting")
# Create regularization penalty space
penalty = ['l1', 'l2']

max_iter = [30, 50, 75, 100, 150]
# Create regularization hyperparameter space
C = np.logspace(0, 5, 20)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty, max_iter=max_iter)

# Create grid search using 5-fold cross validation
logr_gs = GridSearchCV(logreg, hyperparameters, cv=5, verbose=0, scoring='accuracy')

# Fit grid search
best_model = logr_gs.fit(X_train, y_train)

# View best hyperparameters
#print('Best Params:', best_model.best_estimator_.get_params())
print("Normal model Gridsearch:")
print(best_model.best_score_)
print(best_model.best_params_)

logreg = LogisticRegression(C=6.158, max_iter=30,penalty='l1')
logreg.fit(X_train,y_train)
pred_logreg = logreg.predict(X_test)
#print(confusion_matrix(y_test, pred_logreg))
#print(classification_report(y_test, pred_logreg))
print(accuracy_score(y_test, pred_logreg))

best_model_sc = logr_gs.fit(X_train_sc, y_train)
best_model_sc1 = logr_gs.fit(X_train_sc1, y_train)

# View best hyperparameters
#print('Best Params:', best_model_sc.best_estimator_.get_params())

print("StandardScaler model gridsearch:")
print(best_model_sc.best_score_)
print(best_model_sc.best_params_)
"""
logreg = LogisticRegression(C=6.158, max_iter=30,penalty='l1')
logreg.fit(X_train,y_train)
pred_logreg = logreg.predict(X_test)
print(confusion_matrix(y_test, pred_logreg))
print(classification_report(y_test, pred_logreg))
print(accuracy_score(y_test, pred_logreg))
"""


# View best hyperparameters
#print('Best Params:', best_model_sc1.best_estimator_.get_params())

print("Minmax model gridsearch:")
print(best_model_sc1.best_score_)
print(best_model_sc1.best_params_)

"""
logreg = LogisticRegression(C=6.158, max_iter=30,penalty='l1')
logreg.fit(X_train,y_train)
pred_logreg = logreg.predict(X_test)
print(confusion_matrix(y_test, pred_logreg))
print(classification_report(y_test, pred_logreg))
print(accuracy_score(y_test, pred_logreg))
"""

"""
logreg.fit(X_train_all, y_train_all)
pred_all_logreg = logreg.predict(X_test_all)
sub_logreg = pd.DataFrame()
sub_logreg['PassengerId'] = df_test['PassengerId']
sub_logreg['Survived'] = pred_all_logreg
sub_logreg.to_csv('output/logmodel.csv',index=False)


from sklearn.svm import SVC
svc = SVC(gamma = 0.01, C = 100)#, probability=True)
svc.fit(X_train_sc, y_train_sc)

pred_svc = svc.predict(X_test_sc)
print(confusion_matrix(y_test_sc, pred_svc))
print(classification_report(y_test_sc, pred_svc))
print(accuracy_score(y_test_sc, pred_svc))

logreg.fit(X_train_all_sc, y_train_all_sc)
pred_all_logreg = logreg.predict(X_test_all_sc)
sub_logreg = pd.DataFrame()
sub_logreg['PassengerId'] = df_test['PassengerId']
sub_logreg['Survived'] = pred_all_logreg
sub_logreg.to_csv('output/logmodel_svc.csv',index=False)
"""