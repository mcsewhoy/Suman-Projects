import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math

df = pd.read_csv('iris.csv')
#print(df.head(5))

m = len(df)
x0 = np.ones(m)
X = np.array([df['Input']]).T
print(X.shape)

df['Outputhot'] = (df['output'] == "Iris-virginica")*1
Y = np.array(df['Outputhot']).T  #note np.array with a df vertical vector makes a horizontal vector.
print(Y.shape)

W = np.array(np.zeros(X.shape[1]))
print(W.shape)


alpha = .001
itr = 1000000
cost = [0] * (itr+1)
lamb = .1

start_time = time.time()

def cost_function(X, Y, W):
       #print(W)
       H = 1/(1+np.exp(X.dot(W.T)))
       J = (np.sum(-Y*np.log(H) - (1-Y)*np.log(1-H)))/m + (lamb/(2*W.shape[0])) * (W.dot(W.T))
    #  J = (np.sum((X.dot(W) - Y)**2) + lamb * (W.dot(W.T)))/(2*m)   # we add regularisation term here, + lambda * sum of Weights squared
       print(J)
       return J
cost[0] = cost_function(X,Y,W)



def grad_dec(X,Y,W,alpha):
    H = 1/(1+np.exp(X.dot(W.T)))
    loss = (-Y*np.log(H) - (1-Y)*np.log(1-H)) + (lamb/(2*W.shape[0])) * (W.dot(W.T))
    grad = np.sum(loss.dot(X))/m + (lamb/(2*W.shape[0]))*W       #<- Cant solve this 
    print(grad)
    W = W - alpha * grad
    C = cost_function(X,Y,W)
    return W,C

# print(grad_dec(X,Y,W,alpha))

i = 0

while True:
    W, cost[i+1] = grad_dec(X,Y,W,alpha)
    print(W)
    i = i+1
    if cost[i-1]-cost[i] < 0.0000001:
        break



print ("__ %s seconds __" % (time.time() - start_time))
print(cost[:i+1])

plt.plot(range(i+1),cost[:i+1])
plt.show()

"""
from sklearn.linear_model import LinearRegression
start_time = time.time()
reg = LinearRegression().fit(X, Y)
Y_pred = reg.predict(X)

print ("__ %s seconds __" % (time.time() - start_time))
# compare our own linear regression vs scikitlearn model)
print(W, cost[i])
print(reg.coef_, np.sum((Y_pred - Y)**2/(2*m)))

# Homwork, make changes to variables and see how it affects the output
# Implement a logistic regression.
"""
"""
# Homework: Add a Scikitlearn Logistic regression 

from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV

# Create logistic regression
logr = LogisticRegression()

# Create regularization penalty space
penalty = ['l1', 'l2']

max_iter = [300, 500, 700]
# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty, max_iter=max_iter)

# Create grid search using 5-fold cross validation
logr_gs = GridSearchCV(logr, hyperparameters, cv=5, verbose=0)

# Fit grid search
best_model = logr_gs.fit(train_x, train_y)'

# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

# Predict target vector
best_model.score(test_x, test_y)
"""