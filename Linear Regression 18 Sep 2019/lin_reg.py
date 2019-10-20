import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

df = pd.read_csv('ex1data1.csv')
#print(df.head(5))

m = len(df)
x0 = np.ones(m)
X = np.array([x0, df['input']]).T
print(X.shape)


Y = np.array(df['output']).T  #note np.array with a df vertical vector makes a horizontal vector.
print(Y.shape)

W = np.array(np.zeros(X.shape[1]))
print(W.shape)

alpha = 0.01
itr = 1000000
cost = [0] * (itr+1)
lamb = 0.1

start_time = time.time()

def cost_function(X, Y, W):
    J = (np.sum((X.dot(W) - Y)**2) + lamb * (W.dot(W.T)))/(2*m)   # we add regularisation term here, + lambda * sum of Weights squared
    return J
cost[0] = cost_function(X,Y,W)



def grad_dec(X,Y,W,alpha):
    H = X.dot(W)
    loss = H - Y
    grad = (X.T.dot(loss)+ lamb*W)/m
    W = W - alpha * grad
    C = cost_function(X,Y,W)
    return W,C

# print(grad_dec(X,Y,W,alpha))

i = 0

while True:
    W, cost[i+1] = grad_dec(X,Y,W,alpha)
    i = i+1
    if cost[i-1]-cost[i] < 0.0000001:
        break

print ("__ %s seconds __" % (time.time() - start_time))
#print(cost[:i+1])

plt.plot(range(i+1),cost[:i+1])
plt.show()

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
