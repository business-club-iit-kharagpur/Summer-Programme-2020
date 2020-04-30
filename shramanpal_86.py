#!/usr/bin/env python
# coding: utf-8

# In[81]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
orig_dataset = pd.read_csv("BCLUB train file.csv")
orig_dataset_test = pd.read_csv("Testfile.csv")
data = orig_dataset.copy()
test_data = orig_dataset_test.copy()


# In[ ]:





# In[82]:


def normalise(data):
    X_1 = data[["MSSubClass","LotArea","OverallQual","OverallCond","TotalBsmtSF","GrLivArea","BedroomAbvGr","KitchenAbvGr","GarageCars"]]
    Y_1 = data[["SalePrice"]]
    for x in X_1:
        X_1[[x]].fillna(X_1[[x]].mean(),inplace = True)
    Xnorm_1 = X_1.copy()
    for x in X_1:
        Xnorm_1[x] = (X_1[x] - X_1[x].mean())/X_1[x].std()
    Ynorm_1 = (Y_1 - Y_1.mean())/Y_1.std() 
    return Xnorm_1,Y_1


# In[83]:


def param_initialise(m):
    W = np.random.randn(m,1)*10
    b = 0
    print(W,b)
    return W,b


# In[84]:


def forward_prop(X,W,b):
    Z = np.dot(X,W)+b
    return Z


# In[85]:


def compute_cost(Y,Z):
    m = len(Z)
    cost = np.sum((Y - Z)**2)/m
    return cost


# In[86]:


def update_param(X,Y,Z,W,b,learning_rate):
    delta = np.multiply((Z - Y),X)
    update = np.sum(delta, axis = 0,keepdims = True)/len(Y)
    W -= learning_rate*(update.T)/len(Y)
    b -= learning_rate*(np.sum((Z - Y)))/len(Y)
    return W,b


# In[87]:


def model(X,Y,iters,learning_rate):
    n_W = len(X[0])
    cost = []
    W,b = param_initialise(n_W)
    for i in range(1,iters):
        Z = forward_prop(X,W,b)
        if i%50 == 0:
            cost.append(compute_cost(Y,Z))
        W,b = update_param(X,Y,Z,W,b,learning_rate)
    return cost,W,b


# In[88]:


def evaluate(X,Y,W,b):
    Z = np.dot(X,W) + b
    delta = (Y - Z)/Y
    mape = np.sum(abs(delta))/len(Y)
    return mape


# In[89]:


no_iters = 10000
learning_rate = 0.8
X_1,Y_1 = normalise(data)
X_test1,Y_test1 = normalise(test_data)
X = X_1.to_numpy()
Y = Y_1.to_numpy()
X_test = X_test1.to_numpy()
Y_test = Y_test1.to_numpy()
cost,W,b = model(X,Y,no_iters,learning_rate)
z = np.dot(X,W)+b
mape = evaluate(X,Y,W,b)
print("MAPE OF TRAIN SET IS: "+str(mape))
mape_test = evaluate(X_test,Y_test,W,b)
print("MAPE OF TEST SET IS: " + str(mape_test))
plt.plot(cost)
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Cost Curve")


# In[ ]:





# In[ ]:




