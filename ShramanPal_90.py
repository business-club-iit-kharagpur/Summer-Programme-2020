#!/usr/bin/env python
# coding: utf-8

# In[620]:


import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
dataset = pd.read_csv("BCLUB train file.csv")                    #Reading Training Dataset
test_dataset = pd.read_csv("Testfile.csv")                       #Reading Test Dataset
Y = (dataset.iloc[:,-1])                                         
testY = test_dataset.iloc[:,-1]                                  
dataset = dataset.drop(columns=["Idx","SalePrice"])              #Removing Target Variable and Index from Training Set
test_dataset = test_dataset.drop(columns=["Idx","SalePrice"])    #Removing Target Variable and Index from Test Set
encoder = OneHotEncoder(categories="auto",sparse=False)          
dataset1 = dataset.copy()                                        #Making copy for another model
test_dataset1 = test_dataset.copy()
for i in ["Electrical","MasVnrType","MasVnrArea"]:               #Replacing the nan values with mode values
    x = dataset[[i]].fillna(dataset[i].value_counts().index[0])
    dataset1[[i]] = x
    y = test_dataset[[i]].fillna(test_dataset[i].value_counts().index[0])
    test_dataset1[[i]] = y
droppers = ["SaleType","YrSold","MiscVal","PoolArea","3SsnPorch","PavedDrive","KitchenAbvGr","OpenPorchSF","LandContour","WoodDeckSF","HalfBath","BsmtHalfBath","BsmtFullBath","LowQualFinSF","BsmtUnfSF","BsmtFinSF2","ExterCond","Condition2","Utilities","Alley","Street","LotFrontage"]
dataset1 = dataset1.drop(columns=droppers)                       #dropping certain Variables for Model 2
test_dataset1 = test_dataset1.drop(columns=droppers)


# In[ ]:





# In[621]:


def split(train,test):
    cat = train.copy()
    noncat = train.copy()
    testcat = test.copy()
    testnoncat = test.copy()
    for x in train: 
        if train.loc[:,x].dtype != object:
            cat = cat.drop(x,axis=1)                #categorical and non categorical division
        else:
            noncat = noncat.drop([x],axis=1)
    for y in test:
        if test.loc[:,y].dtype != object:
            testcat = testcat.drop([y],axis=1)
        else:
            testnoncat = testnoncat.drop([y],axis=1)

    print("Categorical Variables of train set: \n",cat.head())
    print("Non-Categorical Variables of train set: \n",noncat.head())
    print("Categorical Variables of test set: \n",testcat.head())
    print("Non-Categorical Variables of test set: \n",testnoncat.head())
    return cat,noncat,testcat,testnoncat


# In[622]:


def normalise(train,test):
    for x in train:
        train[[x]] = train[[x]].fillna(train[[x]].mean(),inplace = False)   #filling the nan values with mean
    
    for y in test:
        test[[y]] = test[[y]].fillna(train[[y]].mean(),inplace = False)   #filling the nan values with train mean
    
    norm_train = train.copy()
    norm_test = test.copy()
    for x in train:
        norm_train[x] = (train[x] - train[x].mean())/train[x].std()  #normalising values
    for y in test:
        norm_test[y] = (test[y] - train[y].mean())/train[y].std()    #normalising with train mean and std
    
    print("Normalised variables in train set: \n",norm_train.head())
    print("Normalised variables in test set: \n",norm_test.head())

    return pd.DataFrame(norm_train),pd.DataFrame(norm_test)


# In[623]:


def drop(train,test):                                    #dropping columns with any nan value
    nulls = train.isna()
    nan_cols = nulls.any()
    cols = train.columns[nan_cols].to_list()
    testnulls = test.isna()
    testnan_cols = testnulls.any()
    testcols = test.columns[testnan_cols].to_list()
    train = train.drop(columns = cols,axis = 1)
    test = test.drop(columns = cols,axis = 1)
    print("Shape of Train Set after dropping: \n",train.shape)
    print("Shape of Test Set after dropping: \n",test.shape)
    return pd.DataFrame(train),pd.DataFrame(test)


# In[624]:


def encode(train,test):                             #one hot encoding of both test and train set
    a = train.copy()
    b = test.copy()
    c = pd.concat([a,b],axis = 0)
    enc = encoder.fit(c)
    train = (enc.transform(a))
    test = (enc.transform(b))
    print("Shape of Categorical train set after encoding: \n",train.shape)
    print("Shape of Categorical test set after encoding: \n",test.shape)
    return pd.DataFrame(train),pd.DataFrame(test)


# In[625]:


def preprocess(data,testdata):                       #function to preprocess all data
    cat,num,testcat,testnum = split(data,testdata)
    norm_train,norm_test = normalise(num,testnum)
    dropcat,droptestcat = drop(cat,testcat)
    encoded,testencoded = encode(dropcat,droptestcat)
    X = pd.concat([norm_train,encoded],axis=1)
    testX = pd.concat([norm_test,testencoded],axis=1)
    print("Final preprocessed Train set head: \n",X.head())
    print("Final preprocessed train set shape: ",X.shape)
    print("Final preprocessed Test set head: \n",testX.head())
    print("Final preprocessed test set shape: ",testX.shape)
    return X,testX


# In[626]:


def param_initialise(nw):
    w = np.random.randn(nw,1)*1
    b = 0
    return w,b


# In[627]:


def forward_prop(X,w,b):
    Z = np.dot(w.T,X) + b
    return Z


# In[628]:


def compute_cost(Z,Y):
    m = len(Y[0])
    cost = (np.sum((Z - Y)**2)/m)**0.5
    return cost


# In[629]:


def update_param(X,Y,Z,w,b,lr):
    m = len(X[0])
    delta = np.multiply((Z - Y),X)
    dw = np.sum(delta,axis = 1,keepdims = True)
    db = np.sum(Z - Y)
    w = w - dw*lr/m
    b = b - db*lr/m
    return w,b


# In[634]:


def model(train,target,no_of_iters,learning_rate):    #model 
    x = (train.to_numpy())
    y = (target.to_numpy())
    X = x.T
    Y = y.T
    Y = Y.reshape(1,len(Y))
    cost = []
    nW = len(X)
    w,b = param_initialise(nW)
    for i in range(no_of_iters):
        Z = forward_prop(X,w,b)
        if i%50 == 0:
            cost.append(compute_cost(Z,Y))
        w,b = update_param(X,Y,Z,w,b,learning_rate)
    print("Final cost: ",cost[-1])
    return cost,w,b


# In[635]:


def evaluate(x,y,w,b):                               #MAPE evaluation
    X = x.to_numpy()
    Y = y.to_numpy()
    X = X.T
    Y = Y.reshape(1,len(Y))
    Z = np.dot(w.T,X) + b
    delta = (Y - Z)/Y
    m = len(Y[0])
    mape = np.sum(abs(delta))/m
    return mape


# In[636]:


X,testX = preprocess(dataset,test_dataset)
X1,testX1 = preprocess(dataset1,test_dataset1)
iters = [12000]
learning_rate = [0.01]
for i in iters:
    for l in learning_rate:
        cost,W,b = model(X,Y,i,l)          #apply model
        mape = evaluate(X,Y,W,b)
        print("Mape Score of Train Set Without Dropping: ",mape)
        plt.plot(cost)
        plt.xlabel("Epochs")
        plt.ylabel("Cost")
        plt.title("WITHOUT DROPPING ANY VARIABLES")
        plt.show()
for i in iters:
    for l in learning_rate:
        cost,W1,b1 = model(X1,Y,i,l)        #model that has dropped variables
        mape = evaluate(X1,Y,W1,b1)
        print("Mape Score of Train Set With Dropping: ",mape)
        plt.plot(cost)
        plt.xlabel("Epochs")
        plt.ylabel("Cost")
        plt.title("WITH DROPPING SPECIFIC VARIABLES")
        plt.show()


# In[637]:


mape = evaluate(testX,testY,W,b)
print("MAPE Score of Test Set Without Dropping: ",mape)
mape = evaluate(testX1,testY,W1,b1)
print("MAPE Score of Test Set With Dropping: ",mape)


# In[ ]:




