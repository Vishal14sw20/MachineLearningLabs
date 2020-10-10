import urllib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt




df_train = pd.read_csv('SPECT.train',header=None)
df_test = pd.read_csv('SPECT.test',header=None)

train = df_train.as_matrix()
test = df_test.as_matrix()

y_train = train[:,0]
X_train = train[:,1:]
y_test = test[:,0]
X_test = test[:,1:]

def changeLabels(y_train,y_test):
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1
    return y_train, y_test

def learn_reg_ERM(X,y,lbda):
    max_iter = 100
    e  = 0.001
    alpha = 1.

    w = np.random.randn(X.shape[1]);
    for k in np.arange(max_iter):
        h = np.dot(X,w)
        l,lg = loss(h, y)
        #print('loss: {}'.format(np.mean(l)))
        r,rg = reg(w, lbda)
        g = np.dot(X.T,lg) + rg 
        if (k > 0):
            alpha = alpha * (np.dot(g_old.T,g_old))/(np.dot((g_old - g).T,g_old))
        w = w - alpha * g
        if (np.linalg.norm(alpha * g) < e):
            break
        g_old = g
    return w



def loss(h, y):
    # its hinge loss
    l=np.maximum(0,1-h*y)
    g=np.where(l>0,-y,0)
    return l, g


# ### Exercise 3

def reg(w, lbda):
    #INSERT CODE HERE#
    r = (lbda / 2) * np.matmul(w.transpose(), w)
    g = lbda * w
    return r, g


def predict(w, X):
    #INSERT CODE HERE#
    preds = np.matmul(X, w)
    preds = np.where(preds>0,1,-1)
    return preds

def accuracy(y_actual,y_predicted):
    sum =(y_actual==y_predicted).sum()
    acc = sum/y_actual.shape[0]
    return acc

y_train, y_test= changeLabels(y_train,y_test)
w=learn_reg_ERM(X_train,y_train,5)
y_test_predicted = predict(w,X_test)
y_train_predicted = predict(w,X_train)
test_acc = accuracy(y_test,y_test_predicted)
train_acc = accuracy(y_train,y_train_predicted)
print(train_acc)
print(test_acc)


clf = RandomForestClassifier()
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))
