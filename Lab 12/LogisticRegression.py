import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

np.random.seed(1)
num_observations = 2000

x_class1 = np.random.multivariate_normal([0, 0], [[1, .25],[.25, 1]], num_observations)
x_class2 = np.random.multivariate_normal([1, 4], [[1, .25],[.25, 1]], num_observations)

# Training data:
X_train = np.vstack((x_class1, x_class2)).astype(np.float32)
y_train = np.hstack((np.zeros(num_observations), np.ones(num_observations))) # labels are 0, 1

plt.figure(figsize=(10,8))
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='green', label='label 1')
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='red', label='label 0')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title('Sampled data')
plt.legend()
y_train.shape


def logistic_regression(X, y, num_steps, learning_rate, add_intercept):
    # X: n x d matrix of instances
    # y: vector of n labels

    if add_intercept:
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))

    weights = np.zeros(X.shape[1])

    for step in range(num_steps):
        scores = np.dot(X, weights)
        predictions = sigmoid(scores)

        gradient = log_likelihood_gradient(X, y, weights)
        weights -= learning_rate * gradient

        if step % 10000 == 0:
            print(log_likelihood(X, y, weights))

    return weights

def sigmoid(scores):
    ####################
    # INSERT CODE HERE #
    s=1/(1+np.exp(-scores))
    ####################
    return s

def log_likelihood(X, y, weights):
    ####################
    # INSERT CODE HERE #
    a=y*np.matmul(X,weights)
    b= np.log(1+np.exp(np.matmul(X,weights)))

    ####################
    return -(a-b).sum()

def log_likelihood_gradient(X, y, weights):
    ####################
    # INSERT CODE HERE #
    a= np.matmul(X,weights)
    b= y-sigmoid(a)
    gradient = np.matmul(X.transpose(), b)
    ####################
    return -gradient

wieghts=logistic_regression(X_train,y_train,30000,5*np.exp(-5),0)
print(wieghts)

predictions = np.matmul(X_train,wieghts)
predictions=np.where(predictions>=0.5,1,0)
print(accuracy_score(predictions,y_train))