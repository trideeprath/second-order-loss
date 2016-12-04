# -*- coding: utf-8 -*-
"""
Created on Sat Dec 03 11:54:30 2016

@author: nkim30
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

def accuracy(w, b, test_x, test_y):
    correct = 0
    X_incorrect = []
    for x, y in zip(test_x, test_y):
        y_pred = w.dot(x) + b
        predicted_class= np.argmax(np.array(y_pred), axis=1)
        if predicted_class == y:
            correct += 1
    return float(correct*100)/float(len(test_x))


def cross_entropy_run(create_data=False, plot_fig=False,step = 0.001, second_ord = "vanilla", consider_reg=True):
    if consider_reg is True:
        reg_str = ", With L2 regularizattion"
    else:
        reg_str = ", Without L2 regularization"
    print("****** Cross entropy optimization started with second order as " + second_ord + " *******" + reg_str)
    start_time = time.clock()
    X_train = pickle.load(open("data/train_x.pkl", "rb"))
    Y_train = pickle.load(open("data/train_y.pkl", "rb"))
    # initialize parameters randomly
    W = 0.01 * np.random.randn(2,2)
    b = np.zeros((1,2))
    loss0 = np.inf
    stop = 0.000001
    diff = np.inf

    # some hyperparameters
    step_size = step
    reg = 1e-3 # regularization strength

    # gradient descent loop
    num_examples = X_train.shape[0]

    #Adagrad variables
    cache_W = 0
    cache_B = 0
    #Rmsprop variables
    decay_rate = 0.9
    # Params for Adam
    beta1 = 0.9
    beta2 = 0.999
    m_W = 0
    v_W = 0
    m_B = 0
    v_B = 0


    i = 0
    while np.abs(diff) > stop:
        i += 1
        # evaluate class scores, [N x K]
        scores = np.dot(X_train, W) + b

        # compute the class probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

        # compute the loss: average cross-entropy loss and regularization
        corect_logprobs = -np.log(probs[range(num_examples),Y_train])
        data_loss = np.sum(corect_logprobs)/num_examples
        if consider_reg is True:
            reg_loss = 0.5*reg*np.sum(W*W)
            loss = data_loss + reg_loss
        else:
            loss = data_loss

        diff = loss0 - loss
        loss0 = loss

        if i % 10 == 0:
            #print("iteration %d: loss %f" + str((i, loss)))
            pass
        # compute the gradient on scores
        dscores = probs
        dscores[range(num_examples),Y_train] -= 1
        dscores /= num_examples

        # backpropate the gradient to the parameters (W,b)
        dW = np.dot(X_train.T, dscores)
        db = np.sum(dscores, axis=0, keepdims=True)

        if consider_reg is True:
            dW += reg*W # regularization gradient

        # perform a parameter update
        if second_ord == "vanilla":
            W += -step_size * dW
            b += -step_size * db
        elif second_ord == "adagrad":
            cache_W = cache_W + dW ** 2
            W = W - step * dW / (np.sqrt(cache_W) + 0.00001)
            b += -step_size * db
            #cache_B = cache_B + db**2
            #b = b - step * db / (np.sqrt(cache_B) + 0.00001)
        elif second_ord == "rmsprop":
            cache_W = decay_rate * cache_W + (1 - decay_rate) * (dW ** 2)
            W =  W - step * dW / (np.sqrt(cache_W) + 0.00001)
            b += -step_size * db
            #cache_B = decay_rate * cache_B + (1 - decay_rate) * (db ** 2)
            #b = b - step * db / (np.sqrt(cache_B) + 0.00001)
        elif second_ord == "adam":
            m_W = (beta1 * m_W) + (1 - beta1) * dW
            v_W = (beta2 * v_W) + (1 - beta2) * (dW ** 2)
            W = W - step * m_W / (np.sqrt(v_W) + 0.00000001)
            b += -step_size * db
            #m_B = (beta1 * m_B) + (1 - beta1) * db
            #v_B = (beta2 * v_B) + (1 - beta2) * (db ** 2)
            #b = b - step * m_B / (np.sqrt(v_B) + 0.00000001)

    time_taken = time.clock() - start_time
    print(time.clock() - start_time, "seconds")

    X_test = pickle.load(open("data/test_x.pkl", "rb"))
    Y_test = pickle.load(open("data/test_y.pkl", "rb"))
    acc = accuracy(W,b, X_test, Y_test)
    print("Accuracy : " + str(acc) + "%")
    print("Iteration : " + str(i))

    if plot_fig is True:
        plot_points(X_train, Y_train, W, b)



def plot_points(X, y, W, b):
    # plot the resulting classifier
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    # fig.savefig('spiral_linear.png')
    plt.show()

