import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import pickle
import time
import numpy as np


def hinge_loss(w,x,y,consider_reg):
    """ evaluates hinge loss and its gradient at w

    rows of x are data points
    y is a vector of labels
    """
    loss, grad1 = 0, 0
    reg_loss = 0
    reg_grad = 0
    c = 0.001
    for (x_,y_) in zip(x,y):
        v = y_*np.dot(w,x_)
        loss += max(0, 1-v )
        if v >1 :
            grad1 = grad1
        else:
            grad1 = grad1 - (y_*x_)

    return (loss, grad1)


def grad_descent(x, y, w, step, second_ord= "vanilla", consider_reg=False, stop= None):
    #grad = np.inf
    timing_data = []
    ws = np.zeros((2,0))
    ws = np.hstack((ws,w.reshape(2,1)))
    diff = np.inf
    loss0 = np.inf
    cache = 0
    decay_rate = 0.9 # Used in RmsProp
    #Params for Adam
    beta1 = 0.9
    beta2 = 0.999
    m = 0
    v = 0
    count = 0
    t0 = time.time()
    iteration_count = 0
    c = 0.001
    reg_grad = 0
    loss_list = []
    while np.abs(diff) > stop:
        iteration_count +=1
        loss, grad = hinge_loss(w, x, y, consider_reg)
        #print(loss)

        if consider_reg is True:
            reg_loss = c * (np.linalg.norm(w)**2)
            reg_grad = c * (np.sum(w))
            loss = loss + reg_loss

        diff = loss0 - loss
        loss0 = loss

        if second_ord == "vanilla":
            w = w - step * grad + reg_grad
        elif second_ord == "adagrad":
            cache = cache + grad**2
            w = w - step * grad/(np.sqrt(cache)+ 0.00001) + reg_grad
        elif second_ord == "rmsprop":
            cache = decay_rate * cache + (1- decay_rate) * (grad**2)
            w = w - step * grad/(np.sqrt(cache) + 0.00001) + reg_grad
        elif second_ord == "adam":
            m = (beta1 * m) + (1-beta1) * grad
            v = (beta2 * v) + (1-beta2) * (grad**2)
            w = w - step * m /(np.sqrt(v) + 0.00000001) + reg_grad
        ws = np.hstack((ws,w.reshape((2,1))))
        loss_list.append((iteration_count, loss))

    return np.sum(ws,1)/np.size(ws,1), iteration_count, loss_list


def hinge_run(create_data=False, plot_fig=False, plot_iteration=False, step=0.001, second_ord="vanilla", consider_reg=False, stop=None):

    if consider_reg is True:
        reg_str = ", With L2 regularizattion"
    else:
        reg_str = ", Without L2 regularization"
    banner = "Hinge loss optimization started with second order as " + second_ord + reg_str
    print("******  " + banner + "  *******")
    start_time = time.clock()

    if create_data is True:
        X_train, Y_train = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=0, cluster_std=0.70)
        Y_train = np.array(list(map(lambda x : -1 if x==0 else 1, Y_train)))
    else:
        X_train = pickle.load(open("data/train_x.pkl", "rb"))
        Y_train = pickle.load(open("data/train_y_hinge.pkl", "rb"))

    w, iteration_count, loss_list = grad_descent(X_train, Y_train, np.array((0, 0)), step= step, second_ord=second_ord, consider_reg= consider_reg,stop=stop )
    #print(len(loss_list), loss_list[0:10])
    time_taken = time.clock() - start_time
    print(time.clock() - start_time, "seconds")

    # Accuracy
    X_test = pickle.load(open("data/test_x.pkl", "rb"))
    Y_test = pickle.load(open("data/test_y_hinge.pkl", "rb"))
    acc = accuracy(w, X_test, Y_test)
    print("Accuracy : " + str(acc) + "%")
    print("Iterations : " + str(iteration_count))
    #Plot of iteration and loss
    if plot_iteration is True:
        iteration = []
        loss_value = []
        for x, y in loss_list:
            iteration.append(x)
            loss_value.append(y)

        plt.plot(iteration, loss_value)
        plt.title(banner)
        plt.show()


    # Plot points and decision surface
    if plot_fig is True:
        plot_points(X_train, Y_train, w)
    return time_taken, acc, iteration_count


def accuracy(w, test_x, test_y):
    correct = 0
    X_incorrect = []
    for x, y in zip(test_x, test_y):
        y_pred = w.dot(x)
        if y_pred * y > 0 :
            correct += 1
    return float(correct*100)/float(len(test_x))


def plot_points(x, y, w):
    plt.figure()
    # Features
    x1, x2 = x[:,0], x[:,1]
    x1_min, x1_max = np.min(x1)*.7, np.max(x1)*1.3 #Scales
    x2_min, x2_max = np.min(x2)*.7, np.max(x2)*1.3 #Scales
    gridpoints = 2000
    x1s = np.linspace(x1_min, x1_max, gridpoints)
    x2s = np.linspace(x2_min, x2_max, gridpoints)
    gridx1, gridx2 = np.meshgrid(x1s,x2s)
    grid_pts = np.c_[gridx1.ravel(), gridx2.ravel()]
    predictions = np.array([np.sign(np.dot(w,x_)) for x_ in grid_pts]).reshape((gridpoints,gridpoints))
    plt.contourf(gridx1, gridx2, predictions, cmap=plt.cm.Paired)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)
    plt.show()
