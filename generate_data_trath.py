from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import pickle
import matplotlib.pyplot as plt

def create_blob(out_folder="data/", samples =1000, features=2, std=0.70, plot_fig = False):
    print("*******Generating data*********")
    X, Y = make_blobs(n_samples=samples, centers=2, n_features=features, random_state=0, cluster_std=std)
    Y_hinge = np.array(list(map(lambda x: -1 if x == 0 else 1, Y)))  # Make Y as +1,-1
    train_data_count = int(0.70* samples)
    train_X, train_Y, train_Y_hinge = X[0:train_data_count], Y[0:train_data_count], Y_hinge[0:train_data_count]  # 70 percent as training
    test_X, test_Y, test_Y_hinge = X[train_data_count:], Y[train_data_count:], Y_hinge[train_data_count:]
    # Saving training data
    pickle.dump(train_X, open(out_folder + "train_x.pkl", 'wb'))
    pickle.dump(train_Y, open(out_folder + "train_y.pkl", 'wb'))
    pickle.dump(train_Y_hinge, open(out_folder + "train_y_hinge.pkl", 'wb'))
    # Saving testing data
    pickle.dump(test_X, open(out_folder+ "test_x.pkl", "wb"))
    pickle.dump(test_Y, open(out_folder + "test_y.pkl", "wb"))
    pickle.dump(test_Y_hinge, open(out_folder + "test_y_hinge.pkl", "wb"))
    print("*******Data Generation complete*********")
    if plot_fig is True:
        f1 = plt.figure(1)
        plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, cmap=plt.cm.Paired)
        plt.title("Training data")
        f1.show()
        f2 = plt.figure(2)
        plt.scatter(test_X[:, 0], test_X[:, 1], c=test_Y, cmap=plt.cm.Paired)
        plt.title("Testing data")
        f2.show()
        plt.show()
