# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from warnings import simplefilter
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

# ignore all warnings
simplefilter(action='ignore')


# In[2]:


def loaddata():
    df = pd.read_csv(r'C:\desi_representations_subset.csv')
    # load the data
    columns_to_select = ['feat_pca_{}'.format(i) for i in range(20)] + ['classification_label']
    X = df[columns_to_select]
    return X

# In[77]:


def fit1(df):
    X = df[['feat_pca_{}'.format(i) for i in range(20)]]
    y = df['classification_label']
    clf = LogisticRegression()
    clf.fit(X, y)
    return clf


# In[5]:


def randomapp(times, df, r):
    label = df.sample(n=100)
    pool = df.drop(label.index)
    for i in range(times):
        new = pool.sample(n=1)
        label = pd.concat([label, new])
        pool = pool.drop(new.index)
        print(fit(label).score(loaddata()[['feat_pca_{}'.format(i) for i in range(20)]],
                               loaddata()['classification_label']))
    return label, pool


# In[6]:


def randomapp1(train=None, test=None, times=100, init=50, loop=1):
    # return test scores of random append.
    # times is # of iterations default = 100
    # train&test is the train and test data
    # init is the initial pool of data
    # loop is the number of data points in each batch of iteration (not used in the current function)
    # Check if train and test data are provided
    if train is None or test is None:
        raise ValueError("Both train and test data must be provided.")

    # Debug statement to check the type of init
    if not isinstance(init, int):
        raise TypeError(f"Expected init to be an integer, but got {type(init)} instead.")

    if not isinstance(times, int):
        raise TypeError(f"Expected times to be an integer, but got {type(times)} instead.")

    if not isinstance(loop, int):
        raise TypeError(f"Expected loop to be an integer, but got {type(loop)} instead.")

    label = train.sample(n=init)
    pool = train.drop(label.index)
    trace = []
    for i in range(times):
        new = pool.sample(n=loop)
        label = pd.concat([label, new])
        pool = pool.drop(new.index)
        trace.append(
            fit1(label).score(test[['feat_pca_{}'.format(i) for i in range(20)]], test['classification_label']))
        # print(fit1(label).score(test[['feat_pca_{}'.format(i) for i in range(20)]], test['classification_label']))
    # return label, pool
    return trace


# In[79]:


def ambiguous(train, test, times=100, init=50, loop=1):
    # return test scores of diverse append.
    # times is # of iterations default = 100
    # train&test is the train and test data
    # init is the initial pool of data
    # loop is the number of data points in each batch of iteration (not used in the current function)
    # Check if train and test data are provided
    if train is None or test is None:
        raise ValueError("Both train and test data must be provided.")

    # Debug statement to check the type of init
    if not isinstance(init, int):
        raise TypeError(f"Expected init to be an integer, but got {type(init)} instead.")

    if not isinstance(times, int):
        raise TypeError(f"Expected times to be an integer, but got {type(times)} instead.")

    if not isinstance(loop, int):
        raise TypeError(f"Expected loop to be an integer, but got {type(loop)} instead.")


    label = train.sample(n=init)
    pool = train.drop(label.index).reset_index(drop=True)
    number = []
    for i in range(times):
        model = fit1(label)
        diff = []
        y = pool[['feat_pca_{}'.format(i) for i in range(20)]]
        for j in range(len(pool)):
            predic = model.predict_proba([y.iloc[j]])
            diff.append(abs(0.5 - predic[0, 0]))
        data_array = np.array(diff)
        sorted_indices = np.argsort(data_array)
        n = loop
        top_n_indices = sorted_indices[:n]
        label = pd.concat([label, pool.iloc[top_n_indices]], ignore_index=True)
        pool = pool.drop(top_n_indices).reset_index(drop=True)
        number.append(
            fit1(label).score(test[['feat_pca_{}'.format(i) for i in range(20)]], test['classification_label']))
        # print(fit1(label).score(test[['feat_pca_{}'.format(i) for i in range(20)]], test['classification_label']))
    # return label, pool
    return number


# In[66]:
def diverse(train=None, test=None, times=100, init=50, loop=1):
    # return test scores of diverse append.
    # times is # of iterations default = 100
    # train&test is the train and test data
    # init is the initial pool of data
    # loop is the number of data points in each batch of iteration (not used in the current function)
    # Check if train and test data are provided
    if train is None or test is None:
        raise ValueError("Both train and test data must be provided.")

    # Debug statement to check the type of init
    if not isinstance(init, int):
        raise TypeError(f"Expected init to be an integer, but got {type(init)} instead.")

    if not isinstance(times, int):
        raise TypeError(f"Expected times to be an integer, but got {type(times)} instead.")

    if not isinstance(loop, int):
        raise TypeError(f"Expected loop to be an integer, but got {type(loop)} instead.")

    label = train.sample(n=init)
    pool = train.drop(label.index).reset_index(drop=True)
    number = []
    for i in range(times):
        distances = cdist(label, pool)
        minavgdis = []
        for i in range(distances.shape[1]):
            minavgdis.append(min(distances[:, i]))
        data_array = np.array(minavgdis)
        sorted_indices = np.argsort(data_array)
        n = loop
        top_n_indices = sorted_indices[-n:]
        label = pd.concat([label, pool.iloc[top_n_indices]], ignore_index=True)
        pool = pool.drop(top_n_indices).reset_index(drop=True)
        number.append(
            fit1(label).score(test[['feat_pca_{}'.format(i) for i in range(20)]], test['classification_label']))
        # print(fit1(label).score(test[['feat_pca_{}'.format(i) for i in range(20)]], test['classification_label']))
    # return label, pool
    return number


# In[80]:
def plot(times=100, init=50, loop=1, regression_method="L"):
    test = loaddata().sample(n=2000, random_state=1)
    train = loaddata().drop(test.index).reset_index(drop=True)
    test.reset_index(drop=True)
    x = []  # X-axis values
    for i in range(times): x.append(i)
    x = np.array(x)
    if regression_method == "L":
        clf = LogisticRegression()
    elif regression_method == "M":
        clf = MLPClassifier(hidden_layer_sizes=(50,), learning_rate_init=0.01)
    X = train.iloc[:, :(train.shape[1] - 1)]
    y = train.iloc[:, -1]
    y1 = abs(np.log(1 - np.array(randomapp(train, test, times = times, loop=loop, init = init, regression_method=regression_method))))  # Y-axis values for Line 1
    y2 = abs(np.log(1 - np.array(ambiguous(train, test, times = times, loop=loop, init = init, regression_method=regression_method))))  # Y-axis values for Line 2
    y3 = abs(np.log(1 - np.array(diverse_tree(train, test, times = times, loop=loop, init = init, regression_method=regression_method))))  # Y-axis values for Line 3
    plt.axhline(y=abs(math.log(1 - clf.fit(X,y).score(test[['feat_pca_{}'.format(i) for i in range(20)]], test['classification_label']))), color='r', linestyle='-')
    plt.plot(x, y1, label='random', marker='o')  # Plot Line 1
    plt.plot(x, y2, label='ambiguous', marker='s')  # Plot Line 2
    plt.plot(x, y3, label='diverse', marker='s')  # Plot Line 3
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.title('Three Lines with Different Methods')
    plt.legend()
    return plt.show()




# In[66]:
def randomapp(train=None, test=None, times=100, init=50, loop=1, regression_method="L"):
    # return test scores of random append.
    # times is # of iterations default = 100
    # train&test is the train and test data
    # init is the initial pool of data
    # loop is the number of data points in each batch of iteration (not used in the current function)
    # Check if train and test data are provided
    if train is None or test is None:
        raise ValueError("Both train and test data must be provided.")

    # Debug statement to check the type of init
    if not isinstance(init, int):
        raise TypeError(f"Expected init to be an integer, but got {type(init)} instead.")

    if not isinstance(times, int):
        raise TypeError(f"Expected times to be an integer, but got {type(times)} instead.")

    if not isinstance(loop, int):
        raise TypeError(f"Expected loop to be an integer, but got {type(loop)} instead.")

    if regression_method == "L":
        clf = LogisticRegression()
    elif regression_method == "M":
        clf = MLPClassifier(hidden_layer_sizes=(50,), learning_rate_init=0.01)

    label = train.sample(n=init)
    pool = train.drop(label.index)
    trace = []
    for i in range(times):
        new = pool.sample(n=loop)
        label = pd.concat([label, new])
        pool = pool.drop(new.index)
        X = label.iloc[:, :(label.shape[1]-1)]
        y = label.iloc[:, -1]
        trace.append(
            clf.fit(X,y).score(test.iloc[:, :(test.shape[1]-1)], test.iloc[:, -1]))
        # print(fit1(label).score(test[['feat_pca_{}'.format(i) for i in range(20)]], test['classification_label']))
    # return label, pool
    return trace

def ambiguous(train, test, times=100, init=50, loop=1, regression_method="L"):
    # return test scores of diverse append.
    # times is # of iterations default = 100
    # train&test is the train and test data
    # init is the initial pool of data
    # loop is the number of data points in each batch of iteration (not used in the current function)
    # Check if train and test data are provided
    if train is None or test is None:
        raise ValueError("Both train and test data must be provided.")

    # Debug statement to check the type of init
    if not isinstance(init, int):
        raise TypeError(f"Expected init to be an integer, but got {type(init)} instead.")

    if not isinstance(times, int):
        raise TypeError(f"Expected times to be an integer, but got {type(times)} instead.")

    if not isinstance(loop, int):
        raise TypeError(f"Expected loop to be an integer, but got {type(loop)} instead.")

    if regression_method == "L":
        clf = LogisticRegression()
    elif regression_method == "M":
        clf = MLPClassifier(hidden_layer_sizes=(50,), learning_rate_init=0.01)


    label = train.sample(n=init)
    pool = train.drop(label.index).reset_index(drop=True)
    number = []
    for i in range(times):
        X = label.iloc[:, :(label.shape[1] - 1)]
        y = label.iloc[:, -1]
        model = clf.fit(X, y)
        diff = []
        y = pool[['feat_pca_{}'.format(i) for i in range(20)]]
        for j in range(len(pool)):
            predic = model.predict_proba([y.iloc[j]])
            diff.append(abs(0.5 - predic[0, 0]))
        data_array = np.array(diff)
        sorted_indices = np.argsort(data_array)
        n = loop
        top_n_indices = sorted_indices[:n]
        label = pd.concat([label, pool.iloc[top_n_indices]], ignore_index=True)
        pool = pool.drop(top_n_indices).reset_index(drop=True)
        X = label.iloc[:, :(label.shape[1] - 1)]
        y = label.iloc[:, -1]
        number.append(
            clf.fit(X, y).score(test.iloc[:, :(test.shape[1] - 1)], test.iloc[:, -1]))
        # print(fit1(label).score(test[['feat_pca_{}'.format(i) for i in range(20)]], test['classification_label']))
    # return label, pool
    return number

def diverse_matrix(train=None, test=None, times=100, init=50, loop=1, regression_method="L"):
    # return test scores of diverse append.
    # times is # of iterations default = 100
    # train&test is the train and test data
    # init is the initial pool of data
    # loop is the number of data points in each batch of iteration (not used in the current function)
    # Check if train and test data are provided
    if train is None or test is None:
        raise ValueError("Both train and test data must be provided.")

    # Debug statement to check the type of init
    if not isinstance(init, int):
        raise TypeError(f"Expected init to be an integer, but got {type(init)} instead.")

    if not isinstance(times, int):
        raise TypeError(f"Expected times to be an integer, but got {type(times)} instead.")

    if not isinstance(loop, int):
        raise TypeError(f"Expected loop to be an integer, but got {type(loop)} instead.")

    if regression_method == "L":
        clf = LogisticRegression()
    elif regression_method == "M":
        clf = MLPClassifier(hidden_layer_sizes=(50,), learning_rate_init=0.01)

    label = train.sample(n=init)
    pool = train.drop(label.index).reset_index(drop=True)
    number = []
    for i in range(times):
        distances = cdist(label, pool)
        minavgdis = []
        for i in range(distances.shape[1]):
            minavgdis.append(min(distances[:, i]))
        data_array = np.array(minavgdis)
        sorted_indices = np.argsort(data_array)
        n = loop
        top_n_indices = sorted_indices[-n:]
        label = pd.concat([label, pool.iloc[top_n_indices]], ignore_index=True)
        pool = pool.drop(top_n_indices).reset_index(drop=True)
        X = label.iloc[:, :(label.shape[1] - 1)]
        y = label.iloc[:, -1]
        number.append(
            clf.fit(X, y).score(test.iloc[:, :(test.shape[1] - 1)], test.iloc[:, -1]))
        # print(fit1(label).score(test[['feat_pca_{}'.format(i) for i in range(20)]], test['classification_label']))
    # return label, pool
    return number

def diverse_tree(train=None, test=None, times=100, init=50, loop=1, regression_method="L"):
    # return test scores of diverse append.
    # times is # of iterations default = 100
    # train&test is the train and test data
    # init is the initial pool of data
    # loop is the number of data points in each batch of iteration (not used in the current function)
    # Check if train and test data are provided
    if train is None or test is None:
        raise ValueError("Both train and test data must be provided.")

    # Debug statement to check the type of init
    if not isinstance(init, int):
        raise TypeError(f"Expected init to be an integer, but got {type(init)} instead.")

    if not isinstance(times, int):
        raise TypeError(f"Expected times to be an integer, but got {type(times)} instead.")

    if not isinstance(loop, int):
        raise TypeError(f"Expected loop to be an integer, but got {type(loop)} instead.")

    if regression_method == "L":
        clf = LogisticRegression()
    elif regression_method == "M":
        clf = MLPClassifier(hidden_layer_sizes=(50,), learning_rate_init=0.01)

    label = train.sample(n=init)
    pool = train.drop(label.index).reset_index(drop=True)
    number = []
    for i in range(times):
        Tree = cKDTree(label)
        dsitance_ind = Tree.query(pool)
        distance = dsitance_ind[0]
        sort_dis = distance.argsort()
        top_n_indices = sort_dis[-loop:]
        label = pd.concat([label, pool.iloc[top_n_indices]], ignore_index=True)
        pool = pool.drop(top_n_indices).reset_index(drop=True)
        X = label.iloc[:, :(label.shape[1] - 1)]
        y = label.iloc[:, -1]
        number.append(
            clf.fit(X, y).score(test.iloc[:, :(test.shape[1] - 1)], test.iloc[:, -1]))
        # print(fit1(label).score(test[['feat_pca_{}'.format(i) for i in range(20)]], test['classification_label']))
        # return label, pool
    return number
