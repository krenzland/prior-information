import os
import numpy as np; np.random.seed(42)
import pandas as pd
import sklearn.preprocessing as pre
import sklearn.cross_validation as cv
from scipy import stats
from scipy.sparse.linalg import LinearOperator
from pysgpp import DataMatrix, DataVector
import pysgpp as sg

def to_data_matrix(arr):
    (size_x, size_y) = arr.shape
    matrix = DataMatrix(size_x, size_y)
    cur_row = 0
    for x in arr:
        x_vec = DataVector(x.tolist())
        matrix.setRow(cur_row,x_vec)
        cur_row += 1
    return matrix

def write_arff(df, name, filename=None):
    relation = '@RELATION {}'.format(name)
    attributes = '\n'.join('@ATTRIBUTE {} numeric'.format(col) for col in df)
    data = '@DATA'
    data_str = df.to_csv(header=False)
    arff_string = '\n\n'.join([relation, attributes, data, data_str])
    if filename:
        with open(filename, 'w') as fp:
            fp.write(arff_string)
    else:
        return arff_string

def last_col_to_numeric(df):
    y = df.ix[:,-1:]
    y_dummies = pd.get_dummies(y)
    df.ix[:,-1:]= y_dummies.values.argmax(1)

def split(df, test_size=0.2):
    train, test = cv.train_test_split(df, test_size=test_size, random_state=1337)
    return train, test

def scale(df, scaler=None):
    Y = df.ix[:,-1] # save Y (don't need to transform it/useless for cat. data!)
    X = df.values
    if scaler:
        X = scaler.transform(X)
    else:
        scaler = pre.MinMaxScaler()
        X = scaler.fit_transform(X)
    index = df.index
    columns = df.columns
    df = pd.DataFrame(data=X, index=index, columns=columns)
    df.ix[:,-1] = Y
    return scaler, df

def transform_cox(df, lambdas):
    scaler = pre.MinMaxScaler()
    for variable in lambdas:
        lamb = lambdas[variable]
        if lamb == 1:
            continue; # identity transform
        data_trans = stats.boxcox(df[variable] + 10e-1)
        df[variable] = scaler.fit_transform(np.array(data_trans[0]).reshape(-1, 1))
    return df

def get_r_squared(learner, X, y):
    mse = -learner.score(X,y)
    ss_reg = mse * y.size
    ss_tot = np.var(y) * y.size
    return 1 - ss_reg/ss_tot

def get_dataset(name):
    package_directory = os.path.dirname(os.path.abspath(__file__))
    datasets = {
        'concrete': 'concrete/concrete_prep.csv',
        'power_plant': 'power_plant/power_plant_prep.csv',
        'friedman1': 'friedman1/friedman1_prep.csv',
        'friedman3': 'friedman3/friedman3_prep.csv',
        'yeast': 'yeast/yeast_prep.csv',
        'abalone': 'abalone/abalone_prep.csv',
        'diag_test_low_noise': 'diagonal_test/diag_test_low_noise.csv',
        'diag_test_medium_noise': 'diagonal_test/diag_test_medium_noise.csv',
        'diag_test_high_noise': 'diagonal_test/diag_test_high_noise.csv',
        'diag_test_very_high_noise': 'diagonal_test/diag_test_very_high_noise.csv'
    }
    folder = os.path.join(package_directory, '../../datasets/processed/')
    path = os.path.join(folder, datasets[name])
    return pd.read_csv(path)

def get_xy(data):
    X = np.array(data.ix[:,0:-1])
    y = (data.ix[:,-1]).values
    return X,y

def get_used_coords(num):
    zeros = np.zeros(len(num)) + 0.5
    return np.equal(zeros, num)

def coords_to_pred(coords):
    s = ""
    for i, c in enumerate(coords):
        if not c:
            s = s + "x{} ".format(i + 1)
    s = s.strip()
    if s == "":
        return "bias"
    else:
        s = s.replace(" ", "-")
        return s

def group_weights_raw(grid):
    storage = grid.getStorage()
    dim = storage.getDimension()

    coords = []
    for x in range(0, grid.getSize()):
        gen0 = storage.get(x)
        curCoords = []
        for i in range(0,dim):
            curCoords.append(gen0.getCoord(i))
        curCoords = np.array(curCoords)
        coords.append(curCoords)

    terms = {}
    groups = {}
    terms_nums = []
    for num, r in enumerate(coords):
        d = tuple(get_used_coords(r))
        if d not in terms:
            terms[d] = []
            groups[d] = len(groups)
        terms[d].append(num)
        terms_nums.append(groups[d])

    return terms

def group_weights_format(grid):
    terms = group_weights_raw(grid)
    return dict([(coords_to_pred(coords), terms[coords]) for coords in terms])

def group_list(grid):
    groups = group_weights_format(grid)
    glist = [None] * (grid.getSize())
    for group in groups:
        for i in groups[group]:
            glist[i] = group
    return glist

def get_Phi(grid, X_train):
    def eval_op(x, op, size):
        result_vec = sg.DataVector(size)
        x = sg.DataVector(np.array(x).flatten())
        op.mult(x, result_vec)
        return result_vec.array().copy()

    def eval_op_transpose(x, op, size):
        result_vec = sg.DataVector(size)
        x = sg.DataVector(np.array(x).flatten())
        op.multTranspose(x, result_vec)
        return result_vec.array().copy()

    data_train = to_data_matrix(X_train)

    num_elem = X_train.shape[0]

    op = sg.createOperationMultipleEval(grid, data_train)
    matvec = lambda x: eval_op(x, op, num_elem)
    rmatvec = lambda x: eval_op_transpose(x, op, grid.getSize())

    shape = (num_elem, grid.getSize())
    linop = LinearOperator(shape, matvec, rmatvec, dtype='float64')

    Phi = linop.matmat(np.matrix(np.identity(grid.getSize())))
    return Phi

def get_max_lambda(Phi, y, num_points, l1_ratio=1.0):
    max_prod = 0
    for i in range(0, num_points):
        a = np.asarray(Phi[:,i]).flatten()
        prod = np.inner(a, y)
        max_prod = max(max_prod, prod)
    max_lambda = max_prod/(l1_ratio)
    return max_lambda

def calculate_weight_path(estimator, X, y, max_lambda, epsilon=0.001, num_lambdas=25,verbose=0):
    min_lambda = epsilon * max_lambda
    estimator.set_params(regularization_config__lambda_reg = max_lambda)
    estimator.fit(X, y)
    lambda_grid = np.logspace(np.log10(max_lambda), np.log10(min_lambda), num=num_lambdas)
    min_lambda, lambda_grid, min_lambda/X.shape[0]
    weights = []
    for i, lamb in enumerate(lambda_grid):
        estimator.set_params(regularization_config__lambda_reg = lamb)
        if verbose > 0:
            print "Started training estimator {}".format(i)
        estimator.fit(X, y, estimator.get_weights()) # reuse old weights
        if verbose > 0:
            print "Finished training estimator {}".format(i)
        weights.append(estimator.get_weights())
    df = pd.DataFrame(weights, index=lambda_grid)
    df = df.transpose()
    glist = group_list(grid)
    df.index=glist
    return df
