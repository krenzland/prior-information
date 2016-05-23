# coding=utf-8

from __future__ import print_function
import sys
import numpy as np; np.random.seed(42)
import pandas as pd
import sklearn
import sklearn.cross_validation as cv
import sklearn.preprocessing as pre
import sklearn.grid_search

from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style('darkgrid'); sns.set_palette('muted')

from pysgpp import RegressionLearner, ClassificationLearner, RegularGridConfiguration, \
AdpativityConfiguration, SLESolverConfiguration, RegularizationConfiguration, DataMatrix, \
DataVector

def to_data_matrix(arr):
    (size_x, size_y) = arr.shape
    matrix = DataMatrix(size_x, size_y)
    cur_row = 0
    for x in arr:
        x_vec = DataVector(x.tolist())
        matrix.setRow(cur_row,x_vec)
        cur_row += 1
    return matrix

# http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator
class SGRegressionLearner(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__(self, lambdaReg=0.01, T = 0.0, typeReg=2, level=3):
        self.lambdaReg = lambdaReg
        self.typeReg = typeReg
        self.T = T
        self.level = level

    def fit(self, X, y):
        grid_config = RegularGridConfiguration()
        grid_config.dim_ = X.shape[1]
        grid_config.level_ = self.level
        grid_config.type_ = 6 # ModLinear
        grid_config.t_ = self.T

        adaptivity_config = AdpativityConfiguration()
        adaptivity_config.noPoints_ = 0
        adaptivity_config.numRefinements_ = 0

        solver_config = SLESolverConfiguration()
        solver_config.type_ = 0 # CG
        solver_config.maxIterations_ = 500
        solver_config.eps_ = 1e-6

        regularization_config = RegularizationConfiguration()
        regularization_config.exponentBase_ = 0.25
        regularization_config.regType_ = self.typeReg # diagonal = 2
        regularization_config.lambda_ = self.lambdaReg
        self._learner = RegressionLearner(grid_config, adaptivity_config, solver_config, regularization_config)

        X_mat = to_data_matrix(X)
        y_vec = DataVector(y.tolist())
        self._learner.train(X_mat, y_vec)

    def predict(self, X):
        X_mat = to_data_matrix(X)
        result = self._learner.predict(X_mat)
        return result.array()

    def score(self, X, y, sample_weight=None):
        X_mat = to_data_matrix(X)
        y_vec = DataVector(y.tolist())
        mse = self._learner.getMSE(X_mat, y_vec)
        return -mse

    def get_grid_size(self):
        return self._learner.getGridSize()

def plot_grid_search(gridSearch, T=0.0):
    scores = gridSearch.grid_scores_
    diag_scores = [s for s in scores if s[0]['typeReg'] == 2]
    ident_scores = [s for s in scores if s[0]['typeReg'] == 0]
    diag_params,diag_mean, diag_std = zip(*diag_scores)
    ident_params, ident_mean, ident_std = zip(*ident_scores)
    diag_mean = np.sqrt([-x for x in diag_mean])
    ident_mean = np.sqrt([-x for x in ident_mean])
    lambdas = [l['lambdaReg'] for l in diag_params]
    diff_mean = diag_mean - ident_mean

    fig = plt.figure(2, figsize=(15,5))

    ax1 = fig.add_subplot(121)
    ax1.set_ylabel("RMSE")
    ax1.set_xlabel("Lambda")
    ax1.plot(lambdas, diag_mean)
    ax1.plot(lambdas, ident_mean)
    ax1.legend(["Diagonal", "Identity"])
    ax1.set_xscale('log')
    ax1.invert_xaxis()
    ax1.set_title("RMSE for T = " + str(T))

    ax2 = fig.add_subplot(122)
    ax2.plot(lambdas, diff_mean)
    ax2.set_ylabel("RMSE difference")
    ax2.set_xlabel("Lambda")
    ax2.legend(["diag - ident"])
    ax2.set_xscale('log')
    ax2.invert_xaxis()
    ax2.set_title("No. of gridpoints: " + str(gridSearch.best_estimator_.get_grid_size()))

    fig.show()

def auto_grid_search_with_t(X, Y, parameters, lambdaGrid, T=0.0, level=3):
    learner = SGRegressionLearner(T=T, level=level)
    shuffle = cv.ShuffleSplit(X.shape[0], n_iter=5, random_state=42)
    parameters = [{"lambdaReg": lambdaGrid, "typeReg": [2]}, # diag
                  {"lambdaReg": lambdaGrid, "typeReg": [0]}] # ident
    grid_search = sklearn.grid_search.GridSearchCV(learner, parameters, cv=shuffle, verbose=1, n_jobs=4)
    grid_search.fit(X,Y)
    plot_grid_search(grid_search, T)
    return grid_search

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

def plot_cox(name, X, y):
    data_trans = stats.boxcox(X + 10e-1)

    data_scaled = data_trans[0].reshape(-1,1)
    scaler = pre.MinMaxScaler()
    data_scaled = scaler.fit_transform(data_scaled)

    fig = plt.figure(figsize=(14, 7))
    f, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=False)

    sns.distplot(X, ax=axes[0][0], axlabel="")
    axes[0][0].set_title("raw data for " + name)

    axes[0][1].scatter(X,y)

    sns.distplot(data_scaled, ax=axes[1][0], label="")
    axes[1][0].set_title(u"λ = " + str(data_trans[1]))
    axes[1][1].scatter(data_scaled, y)

    return data_trans[1]

def plot_cox_df(df):
    lambdas = {}
    for x in df:
        try:
            lamb = plot_cox(str(x), df[x], df.ix[:,-1])
            lambdas[str(x)] = lamb
        except Exception, e:
            print("Error whilst processing {}: {}".format(x, e))
    last_name = list(df)[-1]
    lambdas[last_name] = 1 # don't transform last col!
    return lambdas

def transform_cox(df, lambdas):
    scaler = pre.MinMaxScaler()
    for variable in lambdas:
        lamb = lambdas[variable]
        if lamb == 1:
            continue; # identity transform
        data_trans = stats.boxcox(df[variable] + 10e-1)
        df[variable] = scaler.fit_transform(data_trans[0])
    return df
