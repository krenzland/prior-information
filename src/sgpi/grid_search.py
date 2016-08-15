from collections import namedtuple
import numpy as np; np.random.seed(42)
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import ParameterGrid
from .learner import SGRegressionLearner

Grid_score = namedtuple('Grid_score', 'parameters mean_validation_score cv_validation_scores cv_grid_sizes')

def evaluate_one(estimator, params, X, y, train, test):
    train_X = X[train]
    train_y = y[train]
    test_X = X[test]
    test_y = y[test]
    estimator.set_params(**params)
    estimator.fit(train_X,train_y)
    error = estimator.score(test_X, test_y)
    grid_size = estimator.get_grid_size()
    return (error, grid_size)

def evaluate(estimator, params, cv, X, y):
    cv_results = Parallel(
                n_jobs=-1
             )(
                delayed(evaluate_one)(clone(estimator), params, X, y, train, test)
                    for (train, test) in cv)
    errors = []
    grid_sizes = []
    for (err, size) in cv_results:
        errors.append(err)
        grid_sizes.append(size)
    return Grid_score(params, np.mean(errors), errors, grid_sizes)

class GridSearch:
    grid_scores_ = []

    def __init__(self, estimator, param_grid, cv, verbose=1):
        self.base_estimator_ = clone(estimator)
        self.verbose_ = verbose
        if not isinstance(param_grid, ParameterGrid):
            param_grid = ParameterGrid(param_grid)

            self.param_grid_ = param_grid
        self.cv_ = cv

    def fit(self, X, y):
        base_estimator = clone(self.base_estimator_)
        self.grid_scores_ = []
        no_params = self.param_grid_.__len__()
        for i, params in enumerate(self.param_grid_):
            self.grid_scores_.append(evaluate(self.base_estimator_, params, self.cv_, X, y))
            if (self.verbose_ > 0):
                print("Trained estimator no. {}, {} remaining.".format(i, no_params - i -1))
        return self
