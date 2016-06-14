from collections import namedtuple
from operator import itemgetter
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import ParameterGrid

Grid_score = namedtuple('Grid_score', 'parameters mean_validation_score cv_validation_scores cv_grid_sizes')

def evaluate(estimator, params, cv, X, y):
    errors = []
    grid_sizes = []
    estimator.set_params(**params)
    for train, validation in cv:
        train_X = X[train]
        train_y = y[train]
        test_X = X[validation]
        test_y = y[validation]

        estimator.fit(train_X,train_y)
        error = estimator.score(test_X, test_y)
        errors.append(error)
        grid_size = estimator.get_grid_size()
        grid_sizes.append(grid_size)

    return Grid_score(params, np.mean(errors), errors, grid_sizes)


class GridSearch:
    grid_scores_ = []

    def __init__(self, estimator, param_grid, cv):
        assert(isinstance(estimator, SGRegressionLearner))
        self.base_estimator_ = clone(estimator)
        if not isinstance(param_grid, ParameterGrid):
            param_grid = ParameterGrid(param_grid)
        self.param_grid_ = param_grid
        self.cv_ = cv

    def fit(self, X, y):
        base_estimator = clone(self.base_estimator_)
        self.grid_scores_ = Parallel(
                n_jobs=3, verbose=2,
                pre_dispatch=2
             )(
                delayed(evaluate)(clone(self.base_estimator_), params, self.cv_, X, y)
                    for params in self.param_grid_)
        return self
