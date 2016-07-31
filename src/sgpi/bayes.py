from .learner import SGRegressionLearner
from .grid_search import evaluate
from collections import namedtuple
import numpy as np
import bayesopt
from bayesoptmodule import BayesOptContinuous

Hyp_param = namedtuple('Hyp_param', 'name lower_bound upper_bound')
class BayesOptReg(BayesOptContinuous):
    def __init__(self, estimator, cv, X, y, params, n_iter, n_iter_relearn=5, n_init_samples = 2):
        n_dimensions = len(params)
        assert(isinstance(estimator, SGRegressionLearner))
        BayesOptContinuous.__init__(self, n_dimensions)

        self.estimator_ = estimator
        self.cv_ = cv
        self.X_ = X
        self.y_ = y
        self.parameters = {}
        self.parameters['n_iterations'] = n_iter
        self.parameters['n_iter_relearn'] = n_iter_relearn
        self.parameters['n_init_samples'] = n_init_samples
        self.best_grid_points_ = None
        self.best_mse_ = float('-inf')

        names = []
        lower_bound = []
        upper_bound = []
        for name, low, high in params:
            names.append(name)
            lower_bound.append(low)
            upper_bound.append(high)
        self.param_names_ = names

        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)

    def evaluateSample(self, query):
        params = dict(zip(self.param_names_,query))
        result = evaluate(self.estimator_, params, self.cv_, self.X_, self.y_)
        mean_cv = result.mean_validation_score
        if mean_cv > self.best_mse_:
            self.best_mse_ = mean_cv
            self.best_grid_points_ = result.cv_grid_sizes
        return -mean_cv

    def optimize(self):
        mse, x_o, _ = super(BayesOptReg, self).optimize()
        params = dict(zip(self.param_names_,x_o))
        return mse, params, self.best_grid_points_

