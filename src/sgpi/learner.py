from pysgpp import RegressionLearner, ClassificationLearner, RegularGridConfiguration, \
AdpativityConfiguration, SLESolverConfiguration, RegularizationConfiguration, DataMatrix, \
DataVector
import sklearn
from .util import to_data_matrix

# http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator
class SGRegressionLearner(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__(self, grid_config, regularization_config, solver_config, final_solver_config, adaptivity_config):
        self.grid_config = grid_config
        self.regularization_config = regularization_config
        self.solver_config = solver_config
        self.final_solver_config = final_solver_config
        self.adaptivity_config = adaptivity_config

    def fit(self, X, y):
        grid_config = RegularGridConfiguration()
        grid_config.dim_ = X.shape[1]
        grid_config.level_ = self.grid_config.level
        grid_config.type_ = self.grid_config.type #6 = ModLinear
        grid_config.t_ = self.grid_config.T

        adaptivity_config = AdpativityConfiguration()
        adaptivity_config.numRefinements_ = self.adaptivity_config.num_refinements
        if adaptivity_config.numRefinements_ == 0:
            adaptivity_config.noPoints_ = 0
            adaptivity_config.percent_ = 0
            adaptivity_config.threshold_ = 0
        else:
            adaptivity_config.noPoints_ = self.adaptivity_config.no_points
            adaptivity_config.percent_ = self.adaptivity_config.percent
            adaptivity_config.threshold_ = self.adaptivity_config.treshold

        solver_config = SLESolverConfiguration()
        solver_config.type_ = self.solver_config.type
        solver_config.maxIterations_ = self.solver_config.max_iterations
        solver_config.eps_ = self.solver_config.epsilon
        solver_config.threshold_ = self.solver_config.threshold

        final_solver_config = SLESolverConfiguration()
        final_solver_config.type_ = self.final_solver_config.type
        final_solver_config.maxIterations_ = self.final_solver_config.max_iterations
        final_solver_config.eps_ = self.final_solver_config.epsilon
        final_solver_config.threshold_ = self.solver_config.threshold

        regularization_config = RegularizationConfiguration()
        regularization_config.exponentBase_ = self.regularization_config.exponent_base
        regularization_config.regType_ = self.regularization_config.type
        regularization_config.lambda_ = self.regularization_config.lambda_reg
        regularization_config.l1Ratio_ = self.regularization_config.l1_ratio

        self._learner = RegressionLearner(grid_config, adaptivity_config, solver_config, final_solver_config,
                                          regularization_config)

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

    def get_weights(self):
        return self._learner.getWeights().array()
