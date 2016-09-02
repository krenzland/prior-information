from pysgpp import RegressionLearner, ClassificationLearner, RegularGridConfiguration, \
AdpativityConfiguration, SLESolverConfiguration, RegularizationConfiguration, DataMatrix, \
DataVector
import sklearn
from .util import to_data_matrix

def _model_to_settings(X, in_grid_config, in_regularization_config, in_solver_config, in_final_solver_config,
                       in_adaptivity_config):
    grid_config = RegularGridConfiguration()
    grid_config.dim_ = X.shape[1]
    grid_config.level_ = in_grid_config.level
    grid_config.type_ = in_grid_config.type #6 = ModLinear
    grid_config.t_ = in_grid_config.T

    adaptivity_config = AdpativityConfiguration()
    adaptivity_config.numRefinements_ = in_adaptivity_config.num_refinements
    if adaptivity_config.numRefinements_ == 0:
        adaptivity_config.noPoints_ = 0
        adaptivity_config.percent_ = 0
        adaptivity_config.threshold_ = 0
    else:
        adaptivity_config.noPoints_ = in_adaptivity_config.no_points
        adaptivity_config.percent_ = in_adaptivity_config.percent
        adaptivity_config.threshold_ = in_adaptivity_config.treshold

    solver_config = SLESolverConfiguration()
    solver_config.type_ = in_solver_config.type
    solver_config.maxIterations_ = in_solver_config.max_iterations
    solver_config.eps_ = in_solver_config.epsilon
    solver_config.threshold_ = in_solver_config.threshold

    final_solver_config = SLESolverConfiguration()
    final_solver_config.type_ = in_final_solver_config.type
    final_solver_config.maxIterations_ = in_final_solver_config.max_iterations
    final_solver_config.eps_ = in_final_solver_config.epsilon
    final_solver_config.threshold_ = in_solver_config.threshold

    regularization_config = RegularizationConfiguration()
    regularization_config.exponentBase_ = in_regularization_config.exponent_base
    regularization_config.regType_ = in_regularization_config.type
    regularization_config.lambda_ = in_regularization_config.lambda_reg
    regularization_config.l1Ratio_ = in_regularization_config.l1_ratio

    return grid_config, adaptivity_config, solver_config, final_solver_config, regularization_config

# http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator
class SGRegressionLearner(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__(self, grid_config, regularization_config, solver_config, final_solver_config, adaptivity_config,
                interactions=None):
        self.grid_config = grid_config
        self.regularization_config = regularization_config
        self.solver_config = solver_config
        self.final_solver_config = final_solver_config
        self.adaptivity_config = adaptivity_config
        self.interactions = interactions

    def fit(self, X, y, weights=None):
        settings = _model_to_settings(X, self.grid_config, self.regularization_config,
                                      self.solver_config, self.final_solver_config,
                                      self.adaptivity_config)
        grid_config, adaptivity_config, solver_config, final_solver_config, regularization_config = settings
        if self.interactions is not None:
            self._learner = RegressionLearner(grid_config, adaptivity_config, solver_config,
                                              final_solver_config, regularization_config, self.interactions)
        else:
            self._learner = RegressionLearner(*settings)

        X_mat = to_data_matrix(X)
        y_vec = DataVector(y.tolist())
        if weights is not None:
            grid_size = self._learner.getGridSize()
            self._learner.setWeights(DataVector(weights[0:grid_size]))

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

class SGClassificationLearner(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__(self, grid_config, regularization_config, solver_config, final_solver_config, adaptivity_config,
                interactions=None):
        self.grid_config = grid_config
        self.regularization_config = regularization_config
        self.solver_config = solver_config
        self.final_solver_config = final_solver_config
        self.adaptivity_config = adaptivity_config
        self.interactions = interactions

    def fit(self, X, y):
        settings = _model_to_settings(X, self.grid_config, self.regularization_config,
                                      self.solver_config, self.final_solver_config,
                                      self.adaptivity_config)
        grid_config, adaptivity_config, solver_config, final_solver_config, regularization_config = settings
        if self.interactions is not None:
            self._learner = ClassificationLearner(grid_config, adaptivity_config, solver_config,
                                              final_solver_config, regularization_config, self.interactions)
        else:
            self._learner = ClassificationLearner(*settings)

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
        accuracy = self._learner.getAccuracy(X_mat, y_vec)
        return accuracy

    def get_grid_size(self):
        return self._learner.getGridSize()
