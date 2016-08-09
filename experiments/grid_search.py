#! /usr/bin/env python

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'src/')))
from sgpi.util import get_dataset, get_xy, get_r_squared, split
from sgpi import model
from sgpi.learner import SGRegressionLearner
from sgpi.grid_search import GridSearch
import numpy as np
from operator import itemgetter
from sacred import Experiment
from sklearn.base import clone
from sklearn.cross_validation import KFold
ex = Experiment('grid_search')

@ex.config
def config():
    level = 2
    num = 20
    T = 0.0
    dataset = 'concrete'

@ex.automain
def main(level, num, T, dataset, _log):
    df = get_dataset(dataset)
    train, test = split(df)
    X_train, y_train = get_xy(train)
    X_test, y_test = get_xy(test)
    _log.debug("Read file.")

    session = model.make_session()
    _log.debug("Created SQL session.")

    grid_config = model.GridConfig(type=6, level=level, T=T)
    adaptivity_config = model.AdaptivityConfig(num_refinements=5, no_points=3, treshold=0.0, percent=0.0)
    solver_config = model.SolverConfig(type=0, max_iterations=50, epsilon=10e-4)
    final_solver_config = model.SolverConfig(type=0, max_iterations=70, epsilon=10e-4)
    regularization_config = model.RegularizationConfig(type=2)
    experiment = model.Experiment(dataset=dataset)

    session.add(grid_config)
    session.add(adaptivity_config)
    session.add(solver_config)
    session.add(final_solver_config)
    session.add(experiment)

    _log.debug("Created configurations.")

    estimator = SGRegressionLearner(grid_config, regularization_config, solver_config,
                                    final_solver_config, adaptivity_config)
    cv = KFold(X_train.shape[0], n_folds=10)
    experiment.cv = str(cv)
    lambda_grid = np.logspace(-1, -4, num=num)
    parameters = {'regularization_config__lambda_reg': lambda_grid,
                  'regularization_config__exponent_base': [1, 4]}
    grid_search = GridSearch(estimator, parameters, cv)
    _log.info("Start learning.")
    grid_search.fit(X_train, y_train)
    _log.info("Finished learning.")


    first = True
    for score in sorted(grid_search.grid_scores_, key=itemgetter(1), reverse=True):
        validation_mse = -score.mean_validation_score
        validation_grid_sizes = score.cv_grid_sizes
        params = estimator.get_params()
        params.update(score.parameters)
        regularization_config = model.RegularizationConfig(
                                type=params['regularization_config__type'],
                                lambda_reg=params['regularization_config__lambda_reg'],
                                exponent_base=params['regularization_config__exponent_base'])
        session.add(regularization_config)
        result = model.Result(validation_mse=validation_mse,
                              grid_config=grid_config,
                              adaptivity_config=adaptivity_config,
                              solver_config=solver_config,
                              final_solver_config=final_solver_config,
                              regularization_config=regularization_config,
                              experiment=experiment,
                              validation_grid_points_mean=np.mean(validation_grid_sizes),
                              validation_grid_points_stdev=np.std(validation_grid_sizes))
        # Retrain best learner and validate test set:
        if first:
            first = False
            estimator.set_params(**params)
            estimator.fit(X_train, y_train)
            result.train_grid_points = estimator.get_grid_size()
            result.train_mse = -estimator.score(X_train, y_train)
            result.train_r2 = get_r_squared(estimator, X_train, y_train)
            result.test_mse = -estimator.score(X_test, y_test)
            result.test_r2 = get_r_squared(estimator, X_test, y_test)

        session.add(result)

    _log.debug("Pushing to database.")
    session.commit()

    _log.info("Finished experiment.")
