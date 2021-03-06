#! /usr/bin/env python

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'src/')))
from sgpi.util import get_dataset, get_xy, get_r_squared, split, get_cv
from sgpi import model
from sgpi.learner import SGRegressionLearner, SGClassificationLearner
from sgpi.mnist import l2_distance, mnist_interactions
from sgpi.grid_search import GridSearch
import numpy as np
from operator import itemgetter
from sacred import Experiment
from sklearn.base import clone
from sklearn.cross_validation import KFold, StratifiedKFold

import pysgpp as sg

ex = Experiment('grid_search')

@ex.config
def config():
    level = 2
    num = 20
    T = 0.0
    dataset = 'concrete'

@ex.automain
def main(level, num, T, dataset, _log):
    sg.omp_set_num_threads(4)
    if 'optdigits' in dataset:
        if 'sub' in dataset:
            df_train = get_dataset('optdigits_sub_train')
            df_test = get_dataset('optdigits_sub_test')
        else:
            df_train = get_dataset('optdigits_train')
            df_test = get_dataset('optdigits_test')
        X_train, y_train = get_xy(df_train)
        X_test, y_test = get_xy(df_test)
    else:
        df = get_dataset(dataset)
        train, test = split(df)
        X_train, y_train = get_xy(train)
        X_test, y_test = get_xy(test)
    _log.debug("Read file.")

    session = model.make_session()
    _log.debug("Created SQL session.")

    grid_config = model.GridConfig(type=6, level=level, T=T)
    adaptivity_config = model.AdaptivityConfig(num_refinements=0, no_points=3, treshold=0.0, percent=0.0)
    epsilon = np.sqrt(np.finfo(np.float).eps)
    solver_type = sg.SLESolverType_CG
    solver_config = model.SolverConfig(type=solver_type, max_iterations=70, epsilon=epsilon, threshold=10e-5)
    final_solver_config = model.SolverConfig(type=solver_type, max_iterations=150, epsilon=epsilon, threshold=10e-6)

    # solver_type = sg.SLESolverType_FISTA
    # solver_config = model.SolverConfig(type=solver_type, max_iterations=200, epsilon=0.0, threshold=10e-5)
    # final_solver_config = model.SolverConfig(type=solver_type, max_iterations=400, epsilon=0.0, threshold=10e-8)
    regularization_type = sg.RegularizationType_Identity
    #regularization_type = sg.RegularizationType_ElasticNet
    regularization_config = model.RegularizationConfig(type=regularization_type, l1_ratio=0.05, exponent_base=1.0)
    experiment = model.Experiment(dataset=dataset)

    session.add(grid_config)
    session.add(adaptivity_config)
    session.add(solver_config)
    session.add(final_solver_config)
    session.add(experiment)

    _log.debug("Created configurations.")

    interactions = None
    #interactions = mnist_interactions(l2_distance, 2*np.sqrt(2), level)
    grid_config.interactions = str(interactions)
    if 'optdigits' in dataset:
        estimator = SGClassificationLearner(grid_config, regularization_config, solver_config,
                                        final_solver_config, adaptivity_config, interactions)
    else:
        estimator = SGRegressionLearner(grid_config, regularization_config, solver_config,
                                        final_solver_config, adaptivity_config, interactions)
    cv = get_cv(dataset, y_train)
    experiment.cv = str(cv)
    #lambda_grid = np.logspace(-6, -1, num=num)
    lambda_grid = [0.1]
    parameters = {'regularization_config__lambda_reg': lambda_grid}
    grid_search = GridSearch(estimator, parameters, cv)
    _log.info("Start learning.")
    grid_search.fit(X_train, y_train)
    _log.info("Finished learning.")

    first = True
    for score in sorted(grid_search.grid_scores_, key=itemgetter(1), reverse=True):
        if 'optdigits' in dataset:
            error_mult = 1
        else:
            error_mult = -1
        validation_mse = error_mult*score.mean_validation_score
        validation_std = np.std(np.abs(score.cv_validation_scores))
        validation_grid_sizes = score.cv_grid_sizes
        params = estimator.get_params()
        params.update(score.parameters)
        regularization_config = model.RegularizationConfig(
                                type=params['regularization_config__type'],
                                lambda_reg=params['regularization_config__lambda_reg'],
                                exponent_base=params['regularization_config__exponent_base'],
                                l1_ratio=params['regularization_config__l1_ratio'])
        session.add(regularization_config)
        result = model.Result(validation_mse=validation_mse,
                              validation_std=validation_std,
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
            result.train_mse = error_mult * estimator.score(X_train, y_train)
            result.test_mse = error_mult * estimator.score(X_test, y_test)
            if 'optdigits' not in dataset:
                result.train_r2 = get_r_squared(estimator, X_train, y_train)
                result.test_r2 = get_r_squared(estimator, X_test, y_test)

        session.add(result)

    _log.debug("Pushing to database.")
    session.commit()

    _log.info("Finished experiment.")
