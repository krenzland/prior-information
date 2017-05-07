#! /usr/bin/env python

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'src/')))
from sgpi.util import get_dataset, get_xy, get_r_squared, split, get_cv
from sgpi import model
from sgpi.learner import SGRegressionLearner
from sgpi.bayes import BayesOptReg, Hyp_param

import numpy as np
from operator import itemgetter
from sacred import Experiment
from sklearn.base import clone
from sklearn.cross_validation import KFold

import pysgpp as sg

ex = Experiment('bayes_search')

@ex.config
def config():
    level = 2
    num = 10
    num_init = 2
    T = 0.0
    dataset = 'concrete'

@ex.automain
def main(level, num, num_init, T, dataset, _log):
    sg.omp_set_num_threads(4)
    if 'optdigits' in dataset:
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
    adaptivity_config = model.AdaptivityConfig(num_refinements=0, no_points=0, treshold=0.0, percent=0.0)
    epsilon = np.sqrt(np.finfo(np.float).eps)
    solver_type = sg.SLESolverType_CG
    solver_config = model.SolverConfig(type=solver_type, max_iterations=70, epsilon=epsilon, threshold=10e-5)
    final_solver_config = model.SolverConfig(type=solver_type, max_iterations=250, epsilon=epsilon, threshold=10e-6)
    # solver_type = sg.SLESolverType_FISTA
    # solver_config = model.SolverConfig(type=solver_type, max_iterations=200, epsilon=0.0, threshold=10e-5)
    # final_solver_config = model.SolverConfig(type=solver_type, max_iterations=400, epsilon=0.0, threshold=10e-6)
    regularization_type = sg.RegularizationType_Identity
    regularization_config = model.RegularizationConfig(type=regularization_type, l1_ratio=1.0, exponent_base=1.0)
    experiment = model.Experiment(dataset=dataset)

    _log.debug("Created configurations.")

    interactions = [[0,1]] + [[i] for i in range(0,11)]
    #interactions = None
    grid_config.interactions = str(interactions)
    estimator = SGRegressionLearner(grid_config, regularization_config, solver_config,
                                    final_solver_config, adaptivity_config, interactions)

    cv = get_cv(dataset, X_train)
    experiment.cv = str(cv)
    params = [Hyp_param('regularization_config__lambda_reg', 0.0, 0.3)]
              #Hyp_param('regularization_config__exponent_base', 3.0, 7.0)]

    bayes_search = BayesOptReg(estimator, cv, X_train, y_train,
                               params, num, n_init_samples=num_init)

    _log.info("Start learning.")
    validation_score, best_params, cv_grid_sizes = bayes_search.optimize()
    _log.info("Finished learning.")
    _log.info("Best CV-MSE was {}. With params {}".format(validation_score, best_params))
    validation_mse = validation_score
    validation_grid_sizes = cv_grid_sizes

    #Retrain estimator with best parameters
    params = estimator.get_params()
    params.update(best_params)
    estimator.set_params(**params)

    estimator.fit(X_train, y_train)

    result = model.Result(validation_mse=validation_mse,
                            grid_config=grid_config,
                            adaptivity_config=adaptivity_config,
                            solver_config=solver_config,
                            final_solver_config=final_solver_config,
                            regularization_config=regularization_config,
                            experiment=experiment,
                            validation_grid_points_mean=np.mean(validation_grid_sizes),
                            validation_grid_points_stdev=np.std(validation_grid_sizes))

    result.train_grid_points = estimator.get_grid_size()
    result.train_mse = -estimator.score(X_train, y_train)
    result.train_r2 = get_r_squared(estimator, X_train, y_train)
    result.test_mse = -estimator.score(X_test, y_test)
    result.test_r2 = get_r_squared(estimator, X_test, y_test)

    regularization_config = model.RegularizationConfig(
                            type=params['regularization_config__type'],
                            lambda_reg=params['regularization_config__lambda_reg'],
                            exponent_base=params['regularization_config__exponent_base'])

    session.add(grid_config)
    session.add(adaptivity_config)
    session.add(solver_config)
    session.add(final_solver_config)
    session.add(experiment)
    session.add(regularization_config)
    session.add(result)

    _log.debug("Pushing to database.")
    session.commit()

    _log.info("Finished experiment.")
