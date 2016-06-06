#!/usr/bin/env python

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'datasets/')))
from tools import *

import pandas as pd
import numpy as np
import sklearn.cross_validation as cv
import sklearn.grid_search
from sacred import Experiment

import sql_log as model

ex = Experiment('power_plant_grid')

@ex.config
def config():
    level = 4
    num = 20
    T = 0.0
    n_jobs = -2

@ex.automain
def main(level, num, T, n_jobs, _log):
    input_file = '../datasets/processed/power_plant/power_plant_prep.csv'
    df = pd.read_csv(input_file)

    train, test = split(df)
    X_train = np.array(train.ix[:,0:-1])
    y_train = train.ix[:,-1]
    X_test = np.array(test.ix[:,0:-1])
    y_test = test.ix[:,-1]

    _log.debug("Read file.")
    session = model.make_session()
    _log.debug("Created SQL session.")

    grid_config = model.GridConfig(type=6, level=level, T=T)
    adaptivity_config = model.AdaptivityConfig(num_refinements=0, no_points=10, treshold=0.0, percent=5.0)
    solver_config = model.SolverConfig(type=0, max_iterations=1000, epsilon=10e-6)
    regularization_config = model.RegularizationConfig(type=2)

    session.add(grid_config)
    session.add(adaptivity_config)
    session.add(solver_config)

    learner = SGRegressionLearner(grid_config, regularization_config, solver_config, adaptivity_config)
    shuffle = cv.ShuffleSplit(X_train.shape[0], n_iter=10, random_state = 42)
    lambda_grid = np.logspace(2,-10, num=num)
    parameters = {'regularization_config__lambda_reg': lambda_grid,
                  'regularization_config__exponent_base': [1.0, 0.5, 0.25]}
    grid_search = sklearn.grid_search.GridSearchCV(learner, parameters, cv=shuffle, verbose=0, n_jobs=n_jobs)
    grid_search.fit(X_train, y_train)
    _log.debug("Finished learning.")

    experiment = model.Experiment(dataset='power_plant')
    first = True
    for score in sorted(grid_search.grid_scores_):
        validation_mse = -score.mean_validation_score
        params = score.parameters
        regularization_lambda = params['regularization_config__lambda_reg']
        regularization_base = params['regularization_config__exponent_base']
        regularization_config = model.RegularizationConfig(type=2)
        regularization_config.lambda_reg = regularization_lambda
        regularization_config.exponent_base = regularization_base
        session.add(regularization_config)
        result = model.Result(validation_mse=validation_mse, grid_config=grid_config,
                                adaptivity_config=adaptivity_config, solver_config=solver_config,
                                regularization_config=regularization_config, experiment=experiment)
        if first: #get more metrics for best learner!
            first = False
            best_learner = SGRegressionLearner(grid_config, regularization_config, solver_config,
                                               adaptivity_config)
            best_learner.set_params(**score.parameters)
            best_learner.fit(X_train, y_train) # refit with entire training set

            result.train_mse = -best_learner.score(X_train, y_train)
            result.train_r2 = get_r_squared(best_learner, X_train, y_train)
            result.test_mse = -best_learner.score(X_test, y_test)
            result.test_r2 = get_r_squared(best_learner, X_test, y_test)

        session.add(result)

    session.add(experiment)

    _log.debug("Pushing to DB!")
    session.commit()

    _log.debug("Finished experiment!")
