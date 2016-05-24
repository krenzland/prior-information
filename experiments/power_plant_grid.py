#!/usr/bin/env python

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'datasets/')))
from tools import *

import pandas as pd
import numpy as np
import sklearn.cross_validation as cv
import sklearn.grid_search
from sacred import Experiment
import sql_log

ex = Experiment('power_plant_grid')

@ex.config
def config():
    level = 3
    num = 20
    T = 0.0
    n_jobs = 3

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

    learner = SGRegressionLearner(T=T, level=level)
    shuffle = cv.ShuffleSplit(X_train.shape[0], n_iter=10, random_state = 42)
    lambda_grid = np.logspace(0,-5, num=num)
    parameters = [{"lambdaReg": lambda_grid, "typeReg": [2]}, # diag
                {"lambdaReg": lambda_grid, "typeReg": [0]}] # ident
    grid_search = sklearn.grid_search.GridSearchCV(learner, parameters, cv=shuffle, verbose=0, n_jobs=3)
    grid_search.fit(X_train, y_train)
    _log.debug("Finished learning.")

    session = sql_log.make_session()
    experiment = sql_log.Experiment(dataset='power_plant')
    for score in grid_search.grid_scores_:
        validation_mse = -score.mean_validation_score
        params = score.parameters
        regularization_lambda = params['lambdaReg']
        regularization_type = params['typeReg']
        result = sql_log.Result(level=level, T=T, regularization_lambda=regularization_lambda,
                                regularization_type=regularization_type, validation_mse=validation_mse,
                                experiment=experiment)
        session.add(result)

    best_learner = grid_search.best_estimator_
    best_learner.fit(X_train, y_train) #refit with entire training set
    train_mse = best_learner.score(X_train, y_train)
    train_r2 = get_r_squared(best_learner, X_train, y_train)
    test_mse = best_learner.score(X_test, y_test)
    test_r2 = get_r_squared(best_learner, X_test, y_test)

    experiment.train_mse = train_mse
    experiment.train_r2 = train_r2
    experiment.test_mse = test_mse
    experiment.test_r2 = test_r2
    session.add(experiment)

    result_train = "Best learner was {} with an (training) MSE of {} and an(training) r^2 of {}.".format(str(learner), train_mse, train_r2)
    result_test =  "Best learner got an testing MSE of {} and a testing r^2 of {}".format(test_mse, test_r2)

    _log.info(result_train)
    _log.info(result_test)


    _log.debug("Pushing to DB!")
    session.commit()

    _log.debug("Finished experiment!")
