#! /usr/bin/env python
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'src/')))
from sgpi.util import get_dataset, get_xy, get_Phi, get_max_lambda, calculate_weight_path
from sgpi import model
from sgpi.learner import SGRegressionLearner
from sgpi.grid_search import GridSearch
import numpy as np
import pandas as pd

import pysgpp as sg

def get_estimator(grid_config, X, y, l1_ratio, reg_type):
    level = grid_config.level
    dim = X.shape[1]
    grid = sg.Grid.createModLinearGrid(dim)
    gen = grid.getGenerator()
    gen.regular(level)

    Phi = get_Phi(grid, X)
    max_lambda = get_max_lambda(Phi, y, grid.getSize(), X.shape[0], l1_ratio)
    regularization_config = model.RegularizationConfig(type=reg_type, exponent_base=1, lambda_reg=max_lambda, l1_ratio=l1_ratio)
    adaptivity_config = model.AdaptivityConfig(num_refinements=0, no_points=0, treshold=0.0, percent=0.0)
    solv_type = sg.SLESolverType_FISTA
    solver_config = model.SolverConfig(type=solv_type, max_iterations=600, epsilon=0, threshold=np.finfo(np.double).eps)
    final_solver_config = solver_config
    estimator = SGRegressionLearner(grid_config, regularization_config, solver_config,
                                    final_solver_config, adaptivity_config)

    return estimator, Phi, max_lambda, grid

def lasso():
    df = get_dataset('friedman1')
    X, y = get_xy(df)

    level = 2
    l1_ratio = 1.0
    reg_type = sg.RegularizationType_ElasticNet
    grid_config = model.GridConfig(type=sg.GridType_ModLinear, T=0.0, level=level)
    estimator, Phi, max_lambda, grid = get_estimator(grid_config, X, y, l1_ratio, reg_type)
    df = calculate_weight_path(estimator, X, y, max_lambda, num_lambdas=50, verbose=1)

    df.to_csv('path-f1-l2-lasso.csv', index=True)

def group_lasso():
    df = get_dataset('friedman1')
    X, y = get_xy(df)

    level = 2
    l1_ratio = 1.0
    reg_type = sg.RegularizationType_GroupLasso
    grid_config = model.GridConfig(type=sg.GridType_ModLinear, T=0.0, level=level)
    estimator, Phi, max_lambda, grid = get_estimator(grid_config, X, y, l1_ratio, reg_type)
    df = calculate_weight_path(estimator, X, y, max_lambda, num_lambdas=50, verbose=1)

    df.to_csv('path-f1-l2-grp.csv', index=True)

def en():
    df = get_dataset('friedman1')
    X, y = get_xy(df)

    level = 2
    l1_ratio = 0.3
    reg_type = sg.RegularizationType_ElasticNet
    grid_config = model.GridConfig(type=sg.GridType_ModLinear, T=0.0, level=level)
    estimator, Phi, max_lambda, grid = get_estimator(grid_config, X, y, l1_ratio, reg_type)
    df = calculate_weight_path(estimator, X, y, max_lambda, num_lambdas=50, verbose=1)

    df.to_csv('path-f1-l2-en.csv', index=True)

def lasso_heat():
    df = get_dataset('friedman1')
    X, y = get_xy(df)

    level = 3
    l1_ratio = 1.0
    reg_type = sg.RegularizationType_ElasticNet
    grid_config = model.GridConfig(type=sg.GridType_ModLinear, T=0.0, level=level)
    estimator, Phi, max_lambda, grid = get_estimator(grid_config, X, y, l1_ratio, reg_type)
    df = calculate_weight_path(estimator, X, y, max_lambda, num_lambdas=10, verbose=1)

    df.to_csv('path-f1-l3-lasso.csv', index=True)

def grp_heat():
    df = get_dataset('friedman1')
    X, y = get_xy(df)

    level = 3
    l1_ratio = 1.0
    reg_type = sg.RegularizationType_GroupLasso
    grid_config = model.GridConfig(type=sg.GridType_ModLinear, T=0.0, level=level)
    estimator, Phi, max_lambda, grid = get_estimator(grid_config, X, y, l1_ratio, reg_type)
    df = calculate_weight_path(estimator, X, y, max_lambda, num_lambdas=10, verbose=1)

    df.to_csv('path-f1-l3-grp.csv', index=True)


def main():
    sg.omp_set_num_threads(4)
    lasso()
    print "Finished lasso"
    en()
    print "Finished en"
    group_lasso()
    print "Finished group lasso"
    lasso_heat()
    print "Finished lasso-heat"
    grp_heat()
    print "Finished grp-heat"

if __name__ == '__main__':
    main()

