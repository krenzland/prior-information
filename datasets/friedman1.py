#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append("../src/")

import sklearn.datasets as data
import sklearn.preprocessing as pre

import numpy as np
import pandas as pd

from sgpi.util import scale, transform_cox

def main():
    dir = sys.argv[1]
    output_csv = dir + '/friedman1/friedman1_prep.csv'
    names = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "y"]

    (X,y) = data.make_friedman1(n_samples=10000, random_state=123456, noise=1.0)
    y = np.matrix(y).T
    df = pd.DataFrame(np.append(X, y, axis=1), columns=names)
    df = scale(df)[1]

    # TODO Transform box-cox.
    lambdas = {'x1': 0.73772299748812553,
                'x10': 0.81728280581171431,
                'x2': 0.80698183857607453,
                'x3': 0.73814877672198154,
                'x4': 0.65907211104558194,
                'x5': 0.88664969513868797,
                'x6': 0.78156577216859524,
                'x7': 0.73707418190834051,
                'x8': 0.77589583265069417,
                'x9': 0.80351813801046301}
    df = transform_cox(df, lambdas)

    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    main()
