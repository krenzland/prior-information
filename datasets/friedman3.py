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
    output_csv = dir + '/friedman3/friedman3_prep.csv'
    names = ["x1", "x2", "x3", "x4", "y"]

    (X,y) = data.make_friedman3(n_samples=10000, random_state=123456, noise=0.01)
    y = np.matrix(y).T
    df = pd.DataFrame(np.append(X, y, axis=1), columns=names)
    df = scale(df)[1]

    # TODO Transform box-cox.
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    main()
