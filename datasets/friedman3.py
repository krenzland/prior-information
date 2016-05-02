#!/usr/bin/env python
from __future__ import print_function
import sklearn.datasets as data
import sklearn.preprocessing as pre
import numpy as np

def write_friedman_with_seed(filename, n, seed, scaler=None):
    print("Generating {} \t with {} \t samples and seed {}.".format(filename, n, seed))
    header = """@RELATION friedmann3

@ATTRIBUTE x1 numeric
@ATTRIBUTE x2 numeric
@ATTRIBUTE x3 numeric
@ATTRIBUTE x4 numeric
@ATTRIBUTE y numeric

@DATA
"""
    (X,Y) = data.make_friedman3(n_samples=n, random_state=seed, noise=0.01)
    if scaler:
        # already created a minmax-scaler, use it
        X = scaler.transform(X)
    else:
        # create a new scaler
        scaler = pre.MinMaxScaler()
        X = scaler.fit_transform(X)
    with open(filename, "w") as fp:
        fp.write(header)
        for xs, y in zip(X,Y):
            row = ",".join([str(x) for x in xs]) + "," + str(y)
            fp.write("{}\n".format(row))
    return scaler

def main():
    #  note: to generate comparable files, use seed 123456 for training, 234567 for testing, and 345678 for validation
    nSamples = 10000
    basename = "../SGpp/datadriven/tests/data/friedman3_10k_"
    files = [("train", 123456), ("validation", 345678), ("testing", 234567)]
    scaler = None
    for (name, seed) in files:
        filename = basename + name + ".arff"
        scaler = write_friedman_with_seed(filename, nSamples, seed, scaler)

if __name__ == "__main__":
    main()
