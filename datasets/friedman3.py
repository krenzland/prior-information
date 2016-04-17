#!/usr/bin/env python
from __future__ import print_function
import sklearn.datasets as data
import numpy as np

def write_friedman_with_seed(filename,n, seed):
    print("Generating {} \t with {} \t samples and seed {}.".format(filename, n, seed))
    header = """@RELATION friedmann3

@ATTRIBUTE x1 numeric
@ATTRIBUTE x2 numeric
@ATTRIBUTE x3 numeric
@ATTRIBUTE x4 numeric
@ATTRIBUTE y numeric

@DATA
"""
    (X,Y) = data.make_friedman3(n_samples=n, random_state=seed)
    with open(filename, "w") as fp:
        fp.write(header)
        for xs, y in zip(X,Y):
            row = ",".join([str(x) for x in xs]) + "," + str(y)
            fp.write("{}\n".format(row))

def main():
    #  note: to generate comparable files, use seed 123456 for training, 234567 for testing, and 345678 for validation
    nSamples = 10000
    basename = "../SGpp/datadriven/tests/data/friedman3_10k_"
    files = [("train", 123456), ("validation", 345678)]
    for (name, seed) in files:
        filename = basename + name + ".arff"
        write_friedman_with_seed(filename, nSamples, seed)

if __name__ == "__main__":
    main()
