#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import pandas as pd

def write_arff(df, name, filename=None):
    relation = "@RELATION {}".format(name)
    attributes = "\n".join("@ATTRIBUTE {} numeric".format(col) for col in df)
    data = "@DATA"
    data_str = df.to_csv(header=False)
    arff_string = "\n\n".join([relation, attributes, data, data_str])
    if filename:
        with open(filename, "w") as fp:
            fp.write(arff_string)
    else:
        return arff_string

def main():
    input = "yeast.csv"
    output = "yeast.arff"
    df = pd.read_csv(input)
    y = df.ix[:,-1:]
    y_dummies = pd.get_dummies(y)
    df.ix[:,-1:]= y_dummies.values.argmax(1)
    #print(write_arff(df = df, name = "yeast"))
    write_arff(df=df, name="yeast", filename="yeast.arff")

if __name__ == "__main__":
    main()
