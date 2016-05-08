from __future__ import print_function
import sys
import numpy as np
import pandas as pd
import sklearn.cross_validation as cv
import sklearn.preprocessing as pre

def write_arff(df, name, filename=None):
    relation = '@RELATION {}'.format(name)
    attributes = '\n'.join('@ATTRIBUTE {} numeric'.format(col) for col in df)
    data = '@DATA'
    data_str = df.to_csv(header=False)
    arff_string = '\n\n'.join([relation, attributes, data, data_str])
    if filename:
        with open(filename, 'w') as fp:
            fp.write(arff_string)
    else:
        return arff_string

def last_col_to_numeric(df):
    y = df.ix[:,-1:]
    y_dummies = pd.get_dummies(y)
    df.ix[:,-1:]= y_dummies.values.argmax(1)

def split(df, test_size=0.2):
    train, test = cv.train_test_split(df, test_size=test_size, random_state=1337)
    return train, test

def scale(df, scaler=None):
    Y = df.ix[:,-1] # save Y (don't need to transform it/useless for cat. data!)
    X = df.values
    if scaler:
        X = scaler.transform(X)
    else:
        scaler = pre.MinMaxScaler()
        X = scaler.fit_transform(X) 
    index = df.index
    columns = df.columns
    df = pd.DataFrame(data=X, index=index, columns=columns)
    df.ix[:,-1] = Y
    return scaler, df
