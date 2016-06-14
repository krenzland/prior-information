import os
import numpy as np; np.random.seed(42)
import pandas as pd
import sklearn.preprocessing as pre
import sklearn.cross_validation as cv
from scipy import stats
from pysgpp import DataMatrix, DataVector

def to_data_matrix(arr):
    (size_x, size_y) = arr.shape
    matrix = DataMatrix(size_x, size_y)
    cur_row = 0
    for x in arr:
        x_vec = DataVector(x.tolist())
        matrix.setRow(cur_row,x_vec)
        cur_row += 1
    return matrix

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

def transform_cox(df, lambdas):
    scaler = pre.MinMaxScaler()
    for variable in lambdas:
        lamb = lambdas[variable]
        if lamb == 1:
            continue; # identity transform
        data_trans = stats.boxcox(df[variable] + 10e-1)
        df[variable] = scaler.fit_transform(data_trans[0])
    return df

def get_r_squared(learner, X, y):
    mse = -learner.score(X,y)
    ss_reg = mse * y.size
    ss_tot = np.var(y) * y.size
    return 1 - ss_reg/ss_tot

def get_dataset(name):
    package_directory = os.path.dirname(os.path.abspath(__file__))
    datasets = {
        'concrete': 'concrete/concrete_prep.csv',
        'power_plant': 'power_plant/power_plant_prep.csv'
    }
    folder = os.path.join(package_directory, '../../datasets/processed/')
    path = os.path.join(folder, datasets[name])
    return pd.read_csv(path)

def get_xy(data):
    X = np.array(data.ix[:,0:-1])
    y = data.ix[:,-1]
    return X,y
