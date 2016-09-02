#!/usr/bin/env python

import sys
sys.path.append("../src/")
from sgpi.util import scale, transform_cox, last_col_to_numeric, write_arff
import pandas as pd

def main():
    dir = sys.argv[1]
    res_dir = dir + '/processed/mnist/'
    raw_dir = dir + '/raw/mnist/'

    columns = ["x{}".format(i) for i in range(0, 64)] + ['digit']
    df_train = pd.read_csv(raw_dir + "optdigits.tra", header=None,index_col=None)
    df_test = pd.read_csv(raw_dir + "optdigits.tes", header=None,index_col=None)
    df_train.columns = columns
    df_test.columns = columns

    # Create a scaler using the complete dataset.
    # Division by 16 also works, but the columns do not all have a maximum of 16,
    # so this results in a better distribution.
    df_complete = df_train.append(df_test, ignore_index=True)
    scaler , _ = scale(df_complete)
    _, df_train = scale(df_train, scaler)
    _, df_test = scale(df_test, scaler)
    # We do not perform a box-cox transformation here, as the features are either
    # sensible distributed or very sparse.

    df_train.to_csv(res_dir + 'optdigits_train.csv', index=False)
    df_test.to_csv(res_dir + 'optdigits_test.csv', index=False)

    # We use the following subsample to estimate parameters.
    df_train = df_train[(df_train['digit'] == 2) | (df_train['digit'] == 7) | (df_train['digit'] == 9)]
    df_test = df_test[(df_test['digit'] == 2) | (df_test['digit'] == 7) | (df_test['digit'] == 9)]
    df_train.to_csv(res_dir + 'optdigits_sub_train.csv', index=False)
    df_test.to_csv(res_dir + 'optdigits_sub_test.csv', index=False)


if __name__ == '__main__':
    main()
