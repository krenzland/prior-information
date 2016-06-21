#!/usr/bin/env python

import sys
sys.path.append("../src/")
from sgpi.util import scale, transform_cox
import pandas as pd

def main():
    dir = sys.argv[1]
    res_dir = dir + '/processed/abalone/'
    raw_dir = dir + '/raw/abalone/'
    input = raw_dir + '/abalone.data.txt'
    output_csv = res_dir + 'abalone_prep.csv'
    names = ['sex', 'length' , 'diameter', 'height', 'whole_weight', 'shucked_weight',
             'viscera_weight', 'shell_weight', 'rings']
    df = pd.read_csv('../datasets/raw/abalone/abalone.data.txt', names=names)
    df = pd.get_dummies(df)
    cols = ['length', 'diameter', 'height', 'whole_weight', 'shucked_weight',
            'viscera_weight', 'shell_weight', 'sex_F', 'sex_I', 'sex_M', 'rings']
    df = df[cols] # reorder columns after dummies, target is always last!
    df = df[cols]
    _, df = scale(df)

    lambdas = { 'diameter': 3.2163821798943193,
                'height': -1.9914054656190996,
                'length': 3.3648670820916347,
                'sex_F': 1,
                'sex_I': 1,
                'sex_M': 1,
                'shell_weight': -1.1448450483002799,
                'shucked_weight': -1.4107540627818103,
                'viscera_weight': -1.2542787866665741,
                'whole_weight': -0.76682358269227757}

    df = transform_cox(df, lambdas)
    df.to_csv(output_csv, index=False)

if __name__ == '__main__':
    main()
