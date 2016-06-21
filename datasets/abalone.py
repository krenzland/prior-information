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

    # Fix obvious data entry errors
    df = df[df.height != 0]
    ix = df[df.height == 1.13].index
    df = df.set_value(ix, 'height', 0.130) # abalone is rather small, 0.13 seems plausible

    df = pd.get_dummies(df)
    cols = ['length', 'diameter', 'height', 'whole_weight', 'shucked_weight',
            'viscera_weight', 'shell_weight', 'sex_F', 'sex_I', 'sex_M', 'rings']
    df = df[cols] # reorder columns after dummies, target is always last!
    _, df = scale(df)

    lambdas = {'diameter': 3.2194196832186464,
                'height': 1.2086273383776132,
                'length': 3.3682627709337996,
                'rings': 1,
                'sex_F': 1,
                'sex_I': 1,
                'sex_M': 1,
                'shell_weight': -1.1449045766430477,
                'shucked_weight': -1.4088820347405902,
                'viscera_weight': -1.2520473863735537,
                'whole_weight': -0.7649280664662178}

    df = transform_cox(df, lambdas)
    df.to_csv(output_csv, index=False)

if __name__ == '__main__':
    main()
