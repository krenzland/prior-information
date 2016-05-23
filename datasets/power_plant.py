#!/usr/bin/env python

from tools import *

def main():
    dir = sys.argv[1]
    res_dir = dir + '/processed/power_plant/'
    raw_dir = dir + '/raw/power_plant/'
    input = raw_dir + '/Folds5x2_pp.xlsx'
    output_csv = res_dir + 'power_plant_prep.csv'

    df = pd.read_excel(input)

    _, df = scale(df)
    lambdas = {'AP': 0,
                'AT': 1.3353837296219406,
                u'PE': 1,
                'RH': 2.4604945158589104,
                'V': -0.43989911518156471}
    df = transform_cox(df, lambdas)

    df.to_csv(output_csv, index=False)

if __name__ == '__main__':
    main()
