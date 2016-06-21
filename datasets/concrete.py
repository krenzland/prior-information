#!/usr/bin/env python
import sys
sys.path.append("../src/")

from sgpi.util import scale, transform_cox
import pandas as pd

def main():
    dir = sys.argv[1]
    res_dir = dir + '/processed/concrete/'
    raw_dir = dir + '/raw/concrete/'
    input = raw_dir + '/Concrete_Data.xls'
    output_csv = res_dir + 'concrete_prep.csv'

    names = ["cement", "blast_slag", "fly_ash", "water", "superplasticizer", "coarse_aggregate", "fine_aggregate",
            "age", "compressive_strength"]
    df = pd.read_excel(input)
    df.columns = names # read_excel ignores name parameter?

    _, df = scale(df)
    lambdas = {'age': -7.2461323674095635,
                'blast_slag': -2.7189584644377498,
                'cement': -0.6839391080474353,
                'coarse_aggregate': 1,
                'compressive_strength': 1,
                'fine_aggregate': 1.6052745658692371,
                'fly_ash': -1.9942706931166276,
                'superplasticizer': -1.4914459190124418,
                'water': 0.806604641002032}
    df = transform_cox(df, lambdas)

    df.to_csv(output_csv, index=False)

if __name__ == '__main__':
    main()
