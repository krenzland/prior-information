#!/usr/bin/env python

import sys
sys.path.append("../src/")
from sgpi.util import scale, transform_cox, last_col_to_numeric, write_arff
import pandas as pd


def main():
    dir = sys.argv[1] + "/yeast/"
    input = dir + 'yeast_conv.csv'
    output_csv = dir + 'yeast_prep.csv'
    output_arff = dir + 'yeast.arff'
    df = pd.read_csv(input)
    last_col_to_numeric(df)
    write_arff(df=df, name='yeast', filename=output_arff)
    df.to_csv(output_csv, index=False)

if __name__ == '__main__':
    main()
