# coding=utf8

from __future__ import print_function
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cox(name, X, y):
    data_trans = stats.boxcox(X + 10e-1)

    data_scaled = data_trans[0].reshape(-1,1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_scaled)

    fig = plt.figure(figsize=(14, 7))
    f, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=False)

    sns.distplot(X, ax=axes[0][0], axlabel="")
    axes[0][0].set_title("raw data for " + name)

    axes[0][1].scatter(X,y)

    sns.distplot(data_scaled, ax=axes[1][0], label="")
    axes[1][0].set_title(u"λ = " + str(data_trans[1]))
    axes[1][1].scatter(data_scaled, y)

    fig.show()

    return data_trans[1]

def plot_cox_df(df):
    lambdas = {}
    for x in df:
        try:
            lamb = plot_cox(str(x), df[x], df.ix[:,-1])
            lambdas[str(x)] = lamb
        except Exception, e:
            print("Error whilst processing {}: {}".format(x, e))
    last_name = list(df)[-1]
    lambdas[last_name] = 1 # don't transform last col!
    return lambdas
