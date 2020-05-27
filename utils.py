
import random
import time, datetime
import os
from functools import reduce
import subprocess
from datetime import datetime, timedelta
from functools import reduce
import pandas as pd
import pathlib
from dfply import *
from lifetimes import BetaGeoFitter
import seaborn as sb
import matplotlib.pyplot as plt
import scipy
import numpy as np
from sklearn.metrics import mean_absolute_error



def load_customer_alive_history(category_id):
    
    import pandas as pd
    import glob

    path = r'/tmp/data/pengcheng/cltv-customer/' + str(category_id) +'.csv/' # use your path
    all_files = glob.glob(path + "/*.csv")

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    cst = pd.concat(li, axis=0, ignore_index=True)
    print(cst.head(2))
    cst['dt'] = pd.to_datetime(cst.dt)
    return cst


def customer_lifetime_value(ggf, bgf, part,
                                 time = 6, discount_rate=0.1, freq='D'):
    
    
    df = pd.DataFrame(index=part['frequency_cal'].index)
    df["clv"] = 0  # initialize the clv column to zeros

    steps = np.arange(1, time + 1)
    factor = {"W": 4.345, "M": 1.0, "D": 30, "H": 30 * 24}[freq]
    
    adjusted_monetary_value = ggf.conditional_expected_average_profit(
        part['frequency_cal'],
        part['monetary_value']
    )

    for i in steps * factor:
        # since the prediction of number of transactions is cumulative, we have to subtract off the previous periods
        part['predicted_purchases'] = bgf.predict(
            i, part['frequency_cal'], part['recency_cal'], part['T_cal']
        ) - bgf.predict(i - factor, part['frequency_cal'], part['recency_cal'], part['T_cal'])
        
        adjust_frequency_prediction(part)
        
        # sum up the CLV estimates of all of the periods and apply discounted cash flow
        df["clv"] += (adjusted_monetary_value * part['predicted_purchases']) / (1 + discount_rate) ** (i / factor)

    return df["clv"] # return as a series



def adjust_frequency_prediction(part):
    #print(part.predicted_purchases.quantile([0.1, .25, .5, .75, .85, .95, .99, .995]))
    part['predicted_purchases'].fillna(0, inplace=True)
    part.loc[part['predicted_purchases'] < 1, 'predicted_purchases'] = 0
    print(part.predicted_purchases.quantile([0.1, .25, .5, .75, .85, .95, .99, .995]))


def _plot_calibration_purchases_vs_holdout_purchases(df, n):
    
    x_labels = {
        "frequency_cal": "Purchases in calibration period",
        "recency_cal": "Age of customer at last purchase",
        "T_cal": "Age of customer at the end of calibration period",
        "time_since_last_purchase": "Time since user made last purchase",
    }
    
    ax = df.groupby('frequency_cal')[["frequency_holdout", "predicted_purchases"]].mean().iloc[:n].plot()
    plt.title("Actual Purchases in Holdout Period vs Predicted Purchases")
    plt.xlabel(x_labels['frequency_cal'])
    plt.ylabel("Average of Purchases in Holdout Period")
    plt.legend()



def perf_trax_freq_by_interval(df, interval, threshold=40, adjusted=False):
    
    def f(row):
        if row['predicted_purchases'] < 0.5:
            val = 0
        elif row['predicted_purchases'] > threshold and adjusted:
            val = threshold
        else:
            val = row['predicted_purchases']
        return val
    
    def measure(row, c1, c2):
        if abs(row[c1]-row[c2]) < interval:
            return 1
        else:
            return 0
        
    def alive(row):
        if row['palive'] < 0.3 and row['frequency_holdout'] == 0:
            return 1
        elif row['palive'] >= 0.3 and row['frequency_holdout'] > 0:
            return 1
        else:
            return 0

    df['pred'] = df.apply(f, axis=1)
    df['pred_in'] = df.apply(measure, args=('frequency_holdout', 'pred'), axis=1)
    df['train_in'] = df.apply(measure, args=('frequency_holdout', 'frequency_cal'), axis=1)
    df['alive'] = df.apply(alive, axis=1)
    
    return (mean_absolute_error(df['frequency_holdout'], df['pred']),
            df['pred_in'].sum()/len(df),
            df['alive'].sum()/len(df),
            mean_absolute_error(df['frequency_holdout'], df['frequency_cal']),
            df['train_in'].sum()/len(df))


def perf_trax_freq(df):
    
    def f(row):
        if row['predicted_purchases'] < 0.5:
            val = 0
        elif row['predicted_purchases'] > 25:
            val = 25
        else:
            val = row['predicted_purchases']
        return val

    df['pred'] = df.apply(f, axis=1)
    
    return mean_absolute_error(df['frequency_holdout'], df['pred']), df


def rank(df, sort_col, asc=False):
    t = df[['id', 'clv', 'total_value', 'total_holdout']]
    t.fillna(0.0, inplace=True)
    return t.sort_values(by=[sort_col], ascending=asc)


def jaccard(s1, s2, top=500000):
    g1 = set(s1.iloc[:top].tolist())
    g2 = set(s2.iloc[:top].tolist())
    intersection = g1.intersection(g2)
    union = g1.union(g2)
    jaccard = len(intersection)/float(len(union))
    return jaccard, top


def mape(m1, m2):
        
    lst=[]
    for i in range(len(m1)):
        g1 = m1[i]
        g2 = m2[i]
        lst.append((g1, g2, abs(g1-g2)/g2 if g2 !=0 else (g1-g2)))
           
    return lst





def get_bins(df, col):
    
    r = rank(df[df[col] > 0], col, True)
    out, bs = pd.qcut(r[col], q=10, precision=2, duplicates='raise', retbins=True)
    
    bs = list(bs)
    bs[0] = 0
    
    return [round(x, 3) for x in bs]



def bucket_mean(df, bucket_col, col1, col2, col3, bin_col=None, bins=None):
    
    if not bins:
        bins = get_bins(df, 'clv')
    
    def bucket(row, col):
        for i in range(len(bins)):
            if row[col] <= bins[i]:
                return i
        return len(bins)-1

    r = rank(df, bucket_col, True)
    
    if not bin_col:
        bin_col = bucket_col+'_decile'
     
    r[bin_col]=r.apply(bucket, args=(bucket_col,), axis=1)
    
    m1 = r.groupby([bin_col])[col1].agg(['mean','count'])
    m1['bins'] = bins
    
    m2 = r.groupby([bin_col])[col2].agg(['mean','count'])
    m2['bins'] = bins
    
    m3 = r.groupby([bin_col])[col3].agg(['mean','count'])
    m3['bins'] = bins
    
    churn = r[r[bin_col] == 0]
    alive = r[r[bin_col] != 0]
    print('TPR: ', len((r[(r[bin_col] == 0) & (r[col2] == 0)])) / float(len(churn)))
    print('FPR: ', len((r[(r[bin_col] != 0) & (r[col2] == 0)])) / float(len(alive)))
    
    return m1, m2, m3


def plot_bucket_mean_(m1, m2, label1, label2):
    
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = int(rect.get_height())
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 2),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
    fig, ax = plt.subplots()  # Create a figure and an axes.
    
    labels = np.arange(len(m1))
    x = labels  # the label locations
    width = 0.35
    rects1 = ax.bar(x - width/2, m1['mean'], width, label=label1)
    rects2 = ax.bar(x + width/2, m2['mean'], width, label=label2)
    #rects2 = ax.bar(x + width/2, m3['mean'], width, label=dcol3)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('CLTV Mean')
    ax.set_xlabel('holdout_cltv_decile')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()


def plot_bucket_mean_diff(lst1, lst2):
    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot([x[2] for x in lst1], label='prediction_testing')  # Plot some data on the axes.
    ax.plot([x[2] for x in lst2], label='training_testing')  # Plot more data on the axes...
    ax.set_xlabel('mean absolute percentage error')  # Add an x-label to the axes.
    ax.set_ylabel('number')  # Add a y-label to the axes.
    ax.set_title("plot_bucket_mean_diff")  # Add a title to the axes.
    ax.legend(loc='upper right')


def spearman_cor(df, col1, col2):
    df[col1].fillna(0.0, inplace=True)
    df[col2].fillna(0.0, inplace=True)
    
    r = df[(df[col1]!=0) | (df[col2]!=0)]

    print(len(df), len(r), len(r)/float(len(df)))
    return scipy.stats.spearmanr(r[col1], r[col2])



def alive(row):
        if row['palive'] >= 0.5:
            return 1
        else:
            return 0


### clv:   value of 'ggf.customer_lifetime_value'
### clv_1: use alive rate to calibrate final clv value
### clv_2: use outputs of GEO model and Gamma model compute clv, 
###        not use 'ggf.customer_lifetime_value'
def clv_perf(training_money, c1, c2, c3, threshold=0):
    
    training_money[c1].fillna(0.0, inplace=True)
    training_money_ = training_money[training_money['clv'] <= threshold]
    #training_money_[c1] = training_money[c1]
    #training_money[c1] = training_money[c2]*(training_money[c3]+1)
    training_money_['clv_1'] = training_money_['clv']*training_money_['alive_status']
    training_money_.loc[training_money_['clv_1'] < 0, 'clv_1'] = 0
    
    print("clv")
    #print(np.sqrt(mean_squared_error(training_money_['clv'], training_money_[c1])))
    print('MAE:', mean_absolute_error(training_money_['clv'], training_money_[c1]), '\n')
    
    print("clv*alive")
    #print(np.sqrt(mean_squared_error(training_money_['clv_1'], training_money_[c1])))
    print('MAE:', mean_absolute_error(training_money_['clv_1'], training_money_[c1]), '\n')
    
    training_money_['clv_2'] = training_money_['avenue']*(training_money_['pred'])*training_money_['alive_status']
    training_money_.loc[training_money_['clv_2'] < 0, 'clv_2'] = 0

    print("ave*pred*alive")
    #print(np.sqrt(mean_squared_error(training_money_['clv_2'], training_money_[c1])))
    print('MAE:', mean_absolute_error(training_money_['clv_2'], training_money_[c1]), '\n')
    
    training_money_['clv_3'] = training_money_['avenue']*(training_money_['pred'])
    training_money_.loc[training_money_['clv_3'] < 0, 'clv_3'] = 0
    print("ave*pred")
    print('MAE:', mean_absolute_error(training_money_['clv_3'], training_money_[c1]))
    
    return training_money_


def plot_percentile_cltv_1(df, col1, col2, tor1=1, tor2=1):
    
    df1 = df[df[col1] < df[col1].quantile(tor1)]
    r1 = rank(df1, col1)
    r1 = r1.reset_index()
    r1['nid'] = r1.index
    x1 = r1['nid'].rank(pct=True).values
    y1 = r1[col1].rank(method='dense', pct=True, ascending=False).values
    plt.figure()
    plt.subplot(211)
    plt.plot(x1, y1, 'b')    
    
    df2 = df[df[col2] < df[col2].quantile(tor2)]
    r2 = rank(df2, col2)
    r2 = r2.reset_index()
    r2['nid'] = r2.index
    x2 = r2['nid'].rank(pct=True).values
    y2 = r2[col2].rank(method='dense', pct=True, ascending=False).values
    plt.subplot(212)
    plt.plot(x2, y2, 'r')



def cum_percent_2(df, col, asc=False, method='dense'): 
    return df[col].rank(method=method, pct=True, ascending=asc).values



def plot_percentile_cltv(df, col1, col2, col3, tor1=1, tor2=1, tor3=1, method='dense'):
    
    def cum_percent_1(df, col):  
        return df[col].cumsum()/df[col].sum()
    
    def compute(col, tor):
        sdf = df[df[col] < df[col].quantile(tor)]
        r1 = rank(sdf, col)
        r1 = r1.reset_index()
        r1['nid'] = r1.index
        x = cum_percent_1(r1, 'nid')
        y = cum_percent_1(r1, col)      
        return x, y
    
    x1, y1 = compute(col1, tor1)
    x2, y2 = compute(col2, tor2)  
    x3, y3 = compute(col3, tor3)
    
    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(x1, y1, label='predicted_cltv')  # Plot some data on the axes.
    ax.plot(x2, y2, label='holdout_cltv')  # Plot more data on the axes...
    ax.plot(x3, y3, label='baseline_cltv')  # Plot more data on the axes...
    ax.set_xlabel('customer percentile')  # Add an x-label to the axes.
    ax.set_ylabel('cltv percentile')  # Add a y-label to the axes.
    ax.set_title("plot_percentile_cltv")  # Add a title to the axes.
    ax.legend(loc='lower right')
    
    return (x1, y1, x2, y2, x3, y3)



# +
## https://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
## https://github.com/oliviaguest/gini

def arary_gini(array):
        array = array.flatten()
        if np.amin(array) < 0:
            # Values cannot be negative:
            array -= np.amin(array)
        # Values cannot be 0:
        array += 0.0000001
        # Values must be sorted:
        array = np.sort(array)
        # Index per array element:
        index = np.arange(1,array.shape[0]+1)
        # Number of array elements:
        n = array.shape[0]
        # Gini coefficient:
        return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))
    


# -

def gini(df, col1, col2):
    return (arary_gini(df[col1].values), arary_gini(df[col2].values))


