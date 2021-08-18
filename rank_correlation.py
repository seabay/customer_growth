# -*- coding: utf-8 -*-
from pyspark.sql.functions import udf
from pyspark.sql.types import *
import pyspark.sql.functions as F
from scipy.stats import kendalltau


# # Correlation metrcis

# ## python method

# use Kendall’s tau measure the correlation of ordinal data
def compute_kenall(x, y):
    corr, p = kendalltau(x, y)
    return float(corr)


# ## spark method

def compute_kenall_udf(expected_rank):
    return udf(lambda x: compute_kenall(x, expected_rank), DoubleType())


def compute_rank_correlation(df, rank_col, expected_rank):
    corrs = (df
             .withColumn("kendall", compute_kenall_udf(expected_rank)(rank_col))
             .select('customer_id', 'kendall'))
    print(corrs.count())
    return corrs, aggrage_rank_correlation(corrs)


# # Correlation aggregation

# ## spark method

def aggrage_rank_correlation(corr):
    return corr.approxQuantile("kendall", [.1, .2, .3, .4, .5, .6, .7, .8, .9, .99], 1e-4)


