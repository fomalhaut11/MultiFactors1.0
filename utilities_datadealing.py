import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, t, shapiro, norm


def Remove_Outlier(input_x, method="mean", para=3):
    # 去极值
    x = input_x.astype(
        float
    )  # 使用.astype(float)将数据转换为浮点型，则是已经创建了一个新的对象。
    if isinstance(x, np.ndarray):
        xmax = np.nanmax(x)
        xmin = np.nanmin(x)
        xmedian = np.nanmedian(x)
        x[np.isposinf(x)] = xmax
        x[np.isneginf(x)] = xmin
        x[np.isnan(x)] = xmedian
        if method == "IQR":
            medianvalue = np.nanmedian(x)
            Q1 = np.percentile(x, 5)
            Q3 = np.percentile(x, 95)
            IQR = Q3 - Q1
            x[x > Q3 + para * IQR] = Q3 + para * IQR
            x[x < Q1 - para * IQR] = Q1 - para * IQR
        if method == "median":
            medianvalue = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - medianvalue))
            x[x - medianvalue > para * mad] = para * mad
            x[x - medianvalue < -para * mad] = -para * mad
        elif method == "mean":
            meanvalue = np.nanmean(x)
            std = np.std(x)
            x[x - meanvalue > para * std] = para * std
            x[x - meanvalue < -para * std] = -para * std
    else:
        x = x.copy()
        if method == "IQR":
            medianvalue = x[np.isfinite(x)].median()
            Q1 = np.percentile(x[np.isfinite(x)], 5)
            Q3 = np.percentile(x[np.isfinite(x)], 95)
            x.fillna(medianvalue, inplace=True)
            IQR = Q3 - Q1
            x[x > Q3 + para * IQR] = Q3 + para * IQR
            x[x < Q1 - para * IQR] = Q1 - para * IQR
        if method == "median":
            medianvalue = x[np.isfinite(x)].median()
            mad = np.nanmedian(np.abs(x - medianvalue))
            x.fillna(medianvalue, inplace=True)
            x[x - medianvalue > para * mad] = para * mad
            x[x - medianvalue < -para * mad] = -para * mad
        elif method == "mean":
            meanvalue = x[np.isfinite(x)].mean(axis=0)
            std = np.std(x[np.isfinite(x)], axis=0)
            x.fillna(meanvalue, inplace=True)
            x[x - meanvalue > para * std] = para * std
            x[x - meanvalue < -para * std] = -para * std
    return x


def Normlization(x_input, method="zscore"):
    # 标准化
    x = x_input.copy()
    if isinstance(x, pd.core.frame.DataFrame):
        if x.std().item() <= 0.000001:
            x = x * 0
        else:
            if method == "zscore":
                x = (x - x.mean()) / x.std()
            if method == "ppf":
                cdf_values = (np.argsort(np.argsort(x)) + 0.5) / len(x)
                x = norm.ppf(cdf_values)
    if isinstance(x, pd.core.series.Series):
        if x.std() <= 0.000001:
            x = x * 0
        else:
            if method == "zscore":
                x = (x - x.mean()) / x.std()
            if method == "ppf":
                cdf_values = (np.argsort(np.argsort(x)) + 0.5) / len(x)
                x = norm.ppf(cdf_values)
    if isinstance(x, np.ndarray):
        if np.std(x) <= 0.000001:
            x = x * 0
        else:
            if method == "zscore":
                x = (x - np.mean(x)) / np.std(x)
            if method == "ppf":
                cdf_values = (np.argsort(np.argsort(x)) + 0.5) / len(x)
                x = norm.ppf(cdf_values)
    return x


def RollingZscores(vec, len1):
    rz = np.zeros(len(vec))
    for i in range(len(rz)):
        if i < len1:
            continue
        try:
            mean1 = np.nanmean(vec[i - len1 : i + 1])
            std1 = np.nanstd(vec[i - len1 : i + 1]) + 0.0000001
            rz[i] = (vec[i] - mean1) / std1
            if rz[i] > 5:
                rz[i] = 5
            if rz[i] < -5:
                rz[i] = -5
        except:
            rz[i] = 0
    return rz