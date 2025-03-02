import numpy as np
import pandas as pd
import os
import sys

sys.path.append(r"E:\Documents\PythonProject\StockProject")
import 股票数据接口_zx as FinData
from datetime import date
from datetime import datetime
import statsmodels.api as sm
from utilities_datadealing import RollingZscores


class FactorMatrix_Report(object):  # 和财务数据有关的因子计算
    def __init__(self, Stock3d=None, cla=None):
        # 初始化 因子生成的类，为了方便数据调用，默认初始化之后数据为空，需要调用data_loading函数加载数据
        if Stock3d is None:
            Stock3d = []
        if cla is None:
            cla = []

        if len(Stock3d) == 0:
            self.data_loading(begindate, int_today)
        else:
            self.Stock3d = Stock3d
            self.PriceMatrix_column = self.Stock3d["datacolumns"]
            self.ReportData = cla
            stockcodes0 = [
                str(s, encoding="utf-8") for s in self.ReportData.财报股票code
            ]
            code0, codei1, codei2 = np.intersect1d(
                Stock3d["StockCodes"], stockcodes0, return_indices=True
            )
            self.StockCodes = code0
            self.TradingDates = self.Stock3d["TradingDates"]
            self.Stock3d_idx_Report = codei1
            self.Report_idx_Stock3d = codei2
            self.PriceMatrix = self.Stock3d["pricematrix"][:, codei1, :]
            today = datetime.today()
            self.int_today = int(today.strftime("%Y%m%d"))

    def Title(self, SheetTitle, Item, begindate=20110101, enddate=[]):
        # 添加第一个财务指标的表名，以及指标名称，开始日期，结束日期
        self.SheetTitle = SheetTitle
        self.item = Item
        self.begindate = begindate
        if len(enddate) == 0:
            self.enddate = self.int_today

    def Title_2(self, SheetTitle, Item, begindate=20110101, enddate=[]):
        # 添加第2个财务指标的表名，以及指标名称，开始日期，结束日期
        self.SheetTitle_2 = SheetTitle
        self.item_2 = Item

    def Title_3(self, SheetTitle, Item, begindate=20110101, enddate=[]):
        """添加第3个财务指标的表名，以及指标名称，开始日期，结束日期
        可能将来有某些指标需要调用3个不同的财务栏目数据"""
        self.SheetTitle_3 = SheetTitle
        self.item_3 = Item

    def data_loading(self, begindate, enddate):
        # 从数据库中读取所有行情数据，所有的三张表的数据，以及股票价格数据，self.PriceMatrix是股票价格的顺序按三张表里股票名称排序
        cla = FinData.财报数据()
        cla.读取财报数据(begindate, enddate)
        self.ReportData = cla
        StockDataDF = GetStockDayDataDFFromSql()
        self.Stock3d = StockDataDF2Matrix(StockDataDF)
        stockcodes0 = [str(s, encoding="utf-8") for s in self.ReportData.财报股票code]
        code0, codei1, codei2 = np.intersect1d(
            self.Stock3d["StockCodes"], stockcodes0, return_indices=True
        )
        self.StockCodes = code0
        self.Stock3d_idx_Report = codei1
        self.Report_idx_Stock3d = codei2
        self.PriceMatrix = self.Stock3d["pricematrix"][:, codei1, :]

    def addData(self, Stock3d, cla):
        # 从内存中加载数据，不从数据库中加载数据
        self.ReportData = cla
        self.Stock3d = Stock3d
        stockcodes0 = [str(s, encoding="utf-8") for s in self.ReportData.财报股票code]
        code0, codei1, codei2 = np.intersect1d(
            Stock3d["StockCodes"], stockcodes0, return_indices=True
        )
        self.StockCodes = code0
        self.Stock3d_idx_Report = codei1
        self.Report_idx_Stock3d = codei2
        self.PriceMatrix = self.Stock3d["pricematrix"][:, codei1, :]

    def InitFactor(self):
        # 读取三张表中原始
        if self.SheetTitle == "fzb":
            ReportSheetData = self.ReportData.fzb
            list1 = self.ReportData.fzb_str
        elif self.SheetTitle == "lrb":
            ReportSheetData = self.ReportData.lrb
            list1 = self.ReportData.lrb_str
        elif self.SheetTitle == "xjlb":
            ReportSheetData = self.ReportData.xjlb
            list1 = self.ReportData.xjlb_str
        itemidx = np.where(list1 == self.item.encode("utf-8"))[0]
        templrb = self.ReportData.lrb[self.Report_idx_Stock3d, :, :]
        index = [0, 1, -3]
        self.initfactor_ReportDates = templrb[:, :, index]
        if len(itemidx) > 0:
            self.itemidx = itemidx[0]
        else:
            print("科目名错误")
            return
        return ReportSheetData[self.Report_idx_Stock3d, :, self.itemidx]

    def InitFactor_2(self):
        if self.SheetTitle_2 == "fzb":
            ReportSheetData_2 = self.ReportData.fzb
            list1 = self.ReportData.fzb_str
        elif self.SheetTitle_2 == "lrb":
            ReportSheetData_2 = self.ReportData.lrb
            list1 = self.ReportData.lrb_str
        elif self.SheetTitle_2 == "xjlb":
            ReportSheetData_2 = self.ReportData.xjlb
            list1 = self.ReportData.xjlb_str
        itemidx_2 = np.where(list1 == self.item_2.encode("utf-8"))[0]

        if len(itemidx_2) > 0:
            self.itemidx_2 = itemidx_2[0]
        else:
            print("科目名错误")
            return
        return ReportSheetData_2[self.Report_idx_Stock3d, :, self.itemidx_2]

    def InitFactor_3(self):
        if self.SheetTitle_3 == "fzb":
            ReportSheetData_3 = self.ReportData.fzb
            list1 = self.ReportData.fzb_str
        elif self.SheetTitle_3 == "lrb":
            ReportSheetData_3 = self.ReportData.lrb
            list1 = self.ReportData.lrb_str
        elif self.SheetTitle_3 == "xjlb":
            ReportSheetData_3 = self.ReportData.xjlb
            list1 = self.ReportData.xjlb_str
        item_3 = np.where(list1 == self.item_3.encode("utf-8"))[0]

        if len(item_3) > 0:
            self.itemidx_3 = item_3[0]
        else:
            print("科目名错误")
            return
        return ReportSheetData_3[self.Report_idx_Stock3d, :, self.itemidx_3]

    def montage_foreshow_NP_BELONGTO_PARCOMSH_YoY(self, forshowdata):
        # forshowdata=foreshowdata
        item = "NP_BELONGTO_PARCOMSH"
        itemidx = np.where(self.ReportData.lrb_str == item.encode("utf-8"))[0]
        data = self.ReportData.lrb[self.Report_idx_Stock3d, :, itemidx]
        datadate0 = self.ReportData.lrb[self.Report_idx_Stock3d, :, 0]  # 公布日期
        datadate1 = self.ReportData.lrb[self.Report_idx_Stock3d, :, 1]  # 报告期
        df = pd.DataFrame(
            data=data, index=self.StockCodes, columns=datadate1[0, :]
        ).unstack()
        df1 = pd.DataFrame(
            data=datadate0, index=self.StockCodes, columns=datadate1[0, :]
        ).unstack()
        df2 = pd.concat([df, df1], axis=1)
        df2.columns = ["value", "TradingDates"]
        df2.index.names = ["ReportDate", "StockCodes"]
        df3 = forshowdata.set_index(["reportday", "code"]).rename_axis(
            ["ReportDate", "StockCodes"]
        )
        dupl = df3.index.duplicated(keep="first")
        df3 = df3.loc[~dupl]
        df3 = df3.drop_duplicates(keep="last")
        df4 = pd.merge(df2, df3, left_index=True, right_index=True, how="left")
        Matrix = np.zeros((len(self.TradingDates), len(self.StockCodes))) * np.nan
        df2["mod"] = np.mod(df2.index.get_level_values(0), 1000)
        for i in range(len(self.StockCodes)):
            tempdata = df2.loc[df2.index.get_level_values(1) == self.StockCodes[i]]
            tempdata["pct_change"] = tempdata.groupby("mod").pct_change()
            for j in range(len(self.TradingDates)):
                pass

        df_l = df["l"].combine_first(df["value"])
        df_h = df["h"].combine_first(df["value"])
        df_d = df["tradingday"].combine_first(df["TradingDates"])
        df2 = pd.concat([df_l, df_h, df_d], axis=1)
        df2["mod"] = np.mod(df2.index.get_level_values(1), 1000)
        df2.sort_index(level=1)
        grouped = df2.groupby([df2.index.get_level_values(0), "mod"])["l"]
        df2["YoY_l"] = grouped.pct_change()
        grouped = df2.groupby([df2.index.get_level_values(0), "mod"])["h"]
        df2["YoY_h"] = grouped.pct_change()

        df2["max_YoY"] = df2[["YoY_l", "YoY_h"]].apply(
            lambda row: (
                row["YoY_l"] if abs(row["YoY_l"]) > abs(row["YoY_h"]) else row["YoY_h"]
            ),
            axis=1,
        )
        df2 = df2.reset_index()
        df3 = df2.set_index(["ReportDate", "StockCodes"])
        df3 = df3.sort_index()
        df3 = df3.groupby(level=[0, 1]).first()
        df4 = df3["max_YoY"].unstack()

        def cal_zscore(x):
            return (x[-1] - np.mean(x)) / (np.std(x) + 0.0000001)

        df2.groupby("StockCodes").rolling(window=4)["max_YoY"].apply(cal_zscore)
        df3 = df2[["StockCodes", "ReportDate", "max_YoY"]]
        df3.set_index(["ReportDate", "StockCodes"], inplace=True)
        df3.sort_index(level=0, inplace=True)
        return df3

    def apply_initial_data(self, func):
        # 对原始数据进行处理，func是处理函数，可以是累进数据，ttm数据，季度环比数据，同比数据，期末期初平均值数据
        if hasattr(self, "initfactor"):
            if hasattr(self, "initfactor_ReportDates"):
                s = np.shape(self.initfactor)
                if func == "ss":  # 累进数据单季度数据:非1季度数据都减去前一季度数据
                    print("ss")
                    data = np.zeros((s[0], s[1])) * np.nan
                    for stock in range(s[0]):
                        datesdata = self.initfactor_ReportDates[stock, :, 2]
                        for i in range(s[1]):
                            if i == 0:
                                continue
                            seasonlabel = datesdata[i]
                            if seasonlabel == 1:
                                data[stock, i] = self.initfactor[stock, i]
                            else:
                                data[stock, i] = (
                                    self.initfactor[stock, i]
                                    - self.initfactor[stock, i - 1]
                                )

                if func == "ttm":  # 累进数据ttm
                    data = np.zeros((s[0], s[1])) * np.nan
                    for stock in range(s[0]):
                        datesdata = self.initfactor_ReportDates[stock, :, 2]
                        for i in range(s[1]):
                            if i <= 4:
                                continue
                            seasonlabel = datesdata[i]
                            data0 = self.initfactor[stock, i]
                            data_1 = self.initfactor[stock, i - 1]
                            data_2 = self.initfactor[stock, i - 2]
                            data_3 = self.initfactor[stock, i - 3]
                            data_4 = self.initfactor[stock, i - 4]
                            if seasonlabel == 1:
                                data[stock, i] = data0 + data_1 - data_4
                            elif seasonlabel == 2:
                                data[stock, i] = data0 + data_2 - data_4
                            elif seasonlabel == 3:
                                data[stock, i] = data0 + data_3 - data_4
                            else:
                                data[stock, i] = data0
                if func == "avg":  # 期末期初平均值
                    data = np.zeros((s[0], s[1])) * np.nan
                    for stock in range(s[0]):
                        for i in range(s[1]):
                            if i < 1:
                                continue
                            data[stock, i] = (
                                self.initfactor[stock, i]
                                + self.initfactor[stock, i - 1]
                            ) / 2

                if func == "yoy":  # 累进数据同比数据
                    data = np.zeros((s[0], s[1])) * np.nan
                    for i in range(4):
                        for j in range(s[0]):
                            temp = self.initfactor[j, :]
                            temp1 = self.initfactor_ReportDates[j, :, 2]
                            temp2 = temp[np.argwhere(temp1 == i + 1)].flatten()
                            temp3 = np.diff(temp2) / temp2[0:-1]
                            data[j, np.argwhere(temp1 == i + 1)[1:]] = temp3.reshape(
                                (len(temp3), 1)
                            )

                if func == "qoq":  # 单季数据季度环比
                    data = np.zeros((s[0], s[1])) * np.nan
                    for stock in range(s[0]):
                        datesdata = self.initfactor_ReportDates[stock, :, 2]
                        for i in range(s[1]):
                            if i <= 1:
                                continue
                            seasonlabel = datesdata[i]
                            if seasonlabel == 1:
                                data0 = self.initfactor[stock, i]
                                data_1 = (
                                    self.initfactor[stock, i - 1]
                                    - self.initfactor[stock, i - 2]
                                )
                                data[stock, i] = data0 / data_1 - 1
                            elif seasonlabel == 2:
                                data0 = (
                                    self.initfactor[stock, i]
                                    - self.initfactor[stock, i - 1]
                                )
                                data_1 = self.initfactor[stock, i - 1]
                                data[stock, i] = data0 / data_1 - 1
                            else:
                                data0 = (
                                    self.initfactor[stock, i]
                                    - self.initfactor[stock, i - 1]
                                )
                                data_1 = (
                                    self.initfactor[stock, i - 1]
                                    - self.initfactor[stock, i - 2]
                                )
                                data[stock, i] = data0 / data_1 - 1

            else:
                print("initfactor_ReportDates,please add it ")
        else:
            print("initfactor Missing,please produce it")

        return data

    def TTMFactor(self):

        if hasattr(self, "ReportData"):
            FactorTTM = self.ReportData.fun_ttm(self.SheetTitle, self.item)
            initfactor = FactorTTM[self.Report_idx_Stock3d, :]
            templrb = self.ReportData.lrb[self.Report_idx_Stock3d, 3:, :]
            index = [0, 1, -3]
            self.initfactor_ReportDates = templrb[:, :, index]
            return FactorTTM
        else:
            print("ReportData Missing ,please addData")
            return

    def TTMFactor_2(self):
        if hasattr(self, "ReportData"):
            return self.ReportData.fun_ttm(self.SheetTitle_2, self.item_2)
        else:
            print("ReportData Missing ,please addData")
            return

    def TTMFactor_3(self):
        if hasattr(self, "ReportData"):
            return self.ReportData.fun_ttm(self.SheetTitle_3, self.item_3)
        else:
            print("ReportData Missing ,please addData")
            return

    def YoYgroFactor(self):

        if hasattr(self, "ReportData"):
            FactorYoYG = self.ReportData.fun_同比函数(self.SheetTitle, self.item)
            self.initfactor = FactorYoYG.values[self.Report_idx_Stock3d, :]
            templrb = self.ReportData.lrb[self.Report_idx_Stock3d, 4:, :]
            index = [0, 1, -3]
            self.initfactor_ReportDates = templrb[:, :, index]

            return FactorYoYG.values
        else:

            print("ReportData Missing ,please addData")
            return

    def FinDataRollingZScores(self, window=4):
        FactorRollingZScores = np.zeros(np.shape(self.initfactor)) * np.nan
        if hasattr(self, "initfactor"):
            for i in range(len(self.StockCodes)):
                FactorRollingZScores[i, :] = RollingZscores(
                    self.initfactor[i, :], window
                )
            self.initfactor = FactorRollingZScores
            return FactorRollingZScores
        else:
            print("initfactor Missing ,please addData")
            return

    def initialdata2DataFrame(self):
        factor = self.initfactor.flatten()
        StockCodes = np.repeat(
            self.StockCodes, np.shape(x1.initfactor_ReportDates[:, :, 0])[1]
        )
        TradingDates = self.initfactor_ReportDates[:, :, 0].flatten()
        ReportDates = self.initfactor_ReportDates[:, :, 1].flatten()
        ReportDates = pd.to_datetime(ReportDates, format="%Y%m%d")
        index = pd.MultiIndex.from_arrays(
            [ReportDates, StockCodes], names=["ReportDates", "StockCodes"]
        )
        columns = ["factor", "TradingDates"]
        data = np.vstack([factor, TradingDates]).T
        df = pd.DataFrame(data=data, index=index, columns=columns).sort_index(0)
        return df

    def FactorExpanding(self):
        # 不推荐使用dataframe合并的方式来扩展财务数据的原因是，财报更新时间可能不在交易日序列中
        if hasattr(self, "initfactor"):
            FactorMatrix = (
                np.ones([len(self.StockCodes), len(self.Stock3d["TradingDates"])])
                * np.nan
            )
            Timeserries = self.Stock3d["TradingDates"].astype(int)
            if type(Timeserries[0]) == np.datetime64:
                t1 = Timeserries.astype("M8[D]").astype("O")
                Timeserries = [int(date.strftime("%Y%m%d")) for date in t1]

            for i in range(len(self.StockCodes)):
                # 按照股票循环
                releasdates = self.initfactor_ReportDates[
                    i, :, 0
                ].copy()  # 财报发布日期
                pointer1 = 0  # 指针移动数据填充
                for idx, j in enumerate(Timeserries):
                    if idx == 0:
                        if releasdates[pointer1] > j:
                            FactorMatrix[i, idx] = np.nan
                        else:
                            FactorMatrix[i, idx] = self.initfactor[i, pointer1]
                            continue
                        continue
                    if idx > 0:
                        if pointer1 < len(releasdates) - 1:
                            if ~np.isnan(releasdates[pointer1 + 1]):
                                if j >= releasdates[pointer1 + 1]:
                                    pointer1 = (
                                        pointer1 + 1
                                    )  ##如果财报发部日期中间为nan，后来又有更新有效数值，则结果有错误

                        FactorMatrix[i, idx] = self.initfactor[i, pointer1]

        else:
            print("Chose a base Factor first")
            return
        return FactorMatrix

    def ExpandedFactor2DataFrame(self, ExpandedFactorMatrix):

        TradingDates = pd.to_datetime(self.TradingDates, format="%Y%m%d")
        df = pd.DataFrame(
            ExpandedFactorMatrix.T, index=TradingDates, columns=self.StockCodes
        )
        df = df.stack()
        df.index.names = ["TradingDates", "StockCodes"]
        return df


def RoE(x1, r_method="ttm", e_method="avg"):
    x1.Title("lrb", "DEDUCTEDPROFIT")
    x1.Title_2("fzb", "EQY_BELONGTO_PARCOMSH")
    x1.initfactor = x1.InitFactor()
    DEDUCTEDPROFIT = x1.apply_initial_data(r_method)
    x1.initfactor = x1.InitFactor_2()
    EQY_BELONGTO_PARCOMSH = x1.apply_initial_data(e_method)
    x1.initfactor = DEDUCTEDPROFIT / EQY_BELONGTO_PARCOMSH
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF


def RoE_zscores(x1, r_method="ttm", e_method="avg", len1=8):
    x1.Title("lrb", "DEDUCTEDPROFIT")
    x1.Title_2("fzb", "EQY_BELONGTO_PARCOMSH")
    x1.initfactor = x1.InitFactor()
    DEDUCTEDPROFIT = x1.apply_initial_data(r_method)
    x1.initfactor = x1.InitFactor_2()
    EQY_BELONGTO_PARCOMSH = x1.apply_initial_data(e_method)
    x1.initfactor = DEDUCTEDPROFIT / EQY_BELONGTO_PARCOMSH
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF


def RoE_ratio(x1, r_method="ttm", e_method="avg", len1=4):
    x1.Title("lrb", "DEDUCTEDPROFIT")
    x1.Title_2("fzb", "EQY_BELONGTO_PARCOMSH")
    x1.initfactor = x1.InitFactor()
    DEDUCTEDPROFIT = x1.apply_initial_data(r_method)
    x1.initfactor = x1.InitFactor_2()
    EQY_BELONGTO_PARCOMSH = x1.apply_initial_data(e_method)
    x1.initfactor = DEDUCTEDPROFIT / EQY_BELONGTO_PARCOMSH
    RoE_yoy_ratio = x1.apply_initial_data("ss")
    x1.initfactor = RoE_yoy_ratio
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def EP_ttm(x1, MarketCap):
    x1.Title("lrb", "DEDUCTEDPROFIT")
    x1.initfactor = x1.InitFactor()
    DEDUCTEDPROFIT = x1.apply_initial_data("ttm")
    x1.initfactor = DEDUCTEDPROFIT
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    MarketCap1 = MarketCap.groupby(level=1).shift(1)
    factorDF1 = factorDF / MarketCap1
    return factorDF1


def EP_ss(x1, MarketCap):
    x1.Title("lrb", "DEDUCTEDPROFIT")
    x1.initfactor = x1.InitFactor()
    DEDUCTEDPROFIT = x1.apply_initial_data("ss")
    x1.initfactor = DEDUCTEDPROFIT
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    MarketCap1 = MarketCap.groupby(level=1).shift(1)
    factorDF1 = factorDF / MarketCap1
    return factorDF1


def BP(x1, MarketCap):
    x1.Title("fzb", "EQY_BELONGTO_PARCOMSH")
    x1.initfactor = x1.InitFactor()
    EQY_BELONGTO_PARCOMSH = x1.apply_initial_data("avg")
    x1.initfactor = EQY_BELONGTO_PARCOMSH
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    MarketCap1 = MarketCap.groupby(level=1).shift(1)
    factorDF1 = factorDF / MarketCap1
    return factorDF1


def SP_ttm(x1, MarketCap):
    x1.Title("xjlb", "CASH_RECP_SG_AND_RS")
    x1.initfactor = x1.InitFactor()
    DEDUCTEDPROFIT = x1.apply_initial_data("ttm")
    x1.initfactor = DEDUCTEDPROFIT
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    MarketCap1 = MarketCap.groupby(level=1).shift(1)
    factorDF1 = factorDF / MarketCap1
    return factorDF1


def SP_ss(x1, MarketCap):
    x1.Title("xjlb", "CASH_RECP_SG_AND_RS")
    x1.initfactor = x1.InitFactor()
    DEDUCTEDPROFIT = x1.apply_initial_data("ss")
    x1.initfactor = DEDUCTEDPROFIT
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    MarketCap1 = MarketCap.groupby(level=1).shift(1)
    factorDF1 = factorDF / MarketCap1
    return factorDF1


def SUE_ss(x1, len1=4):
    x1.Title("lrb", "DEDUCTEDPROFIT")
    x1.initfactor = x1.InitFactor()
    x1.initfactor = x1.apply_initial_data("ss")
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF


def SUE_ttm(x1, len1=4):
    x1.Title("lrb", "DEDUCTEDPROFIT")
    x1.initfactor = x1.InitFactor()
    x1.initfactor = x1.apply_initial_data("ttm")
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF


def SUE_yoy(x1, len1=4):
    x1.Title("lrb", "DEDUCTEDPROFIT")
    x1.initfactor = x1.InitFactor()
    x1.initfactor = x1.apply_initial_data("yoy")
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF


def SUE_qoq(x1, len1=4):
    x1.Title("lrb", "DEDUCTEDPROFIT")
    x1.initfactor = x1.InitFactor()
    x1.initfactor = x1.apply_initial_data("qoq")
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF


def DEDUCTEDPROFIT_qoq(x1, len1=4):
    x1.Title("lrb", "DEDUCTEDPROFIT")
    x1.initfactor = x1.InitFactor()
    x1.initfactor = x1.apply_initial_data("qoq")
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def DEDUCTEDPROFIT_yoy(x1, len1=4):
    x1.Title("lrb", "DEDUCTEDPROFIT")
    x1.initfactor = x1.InitFactor()
    x1.initfactor = x1.apply_initial_data("yoy")
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def OPER_REV_yoy(x1, len1=4):
    # 营业收入同比增长率
    x1.Title("lrb", "OPER_REV")
    x1.initfactor = x1.InitFactor()
    x1.initfactor = x1.apply_initial_data("yoy")
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def STOT_CASH_INFLOWS_OPER_ACT_yoy(x1, len1=4):
    # 经营活动现金流入小计同比增长率
    x1.Title("xjlb", "STOT_CASH_INFLOWS_OPER_ACT")
    x1.initfactor = x1.InitFactor()
    x1.initfactor = x1.apply_initial_data("yoy")
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def STOT_CASH_INFLOWS_OPER_ACT_divedeby_TOT_CUR_LIAB(x1, len1=4):
    # 经营活动现金流入小计/流动负债
    x1.Title("xjlb", "STOT_CASH_INFLOWS_OPER_ACT")
    STOT_CASH_INFLOWS_OPER_ACT = x1.InitFactor()
    x1.Title("fzb", "TOT_CUR_LIAB")
    TOT_CUR_LIAB = x1.InitFactor()
    x1.initfactor = STOT_CASH_INFLOWS_OPER_ACT / TOT_CUR_LIAB
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def FreeCashflow(x1, len1=4):
    x1.Title("xjlb", "NET_CASH_FLOWS_OPER_ACT")  # 经营活动产生的现金流量净额
    NET_CASH_FLOWS_OPER_ACT = x1.InitFactor()
    x1.Title_2(
        "xjlb", "CASH_PAY_ACQ_CONST_FIOLTA"
    )  # 构建固定资产、无形资产和其他长期资产支付的现金
    CASH_PAY_ACQ_CONST_FIOLTA = x1.InitFactor_2()
    x1.initfactor = NET_CASH_FLOWS_OPER_ACT - CASH_PAY_ACQ_CONST_FIOLTA
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF


def FreeCashflow_ratio(x1, len1=4):
    x1.Title("xjlb", "NET_CASH_FLOWS_OPER_ACT")  # 经营活动产生的现金流量净额
    NET_CASH_FLOWS_OPER_ACT = x1.InitFactor()
    x1.Title_2(
        "xjlb", "CASH_PAY_ACQ_CONST_FIOLTA"
    )  # 构建固定资产、无形资产和其他长期资产支付的现金
    CASH_PAY_ACQ_CONST_FIOLTA = x1.InitFactor_2()
    x1.initfactor = NET_CASH_FLOWS_OPER_ACT - CASH_PAY_ACQ_CONST_FIOLTA
    x1.initfactor = x1.apply_initial_data("yoy")
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def NET_CASH_FLOWS_OPER_ACT_yoy(x1, len1=4):
    # 经营活动产生的现金流量净额同比增长率
    x1.Title("xjlb", "NET_CASH_FLOWS_OPER_ACT")
    x1.initfactor = x1.InitFactor()
    x1.initfactor = x1.apply_initial_data("yoy")
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def CurrentRatio(x1, len1=8):
    # 流动比率
    x1.Title("fzb", "TOT_CUR_ASSETS")
    x1.initfactor = x1.InitFactor()
    tca = x1.apply_initial_data("avg")  # 流动资产
    x1.Title("fzb", "TOT_CUR_LIAB")
    x1.initfactor = x1.InitFactor()
    tcl = x1.apply_initial_data("avg")  # 流动负债
    x1.initfactor = tca / tcl
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def QuickRatio(x1, len1=8):
    # 速动比率
    x1.Title("fzb", "TOT_CUR_ASSETS")
    x1.initfactor = x1.InitFactor()
    tca = x1.apply_initial_data("avg")  # 流动资产

    x1.Title("fzb", "INVENTORIES")
    x1.initfactor = x1.InitFactor()
    INVENTORIES = x1.apply_initial_data("avg")  # 存货

    x1.Title("fzb", "PREPAY")
    x1.initfactor = x1.InitFactor()
    PREPAY = x1.apply_initial_data("avg")  # 预付款项

    x1.Title("fzb", "TOT_CUR_LIAB")
    x1.initfactor = x1.InitFactor()
    tcl = x1.apply_initial_data("avg")  # 流动负债
    x1.initfactor = (tca - INVENTORIES - PREPAY) / tcl
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def GOODWILLRatio(x1, len1=8):
    x1.Title("fzb", "GOODWILL")
    x1.initfactor = x1.InitFactor()
    GOODWILL = x1.apply_initial_data("avg")  # 商誉
    x1.Title("fzb", "TOT_ASSETS")
    x1.initfactor = x1.InitFactor()
    TOT_ASSETS = x1.apply_initial_data("avg")  # 总资产
    x1.initfactor = GOODWILL / TOT_ASSETS
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def ACCT_RCV__NET_CASH_FLOWS_OPER_ACT(x1, len1=8):
    x1.Title("fzb", "ACCT_RCV")
    x1.initfactor = x1.InitFactor()
    ACCT_RCV = x1.apply_initial_data("ttm")  # 应收账款
    x1.Title("xjlb", "NET_CASH_FLOWS_OPER_ACT")
    x1.initfactor = x1.InitFactor()
    NET_CASH_FLOWS_OPER_ACT = x1.apply_initial_data("ttm")  # 经营活动产生的现金流量净额
    x1.initfactor = ACCT_RCV / NET_CASH_FLOWS_OPER_ACT
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def wet_profit_ratio(x1, len1=8):
    x1.Title("xjlb", "STOT_CASH_INFLOWS_OPER_ACT")
    x1.initfactor = x1.InitFactor()
    CASH_RECP_SG_AND_RS = x1.apply_initial_data("ttm")  # 经营活动现金流入小计
    x1.Title("xjlb", "STOT_CASH_OUTFLOWS_OPER_ACT")
    x1.initfactor = x1.InitFactor()
    NET_INCR_LENDING_FUND = x1.apply_initial_data("ttm")  # 经营活动现金流出小计
    x1.initfactor = CASH_RECP_SG_AND_RS / NET_INCR_LENDING_FUND
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def net_profit_FIN_EXP_CS_ratio(x1, len1=8):
    x1.Title("xjlb", "FIN_EXP_CS")
    x1.initfactor = x1.InitFactor()
    FIN_EXP_CS = x1.apply_initial_data("ttm")  # 财务费用,但是只有年报和半年报
    x1.Title("xjlb", "NET_PROFIT_CS")
    x1.initfactor = x1.InitFactor()
    NETPROFIT = x1.apply_initial_data("ttm")  # 净利润,但是只有年报和半年报
    x1.initfactor = FIN_EXP_CS / NETPROFIT
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def none_linear_marketcap(MarketCap):  # 非线性市值因子
    logmc = np.log(MarketCap)
    logmc3 = logmc**3
    data = pd.DataFrame({"logmc": logmc, "logmc3": logmc3})
    logmcustack = data["logmc"].unstack()
    logmc3ustack = data["logmc3"].unstack()
    TradingDates = logmcustack.index
    StockCodes = logmcustack.columns
    none_linear_np = np.ones((len(TradingDates), len(StockCodes))) * np.nan
    np1 = logmcustack.values
    np2 = logmc3ustack.values
    for i in range(len(TradingDates)):
        temp1 = np1[i, :]
        temp2 = np2[i, :]
        nanmask = np.isnan(temp1)
        x = temp1[~nanmask]
        y = temp2[~nanmask]
        if len(x) == 0:
            continue
        model = sm.OLS(y, sm.add_constant(x), missing="drop")
        results = model.fit()
        none_linear_np[i, ~nanmask] = results.resid
    none_linear_marketcap_pd = pd.DataFrame(
        none_linear_np, index=TradingDates, columns=StockCodes
    )
    none_linear_marketcap_pd = none_linear_marketcap_pd.shift(1).stack()
    return none_linear_marketcap_pd


def log3marketcap(MarketCap):
    logmc = np.log(MarketCap)
    logmc3 = logmc**3
    return logmc3


def half_decay_factor(factor, releaseddata, para=20, show=False):
    if isinstance(factor, pd.DataFrame):
        factor.columns = ["factor"]
    else:
        factor = factor.to_frame(name="factor")
    logbase = np.exp(-np.log(2) / para)
    minvalues = releaseddata.min(axis=1).to_frame(name="releasedmin")
    minvalues[minvalues < 0] = np.nan
    factor1 = factor.join(minvalues, how="left")
    factor1["indice"] = logbase ** factor1["releasedmin"]
    df = factor1["factor"] * factor1["indice"]
    return df.to_frame(name="factor")


def IndexComponentWeight(IndexComponent, FreeMarketCap):
    weightdata = FreeMarketCap.loc[IndexComponent.index]
    weightdata1 = (
        weightdata.groupby(level=0, group_keys=False)
        .apply(lambda x: x / x.sum())
        .to_frame(name="weight")
    )

    return weightdata1


def BetaFactorbyIndex(PriceDf, indexname, window=[20, 60, 120, 240]):
    benchmark = SDP.IndexPoint(indexname)
    benchmark["tradingday"] = pd.to_datetime(benchmark["tradingday"], format="%Y%m%d")
    benchmark = benchmark.rename(columns={"tradingday": "TradingDates"})
    benchmark.set_index("TradingDates", inplace=True)
    benchmarkreturn = np.log(benchmark["c"] / benchmark["c"].shift(1)).to_frame(
        name="benchmarkreturn"
    )
    stockreturn = (
        PriceDf.groupby(level=1, group_keys=False)
        .apply(
            lambda x: np.log(
                (x["c"] * x["adjfactor"]) / (x["c"].shift(1) * x["adjfactor"].shift(1))
            )
        )
        .sort_index(level=0)
    )
    stockreturn = stockreturn.unstack()
    stockreturn = stockreturn.fillna(0)
    data = pd.concat([stockreturn, benchmarkreturn], axis=1)
    data = data.loc[benchmarkreturn.index]
    data = data.loc[stockreturn.index]
    nan_rows = stockreturn[stockreturn.isna().all(axis=1)]
    nan_index = nan_rows.index
    data = data.drop(nan_index)
    columns_without_benchmarkreturn = [
        col for col in data.columns if col != "benchmarkreturn"
    ]
    betas0 = np.zeros((len(data.index), len(columns_without_benchmarkreturn)))
    betas1 = np.zeros((len(data.index), len(columns_without_benchmarkreturn)))
    betas2 = np.zeros((len(data.index), len(columns_without_benchmarkreturn)))
    betas3 = np.zeros((len(data.index), len(columns_without_benchmarkreturn)))

    resid0 = np.zeros((len(data.index), len(columns_without_benchmarkreturn)))
    resid1 = np.zeros((len(data.index), len(columns_without_benchmarkreturn)))
    resid2 = np.zeros((len(data.index), len(columns_without_benchmarkreturn)))
    resid3 = np.zeros((len(data.index), len(columns_without_benchmarkreturn)))

    datanp = data.to_numpy()

    for i in range(len(data.index)):
        print(data.index[i])
        if data.index[i] <= pd.Timestamp("2018-01-01"):
            continue
        if i >= window[0]:
            Y0 = datanp[i - window[0] : i, -1]  # 左开右闭不用移动
            for j, col in enumerate(columns_without_benchmarkreturn):
                X = datanp[i - window[0] : i, j]
                X = sm.add_constant(X)
                model = sm.OLS(Y0, X, missing="drop")
                results = model.fit()
                betas0[i, j] = results.params[1]
                resid0[i, j] = results.resid[-1]
        if i >= window[1]:
            Y1 = datanp[i - window[1] : i, -1]
            for j, col in enumerate(columns_without_benchmarkreturn):
                X = datanp[i - window[1] : i, j]
                X = sm.add_constant(X)
                model = sm.OLS(Y1, X, missing="drop")
                results = model.fit()
                betas1[i, j] = results.params[1]
                resid1[i, j] = results.resid[-1]
        if i >= window[2]:
            Y2 = datanp[i - window[2] : i, -1]
            for j, col in enumerate(columns_without_benchmarkreturn):
                X = datanp[i - window[2] : i, j]
                X = sm.add_constant(X)
                model = sm.OLS(Y2, X, missing="drop")
                results = model.fit()
                betas2[i, j] = results.params[1]
                resid2[i, j] = results.resid[-1]
        if i >= window[3]:
            Y3 = datanp[i - window[3] : i, -1]
            for j, col in enumerate(columns_without_benchmarkreturn):
                X = datanp[i - window[3] : i, j]
                X = sm.add_constant(X)
                model = sm.OLS(Y3, X, missing="drop")
                results = model.fit()
                betas3[i, j] = results.params[1]
                resid3[i, j] = results.resid[-1]

    BetaFactor0 = (
        pd.DataFrame(betas0, index=data.index, columns=columns_without_benchmarkreturn)
        .stack()
        .to_frame(name="Beta_" + str(window[0]) + "days")
    )
    BetaFactor0.index.names = ["TradingDates", "StockCodes"]
    BetaFactor1 = (
        pd.DataFrame(betas1, index=data.index, columns=columns_without_benchmarkreturn)
        .stack()
        .to_frame(name="Beta_" + str(window[1]) + "days")
    )
    BetaFactor1.index.names = ["TradingDates", "StockCodes"]
    BetaFactor2 = (
        pd.DataFrame(betas2, index=data.index, columns=columns_without_benchmarkreturn)
        .stack()
        .to_frame(name="Beta_" + str(window[2]) + "days")
    )
    BetaFactor2.index.names = ["TradingDates", "StockCodes"]
    BetaFactor3 = (
        pd.DataFrame(betas3, index=data.index, columns=columns_without_benchmarkreturn)
        .stack()
        .to_frame(name="Beta_" + str(window[3]) + "days")
    )
    BetaFactor3.index.names = ["TradingDates", "StockCodes"]

    ResidFactor0 = (
        pd.DataFrame(resid0, index=data.index, columns=columns_without_benchmarkreturn)
        .stack()
        .to_frame(name="Resid_" + str(window[0]) + "days")
    )
    ResidFactor0.index.names = ["TradingDates", "StockCodes"]
    ResidFactor1 = (
        pd.DataFrame(resid1, index=data.index, columns=columns_without_benchmarkreturn)
        .stack()
        .to_frame(name="Resid_" + str(window[1]) + "days")
    )
    ResidFactor1.index.names = ["TradingDates", "StockCodes"]
    ResidFactor2 = (
        pd.DataFrame(resid2, index=data.index, columns=columns_without_benchmarkreturn)
        .stack()
        .to_frame(name="Resid_" + str(window[2]) + "days")
    )
    ResidFactor2.index.names = ["TradingDates", "StockCodes"]
    ResidFactor3 = (
        pd.DataFrame(resid3, index=data.index, columns=columns_without_benchmarkreturn)
        .stack()
        .to_frame(name="Resid_" + str(window[3]) + "days")
    )
    ResidFactor3.index.names = ["TradingDates", "StockCodes"]
    return (
        BetaFactor0,
        BetaFactor1,
        BetaFactor2,
        BetaFactor3,
        ResidFactor0,
        ResidFactor1,
        ResidFactor2,
        ResidFactor3,
    )  # 20,60,120,240


def BetafactorbyIndexWLS(PriceDf, indexname, halfdecay=60, window=252):
    assert halfdecay <= 0.8 * window, "halfdecay should be larger than 0.5*window"
    t = np.arange(window)[::-1]
    # 生成权重序列
    weights = (0.5) ** (t / halfdecay)
    weights = weights / np.sum(weights)
    benchmark = SDP.IndexPoint(indexname)
    benchmark["tradingday"] = pd.to_datetime(benchmark["tradingday"], format="%Y%m%d")
    benchmark = benchmark.rename(columns={"tradingday": "TradingDates"})
    benchmark.set_index("TradingDates", inplace=True)
    benchmarkreturn = np.log(benchmark["c"] / benchmark["c"].shift(1)).to_frame(
        name="benchmarkreturn"
    )
    begindate = pd.Timestamp("2015-01-01")
    stockreturn = (
        PriceDf.groupby(level=1, group_keys=False)
        .apply(
            lambda x: np.log(
                (x["c"] * x["adjfactor"]) / (x["c"].shift(1) * x["adjfactor"].shift(1))
            )
        )
        .sort_index(level=0)
    ).to_frame(name="stockreturn")

    stockreturn = stockreturn.join(PriceDf["v"]).unstack()
    stockreturn = stockreturn.loc[begindate:]
    benchmarkreturn = benchmarkreturn.loc[begindate:]
    r = stockreturn["stockreturn"]
    v = stockreturn["v"]
    dateindex = r.index
    bench_np = benchmarkreturn.loc[dateindex].values.flatten()
    stockcodes = r.columns
    r_np = r.to_numpy()
    v_np = v.to_numpy()

    temps = np.nan * np.ones((window, len(stockcodes)))
    idx0 = np.zeros(len(stockcodes)).astype(int)
    tempinputlag = np.zeros(len(stockcodes)).astype(int)
    tempinputlagmatrix = -1 * np.ones((window, len(stockcodes))).astype(int)
    linspace1 = np.linspace(0, window - 1, window).astype(int)[::-1]

    alfa_matrix = np.ones((len(dateindex), len(stockcodes))) * np.nan  # 存储alpha
    beta_matrix = np.ones((len(dateindex), len(stockcodes))) * np.nan  # 存储beta
    resid_matrix = np.ones((len(dateindex), len(stockcodes))) * np.nan  # 存储残差
    resid_std_matrix = (
        np.ones((len(dateindex), len(stockcodes))) * np.nan
    )  # 存储残差标准差
    for i in range(len(dateindex)):
        linspace2 = i - linspace1
        print(dateindex[i])
        for j in range(len(stockcodes)):

            if v_np[i, j] > 0:
                tempinputlag[j] = 0  # 如果本日有交易量 ，则将tempinputlag置为0
                if idx0[j] < window:  # 填充位置小于窗口长度，说明动态窗口数据未填充满
                    temps[idx0[j], j] = r_np[i, j]
                    tempinputlagmatrix[idx0[j], j] = tempinputlag[j]
                    idx0[j] += 1
                else:  # 动态窗口数据已经填充满，将数据往前移动一位，最后一位填充新数据
                    temps[:-1, j] = temps[1:, j]
                    temps[-1, j] = r_np[i, j]
                    tempinputlagmatrix[:-1, j] = tempinputlagmatrix[1:, j]
                    tempinputlagmatrix[-1, j] = tempinputlag[j]
            else:  # 如果本日无交易量
                tempinputlag[j] = (
                    tempinputlag[j] + 1
                )  # 当前lag记录+1， 说明当前股票已经连续多少天没有交易
                if idx0[j] < window:
                    tempinputlagmatrix[idx0[j], j] = tempinputlag[j]
                else:
                    tempinputlagmatrix[:-1, j] = tempinputlagmatrix[1:, j]
                    tempinputlagmatrix[-1, j] = tempinputlag[j]
            if ~np.any(np.isnan(temps[:, j])):  # 如果动态窗口数据已经填充满
                benchmarkidx = linspace2 - tempinputlagmatrix[:, j][::-1].cumsum()[::-1]
                if benchmarkidx[0] >= 0:
                    tempbench = bench_np[benchmarkidx]
                    X = sm.add_constant(temps[:, j])
                    model = sm.WLS(tempbench, X, weights=weights)
                    results = model.fit()
                    alfa_matrix[i, j] = results.params[0]
                    beta_matrix[i, j] = results.params[1]
                    resid_matrix[i, j] = results.resid[-1]
                    resid_std_matrix[i, j] = results.resid.std()

    Alfa_Df = (
        pd.DataFrame(alfa_matrix, index=dateindex, columns=stockcodes)
        .shift(1)
        .stack()
        .to_frame(name="Alfa")
    )
    Alfa_Df.index.names = ["TradingDates", "StockCodes"]
    Beta_Df = (
        pd.DataFrame(beta_matrix, index=dateindex, columns=stockcodes)
        .shift(1)
        .stack()
        .to_frame(name="Beta")
    )
    Beta_Df.index.names = ["TradingDates", "StockCodes"]
    Resid_Df = (
        pd.DataFrame(resid_matrix, index=dateindex, columns=stockcodes)
        .shift(1)
        .stack()
        .to_frame(name="Resid")
    )
    Resid_Df.index.names = ["TradingDates", "StockCodes"]
    ResidStd_Df = (
        pd.DataFrame(resid_std_matrix, index=dateindex, columns=stockcodes)
        .shift(1)
        .stack()
        .to_frame(name="ResidStd")
    )
    ResidStd_Df.index.names = ["TradingDates", "StockCodes"]
    return Alfa_Df, Beta_Df, Resid_Df, ResidStd_Df


def run_Index_regression(
            PriceDf,
            datasavepath=None,
        ) -> None:
    if datasavepath is None:
        datasavepath = r"E:\Documents\PythonProject\StockProject\StockData\RawFactors"
    (
        zz2000_252_60_alfa,
        zz2000_252_60_beta,
        zz2000_252_60_resid,
        zz2000_252_60_residstd,
    ) = BetafactorbyIndexWLS(PriceDf, "中证2000", halfdecay=60, window=252)
    pd.to_pickle(zz2000_252_60_alfa, datasavepath + r"\zz2000_252_60_alfa.pkl")
    pd.to_pickle(zz2000_252_60_beta, datasavepath + r"\zz2000_252_60_beta.pkl")
    pd.to_pickle(zz2000_252_60_resid, datasavepath + r"\zz2000_252_60_resid.pkl")
    pd.to_pickle(zz2000_252_60_residstd, datasavepath + r"\zz2000_252_60_residstd.pkl")
    (
        zz2000_126_30_alfa,
        zz2000_126_30_beta,
        zz2000_126_30_resid,
        zz2000_126_30_residstd,
    ) = BetafactorbyIndexWLS(PriceDf, "中证2000", halfdecay=30, window=126)
    pd.to_pickle(zz2000_252_60_alfa, datasavepath + r"\zz2000_126_30_alfa.pkl")
    pd.to_pickle(zz2000_252_60_beta, datasavepath + r"\zz2000_126_30_beta.pkl")
    pd.to_pickle(zz2000_252_60_resid, datasavepath + r"\zz2000_126_30_resid.pkl")
    pd.to_pickle(zz2000_252_60_residstd, datasavepath + r"\zz2000_126_30_residstd.pkl")
    return


def run_finacialfactors(x1, datasavepath, datapath=None) -> None:
    if datapath is None:
        datapath = r"E:\Documents\PythonProject\StockProject\StockData"
    MarketCap = pd.read_pickle(datasavepath + "\\" + "MarketCap.pkl")
    none_linear_marketcap_data = none_linear_marketcap(MarketCap)
    pd.to_pickle(
        none_linear_marketcap_data, datasavepath + r"\none_linear_marketcap.pkl"
    )  # 非线性市值因子
    log3marketcap_data = log3marketcap(MarketCap)
    pd.to_pickle(
        log3marketcap_data, datasavepath + r"\log3marketcap.pkl"
    )  # log3市值因子

    factorDF = EP_ttm(x1, MarketCap)
    pd.to_pickle(factorDF, datasavepath + r"\EP_ttm.pkl")
    factorDF = EP_ss(x1, MarketCap)
    pd.to_pickle(factorDF, datasavepath + r"\EP_ss.pkl")
    factorDF = BP(x1, MarketCap)
    pd.to_pickle(factorDF, datasavepath + r"\BP.pkl")
    factorDF = SP_ttm(x1, MarketCap)
    pd.to_pickle(factorDF, datasavepath + r"\SP_ttm.pkl")
    factorDF = SP_ss(x1, MarketCap)
    pd.to_pickle(factorDF, datasavepath + r"\SP_ss.pkl")
    factorDF, factorDF_zscores = DEDUCTEDPROFIT_yoy(x1)
    pd.to_pickle(factorDF, datasavepath + r"\DEDUCTEDPROFIT_yoy.pkl")
    pd.to_pickle(factorDF_zscores, datasavepath + r"\DEDUCTEDPROFIT_yoy_zscores_4.pkl")
    factorDF = RoE(x1)
    pd.to_pickle(factorDF, datasavepath + r"\ROE.pkl")
    factorDF = RoE_zscores(x1, r_method="ttm", e_method="avg", len1=8)
    pd.to_pickle(factorDF, datasavepath + r"\ROEzscores_8.pkl")
    factorDF = RoE_zscores(x1, r_method="ttm", e_method="avg", len1=4)
    pd.to_pickle(factorDF, datasavepath + r"\ROEzscores_4.pkl")
    factorDF, factorDF_zscores = RoE_ratio(x1)
    pd.to_pickle(factorDF, datasavepath + r"\ROE_ratio.pkl")
    pd.to_pickle(factorDF_zscores, datasavepath + r"\ROE_ratio_zscores_4.pkl")
    factorDF, factorDF_zscores = NET_CASH_FLOWS_OPER_ACT_yoy(
        x1, 4
    )  # 经营活动产生的现金流量净额同比增长率
    pd.to_pickle(factorDF, datasavepath + r"\NET_CASH_FLOWS_OPER_ACT_yoy_4.pkl")
    pd.to_pickle(
        factorDF_zscores, datasavepath + r"\NET_CASH_FLOWS_OPER_ACT_yoy_zscores_4.pkl"
    )
    factorDF, factorDF_zscores = OPER_REV_yoy(x1, 4)  # 营业收入同比增长率
    pd.to_pickle(factorDF, datasavepath + r"\OPER_REV_yoy_4.pkl")
    pd.to_pickle(factorDF_zscores, datasavepath + r"\OPER_REV_yoy_zscores_4.pkl")
    factorDF, factorDF_zscores = STOT_CASH_INFLOWS_OPER_ACT_yoy(
        x1, 4
    )  # 经营活动现金流入小计同比增长率
    pd.to_pickle(factorDF, datasavepath + r"\STOT_CASH_INFLOWS_OPER_ACT_yoy_4.pkl")
    pd.to_pickle(
        factorDF_zscores,
        datasavepath + r"\STOT_CASH_INFLOWS_OPER_ACT_yoy_zscores_4.pkl",
    )
    factorDF, factorDF_zscores = FreeCashflow_ratio(x1, 4)  # 自由现金流入小计同比增长率
    pd.to_pickle(factorDF, datasavepath + r"\FreeCashflow_ratio.pkl")
    pd.to_pickle(factorDF_zscores, datasavepath + r"\FreeCashflow_ratio_zscores_4.pkl")

    factorDF = SUE_ss(x1, 4)
    pd.to_pickle(factorDF, datasavepath + r"\SUE_ss_4.pkl")
    factorDF = SUE_ttm(x1, 4)
    pd.to_pickle(factorDF, datasavepath + r"\SUE_ttm_4.pkl")
    factorDF = SUE_yoy(x1, 4)
    pd.to_pickle(factorDF, datasavepath + r"\SUE_yoy_4.pkl")
    factorDF = SUE_qoq(x1, 4)
    pd.to_pickle(factorDF, datasavepath + r"\SUE_qoq_4.pkl")
    factorDF, factorDF_zscores = CurrentRatio(x1, 8)
    pd.to_pickle(factorDF, datasavepath + r"\CurrentRatio.pkl")
    pd.to_pickle(factorDF_zscores, datasavepath + r"\CurrentRatio_zscores_8.pkl")
    factorDF, factorDF_zscores = QuickRatio(x1, 8)
    pd.to_pickle(factorDF, datasavepath + r"\QuickRatio.pkl")
    pd.to_pickle(factorDF_zscores, datasavepath + r"\QuickRatio_zscores_8.pkl")
    factorDF, factorDF_zscores = GOODWILLRatio(x1, 8)
    pd.to_pickle(factorDF, datasavepath + r"\GOODWILLRatio.pkl")
    pd.to_pickle(factorDF_zscores, datasavepath + r"\GOODWILLRatio_zscores_8.pkl")
    factorDF, factorDF_zscores = ACCT_RCV__NET_CASH_FLOWS_OPER_ACT(x1, 8)
    pd.to_pickle(factorDF, datasavepath + r"\ACCT_RCV__NET_CASH_FLOWS_OPER_ACT.pkl")
    pd.to_pickle(
        factorDF_zscores,
        datasavepath + r"\ACCT_RCV__NET_CASH_FLOWS_OPER_ACT_zscores_8.pkl",
    )
    factorDF, factorDF_zscores = wet_profit_ratio(x1, 8)
    pd.to_pickle(factorDF, datasavepath + r"\wet_profit_ratio.pkl")
    pd.to_pickle(factorDF_zscores, datasavepath + r"\wet_profit_ratio_zscores_8.pkl")
    realead_dates_count_df = pd.read_pickle(
        datapath + "\\" + "realesed_dates_count_df.pkl"
    )
    SUE_ss_4 = pd.read_pickle(datasavepath + r"\SUE_ss_4.pkl")
    SUE_ss_4_hd5 = half_decay_factor(
        SUE_ss_4, realead_dates_count_df, para=5, show=False
    )
    pd.to_pickle(SUE_ss_4_hd5, datasavepath + r"\SUE_ss_4_hd5.pkl")
    testdata1 = pd.read_pickle(datasavepath + r"\ROE.pkl")
    testdata2 = pd.read_pickle(datasavepath + r"\CurrentRatio.pkl")
    ROE_d_Currentratio = testdata1 / testdata2
    pd.to_pickle(ROE_d_Currentratio, datasavepath + r"\ROE_d_Currentratio.pkl")

    gw = pd.read_pickle(datasavepath + r"\GOODWILLRatio.pkl")
    qr = pd.read_pickle(datasavepath + r"\QuickRatio.pkl")
    gw_divide_qr = gw / qr
    pd.to_pickle(gw_divide_qr, datasavepath + r"\GOODWILLRatio_divide_QuickRatio.pkl")

    ep = pd.read_pickle(datasavepath + r"\EP_ttm.pkl")
    sp = pd.read_pickle(datasavepath + r"\SP_ttm.pkl")
    g = pd.read_pickle(datasavepath + r"\DEDUCTEDPROFIT_yoy.pkl")
    PEG = ep * g
    pd.to_pickle(PEG, datasavepath + r"\PEG.pkl")
    spg = sp * (g + 1)
    pd.to_pickle(spg, datasavepath + r"\spg.pkl")
    roe = pd.read_pickle(datasavepath + r"\ROEzscores_4.pkl")
    roe_d_g = roe * g
    pd.to_pickle(roe_d_g, datasavepath + r"\ROEzscores_4_t_profitgrowth.pkl")
    bp = pd.read_pickle(datasavepath + r"\BP.pkl")
    df = bp * (g + 1)
    pd.to_pickle(df, datasavepath + r"\BP_t_profitgrowth.pkl")

    none_linear_marketcap_data = pd.read_pickle(
        datasavepath + r"\none_linear_marketcap.pkl"
    )
    df = roe_d_g * 100 / none_linear_marketcap_data
    pd.to_pickle(
        df,
        datasavepath + r"\ROEzscores_4_t_profitgrowth_divide_none_linear_marketcap.pkl",
    )

    return


def run(basicdatapath=None, rawdatasavepath=None):
    if basicdatapath is None:
        basicdatapath = r"E:\Documents\PythonProject\StockProject\StockData"
    if rawdatasavepath is None:
        rawdatasavepath = (
            r"E:\Documents\PythonProject\StockProject\StockData\RawFactors"
        )
    cla = FinData.财报数据()
    begindate = 20100101
    today = datetime.today()
    int_today = int(today.strftime("%Y%m%d"))
    cla.读取财报数据(begindate, int_today)
    filename = os.path.join(basicdatapath, "Stock3d.pkl")
    Stock3d = pd.read_pickle(filename)
    x1 = FactorMatrix_Report(Stock3d, cla)
    run_finacialfactors(x1, rawdatasavepath)
  
    pricefile = os.path.join(basicdatapath, "Price.pkl")
    PriceDF = pd.read_pickle(pricefile)
    tradablefile = os.path.join(basicdatapath, "TradableDF.pkl")
    StockTradableDF = pd.read_pickle(tradablefile)
    PriceDF = PriceDF[~(PriceDF["exchange_id"] == "BJ")]  # 踢出北交所
    StockTradableDF = StockTradableDF[
        ~(StockTradableDF["exchange_id"] == "BJ")
    ]  # 踢出北交所
    PriceDF = PriceDF.join(
        StockTradableDF, how="left", lsuffix="_left", rsuffix="_right"
    )
    PriceDF = PriceDF[~(PriceDF["trade_status"] == "退市")]
    run_Index_regression(PriceDF)
    return


if __name__ == "__main__":
    print("main")
    run(basicdatapath=None, rawdatasavepath=None)
