""" # -*- coding: utf-8 -*-
从sql或者LAN中获取数据下载到本地
@author: ZhangXi
"""

import os
import numpy as np
import pandas as pd
import pymssql
from datetime import date
from datetime import datetime
import time
import h5py
import sys
int_today = int(date.today().strftime("%Y%m%d"))
str_today = str(date.today().strftime("%Y%m%d"))
sys.path.append(r"E:\Documents\PythonProject\StockProject")
import lgc_板块成份api as ComData


def GetAllDayPriceDataFromSql_save(datasavepath=None):
    # 读取所有股票日线数据， 并存储， 然后返回沪深交易所的股票数据
    if datasavepath is None:
        datasavepath = r"E:\Documents\PythonProject\StockProject\StockData"

    def _PriceDf_storage(datasavepath):

        StockDataDF = GetStockDayDataDFFromSql(20140101)
        StockDataDF["tradingday"] = pd.to_datetime(
            StockDataDF["tradingday"], format="%Y%m%d"
        )
        PriceDf = StockDataDF.set_index(["tradingday", "code"])
        PriceDf.index.set_names(["TradingDates", "StockCodes"], inplace=True)
        PriceDf = PriceDf.sort_index()
        StockStopPrice = GetStockStopPrice()
        StockStopPrice["tradingday"] = pd.to_datetime(
            StockStopPrice["tradingday"], format="%Y%m%d"
        )
        StockStopPrice = StockStopPrice.set_index(["tradingday", "code"])
        StockStopPrice.index.set_names(["TradingDates", "StockCodes"], inplace=True)
        StockStopPrice = StockStopPrice.sort_index()
        PriceDf = PriceDf.join(StockStopPrice, how="left")

        PriceDf["MC"] = PriceDf["total_shares"] * PriceDf["c"]
        PriceDf["FMC"] = PriceDf["free_float_shares"] * PriceDf["c"]
        PriceDf["turnoverrate"] = PriceDf["v"] / PriceDf["total_shares"]
        PriceDf["vwap"] = PriceDf["amt"] / PriceDf["v"]
        PriceDf["freeturnoverrate"] = PriceDf["v"] / PriceDf["free_float_shares"]
        PriceDf.to_pickle(
            datasavepath + r"\Price.pkl"
        )  # 存储原始数据+涨跌停价格+vwap复权数据+换手率+市值的dataframe
        return PriceDf

    PriceDf = _PriceDf_storage(datasavepath)

    def _tradable_storage(datasavepath):
        StockTradableDF = GetTradableStocksFromSql()
        StockTradableDF["tradingday"] = pd.to_datetime(
            StockTradableDF["tradingday"], format="%Y%m%d"
        )

        StockTradableDF = StockTradableDF.set_index(["tradingday", "code"])
        StockTradableDF.index.set_names(["TradingDates", "StockCodes"], inplace=True)
        StockTradableDF = StockTradableDF.sort_index()
        pd.to_pickle(StockTradableDF, datasavepath + r"\TradableDF.pkl")
        return StockTradableDF

    StockTradableDF = _tradable_storage(datasavepath)

    return PriceDf, StockTradableDF


def GetStockStopPrice():
    """从数据库中获取所有股票当天涨停价"""
    db = pymssql.connect(
        host="198.16.102.88", user="sa", password="Sy123456", database="stock_data"
    )
    cursor = db.cursor()
    sql_str = "SELECT distinct [code],[tradingday],[high_limit],[low_limit] FROM [stock_data].[dbo].[lgc_涨跌停板]"
    cursor.execute(sql_str)
    data1 = cursor.fetchall()
    cursor.close()
    db.close
    StockDataDF = pd.DataFrame(data1)
    StockDataDF.columns = ["code", "tradingday", "high_limit", "low_limit"]
    return StockDataDF


def GetIndexNamelistFromSql():
    """从数据库中获取所有板块名称已经对应的板块代码"""
    db = pymssql.connect(
        host="198.16.102.88", user="sa", password="Sy123456", database="stock_data"
    )
    cursor = db.cursor()
    sql_str = "SELECT distinct [concept_code],[concept_name] FROM [stock_data].[dbo].[lgc_板块成份股]"
    cursor.execute(sql_str)
    data1 = cursor.fetchall()
    cursor.close()
    db.close
    StockDataDF = pd.DataFrame(data1)
    StockDataDF.columns = ["concept_code", "concept_name"]
    return StockDataDF


def GetTradableStocksFromSql():
    """从数据库中获取所有股票上市退市时间"""
    db = pymssql.connect(
        host="198.16.102.88", user="sa", password="Sy123456", database="stock_data"
    )
    cursor = db.cursor()
    sql_str = "SELECT [ipo_date],[code],[exchange_id],[last_trade_day],[tradingday],[trade_status] FROM [stock_data].[dbo].[all_stocks]"
    cursor.execute(sql_str)
    data1 = cursor.fetchall()
    cursor.close()
    db.close
    StockDataDF = pd.DataFrame(data1)
    StockDataDF.columns = [
        "ipo_date",
        "code",
        "exchange_id",
        "last_trade_day",
        "tradingday",
        "trade_status",
    ]
    return StockDataDF


def GetIndexComponentFromSql(IndexCode="all", begindate=20160101):
    """从数据库中获取板块成份信息"""
    db = pymssql.connect(
        host="198.16.102.88", user="sa", password="Sy123456", database="stock_data"
    )
    cursor = db.cursor()
    if IndexCode == "all":
        sql_str = (
            "select [tradingday],[sel_day],[指数类型],[concept_code],[concept_name],[code] from [stock_data].[dbo].[lgc_板块成份股] where tradingday= %d  "
            % begindate
        )
    else:
        sql_str0 = "select [tradingday],[sel_day],[指数类型],[concept_code],[concept_name],[code] from [stock_data].[dbo].[lgc_板块成份股] where tradingday={begindate}  and concept_code={IndexCode}"
        sql_str = sql_str0.format(begindate=begindate, IndexCode=IndexCode)
    cursor.execute(sql_str)
    data1 = cursor.fetchall()
    cursor.close()
    db.close
    StockDataDF = pd.DataFrame(data1)
    StockDataDF.columns = [
        "tradingday",
        "sel_day",
        "指数类型",
        "concept_code",
        "concept_name",
        "code",
    ]
    return StockDataDF


def GetIndexPriceFromSql(IndexCodes, begindate, enddate):
    """从数据库中获取板块价格信息"""
    db = pymssql.connect(
        host="198.16.102.88", user="sa", password="Sy123456", database="stock_data"
    )
    cursor = db.cursor()
    sql_str0 = "SELECT [bankuai],[tradingday],[exchange_id],[index_name],[code],[o],[h],[l],[c],[v],[amt],[writing_day] FROM [stock_data].[dbo].[wind_index] where tradingday>={begindate} and tradingday <= {enddate} and code='{IndexCodes}' order by tradingday"
    sql_str = sql_str0.format(
        begindate=begindate, enddate=enddate, IndexCodes=IndexCodes
    )
    cursor.execute(sql_str)
    data1 = cursor.fetchall()
    cursor.close()
    db.close
    StockDataDF = pd.DataFrame(data1)
    StockDataDF.columns = [
        "bankuai",
        "tradingday",
        "exchange_id",
        "index_name",
        "code",
        "o",
        "h",
        "l",
        "c",
        "v",
        "amt",
        "writing_day",
    ]
    return StockDataDF


def GetAllTradingDatesFromSql():
    """从数据库中 所有交易日期"""
    db = pymssql.connect(
        host="198.16.102.88", user="sa", password="Sy123456", database="stock_data"
    )
    cursor = db.cursor()
    sql_str = "select [tradingday] from tradingday order by tradingday"
    cursor.execute(sql_str)
    Tradingdates = cursor.fetchall()
    Tradingdates = np.array(Tradingdates)
    cursor.close()
    db.close
    return Tradingdates


def GetAllSTStocksFromSql():
    # 从数据库中获取所有ST股票信息
    db = pymssql.connect(
        host="198.16.102.88", user="sa", password="Sy123456", database="stock_data"
    )
    cursor = db.cursor()
    sql_str = "SELECT [tradingday],[code],[exchange_id],[sec_name] FROM [stock_data].[dbo].[ST] order by tradingday "
    cursor.execute(sql_str)
    STStockname = cursor.fetchall()
    STStockname = np.array(STStockname)
    cursor.close()
    db.close
    return STStockname


def GetStock1minDataFromSql(StockCode, date):
    """从数据库读取1分钟行情数据"""
    db = pymssql.connect(
        host="198.16.102.88", user="sa", password="Sy123456", database="stock_min1"
    )
    cursor = db.cursor()
    StockData = np.array([])
    if isinstance(date, np.int32):
        date = [date]

    for i in range(len(date)):
        sql_str0 = "select [tradingday],[tradingtime],[o],[h],[l],[c],[v],[amt],[adjfactor] from [stock_min1].[dbo].[{StockCode}] where tradingday='{i}' order by tradingtime"
        sql_str = sql_str0.format(StockCode=StockCode, i=date[i])
        cursor.execute(sql_str)
        StockData0 = cursor.fetchall()
        StockData = np.append(StockData, StockData0)
    StockData = np.reshape(StockData, (int(len(StockData) / 9), 9))
    cursor.close()
    db.close
    return StockData


def GetStockDayDataDFFromSql(begindate=20130101, enddate=0):
    """从数据库中读取所有股票日线信息"""
    db = pymssql.connect(
        host="198.16.102.88", user="sa", password="Sy123456", database="stock_data"
    )
    cursor = db.cursor()
    if enddate == 0:
        sql_str = (
            "select [code],[tradingday],[o],[h],[l],[c],[v],[amt],[adjfactor],[total_shares],[free_float_shares],[exchange_id] from [stock_data].[dbo].[day5] where tradingday> %d order by  tradingday"
            % begindate
        )
    else:
        sql_str = (
            "select [code],[tradingday],[o],[h],[l],[c],[v],[amt],[adjfactor],[total_shares],[free_float_shares],[exchange_id] from [stock_data].[dbo].[day5] where tradingday> %d  and tradingday<= %d order by  tradingday"
            % begindate
        )
    cursor.execute(sql_str)
    StockData = cursor.fetchall()
    cursor.close()
    db.close
    StockDataDF = pd.DataFrame(StockData)
    StockDataDF.columns = [
        "code",
        "tradingday",
        "o",
        "h",
        "l",
        "c",
        "v",
        "amt",
        "adjfactor",
        "total_shares",
        "free_float_shares",
        "exchange_id",
    ]
    return StockDataDF


def GetForeshowFromSql():
    """从数据库中读取所有股票预报信息"""
    db = pymssql.connect(
        host="198.16.102.88", user="sa", password="Sy123456", database="stock_data"
    )
    cursor = db.cursor()
    sql_str = "SELECT [code],[reportday],[tradingday],[l],[h],[lb],[hb],[l_kf],[h_kf],[lb_kf],[hb_kf]   FROM [stock_data].[dbo].[foreshow]"
    cursor.execute(sql_str)
    data1 = cursor.fetchall()
    cursor.close()
    db.close
    StockDataDF = pd.DataFrame(data1)
    StockDataDF.columns = [
        "code",
        "reportday",
        "tradingday",
        "l",
        "h",
        "lb",
        "hb",
        "l_kf",
        "h_kf",
        "lb_kf",
        "hb_kf",
    ]
    return StockDataDF


def GetIPOdateFromSql():
    """从数据库中读取所有ipo日期"""
    db = pymssql.connect(
        host="198.16.102.88", user="sa", password="Sy123456", database="stock_data"
    )
    cursor = db.cursor()
    sql_str = "select [code],[ipo_date] from [stock_data].[dbo].[all_stocks] where tradingday='20220318'"
    cursor.execute(sql_str)
    StockData = cursor.fetchall()
    cursor.close()
    db.close
    StockDataDF = pd.DataFrame(StockData)
    return StockDataDF


def GetAllConceptNameFromSql():
    db = pymssql.connect(
        host="198.16.102.88", user="sa", password="Sy123456", database="jqdata"
    )
    cursor = db.cursor()
    sql_str = "SELECT distinct concept_name  FROM [jqdata].[dbo].[concept_codes]"
    cursor.execute(sql_str)
    StockData = cursor.fetchall()
    cursor.close()
    db.close
    return StockData


def GetConceptComponentByNameFromSql(ConceptName):
    # 从数据库中获取概念板块成份信息
    db = pymssql.connect(
        host="198.16.102.88", user="sa", password="Sy123456", database="jqdata"
    )
    cursor = db.cursor()
    sql_str = (
        "SELECT [traddingday],[code],[exchangeid] FROM [jqdata].[dbo].[concept_codes] where concept_name='%S' order by tradingday "
        % ConceptName
    )
    cursor.execute(sql_str)
    StockData = cursor.fetchall()
    cursor.close()
    db.close
    return StockData


def GetWideBaseComponentFromSql(IndexCode):
    # 获取宽基指数成份股与权重
    """SH000300  SH000905"""
    db = pymssql.connect(
        host="198.16.102.88", user="sa", password="Sy123456", database="jqdata"
    )
    cursor = db.cursor()
    sql_str = (
        "select [tradingday],[index_code] ,[code],[exchange_id],[weight] from [jqdata].[dbo].[index_weights] where index_code='%s' order by tradingday"
        % IndexCode
    )
    cursor.execute(sql_str)
    StockData = cursor.fetchall()
    cursor.close()
    db.close
    StockDataDF = pd.DataFrame(StockData)
    StockDataDF.columns = [
        "TradingDates",
        "index_code",
        "StockCodes",
        "exchange_id",
        "weight",
    ]
    StockDataDF.set_index(["TradingDates", "StockCodes"], inplace=True)
    StockDataDF.sort_index(inplace=True)
    return StockDataDF


def GetAllannouncementFromSql():
    sql_str = "select [secCode],[announcementTitle],[tradingday],[tradingtime],[category] from [Wind].[dbo].[jczx_gg1] order by tradingday"
    db = pymssql.connect(
        host="198.16.102.88", user="sa", password="Sy123456", database="Wind"
    )
    cursor = db.cursor()
    cursor.execute(sql_str)
    StockData = cursor.fetchall()
    cursor.close()
    db.close
    StockDataDF = pd.DataFrame(StockData)
    StockDataDF.columns = [
        "secCode",
        "announcementTitle",
        "tradingday",
        "tradingtime",
        "category",
    ]
    return StockDataDF


def GetFinancialItemFromSql(SheetTitle, item):
    """获取指定表中指定条目数据"""
    sql_str0 = "select [code],[reportday],[tradingday],[d_quarter],[d_year],[{item}]  from [stock_data].[dbo].[{SheetTitle}]  order by tradingday"
    sql_str = sql_str0.format(item=item, SheetTitle=SheetTitle)
    db = pymssql.connect(
        host="198.16.102.88", user="sa", password="Sy123456", database="stock_data"
    )
    cursor = db.cursor()
    cursor.execute(sql_str)
    StockData = cursor.fetchall()
    cursor.close()
    db.close
    StockDataDF = pd.DataFrame(StockData)
    StockDataDF.columns = [
        "code",
        "reportday",
        "tradingday",
        "d_quarter",
        "d_year",
        item,
    ]
    return StockDataDF


def GetIndexPointFromSql(indexname):
    # 获取指数价格信息
    db = pymssql.connect(
        host="198.16.102.88", user="sa", password="Sy123456", database="stock_data"
    )
    cursor = db.cursor()
    sql_str = (
        "select [tradingday],[o],[h],[l],[c],[v],[amt] from [stock_data].[dbo].[wind_index] where index_name = '%s' order by  tradingday"
        % (indexname)
    )
    cursor.execute(sql_str)
    StockData = cursor.fetchall()
    StockData = pd.DataFrame(StockData)
    StockData.columns = ["tradingday", "o", "h", "l", "c", "v", "amt"]
    cursor.close()
    db.close
    return StockData


def GetIndustryOneHotFromApi(
            dateslist,
            indextype="申万行业板块",
            industry="sw_l2",
        ):
    cla = ComData.lgc_板块成份股(指数类型=indextype, industry=industry)
    Industry_one_hot_matrix = []
    if type(dateslist[0]) == pd._libs.tslibs.timestamps.Timestamp:
        dateslist = [int(i.strftime("%Y%m%d")) for i in dateslist]

    for i, date in enumerate(dateslist):
        成份股_data = cla.读取每日成分股(date, date)
        oh = pd.get_dummies(
            成份股_data,
            columns=["concept_name"],
            prefix_sep="_",
            dummy_na=False,
            drop_first=False,
        )
        if i == 0:
            Industry_one_hot_matrix = oh
        else:
            Industry_one_hot_matrix = pd.concat([Industry_one_hot_matrix, oh], axis=0)

    Industry_one_hot_matrix["tradingday"] = pd.to_datetime(
        Industry_one_hot_matrix["tradingday"], format="%Y%m%d"
    )
    Industry_one_hot_matrix["codes"] = Industry_one_hot_matrix["code"].astype(str)
    Industry_one_hot_matrix.rename(
        columns={"tradingday": "TradingDates", "codes": "StockCodes"}, inplace=True
    )
    Industry_one_hot_matrix.set_index(["TradingDates", "StockCodes"], inplace=True)
    classification_one_hot = Industry_one_hot_matrix.drop(
        columns=[
            "sel_day",
            "industry",
            "concept_code",
            "code",
            "code_cn",
            "exchange_id",
        ]
    )

    return classification_one_hot


def GetWideBaseByDateSerriesFromApi(
        dateslist,
        指数类型="沪深交易所核心指数",
        指数code="000905"
        ):
    cla = ComData.lgc_板块成份股(指数类型, 指数code)
    WideBase_matrix = []
    if type(dateslist[0]) == pd._libs.tslibs.timestamps.Timestamp:
        dateslist = [int(i.strftime("%Y%m%d")) for i in dateslist]
    for i, date in enumerate(dateslist):
        print(date)
        成份股_data = cla.读取每日成分股(date, date)
        if i == 0:
            WideBase_matrix = 成份股_data
        else:
            WideBase_matrix = pd.concat([WideBase_matrix, 成份股_data], axis=0)
    WideBase_matrix["tradingday"] = pd.to_datetime(
        WideBase_matrix["tradingday"], format="%Y%m%d"
    )
    WideBase_matrix["codes"] = WideBase_matrix["code"].astype(str)
    WideBase_matrix.rename(
        columns={"tradingday": "TradingDates", "codes": "StockCodes"}, inplace=True
    )
    WideBase_matrix.set_index(["TradingDates", "StockCodes"], inplace=True)
    return WideBase_matrix


def GetFinancialh5fileFromLAN_save(
            # 从局域网中获得财务数据h5文件
            sourcepath=r"\\198.16.102.88\lgc_g\股票数据",
            savepath=None,
        ):
    # 获取最新的财务数据文件
    if savepath is None:
        savepath = r"E:\Documents\PythonProject\StockProject\StockData"
    if not os.path.exists(sourcepath):
        raise FileNotFoundError("sourcepath not exists")

    f_index = os.path.join(sourcepath, "financial_v2.h5")
    savefilename = os.path.join(savepath, "financial_v2.h5")
    str_copy = "copy %s %s" % (f_index, savefilename)
    os.system(str_copy)
    return savefilename


def date_serries(PriceDf, type="daily"):
    Dateserries = PriceDf.index.get_level_values(0).unique()
    if type == "daily":
        return Dateserries
    if type == "weekly":
        weekly_mask = Dateserries.to_series().dt.to_period(
            "W"
        ) != Dateserries.to_series().shift(1).dt.to_period("W")
        return Dateserries[weekly_mask]
    if type == "monthly":
        monthly_mask = Dateserries.to_series().dt.to_period(
            "M"
        ) != Dateserries.to_series().shift(1).dt.to_period("M")
        return Dateserries[monthly_mask]


def run():
    print("os.getcwd()")
    GetFinancialh5fileFromLAN_save()
    PriceDf, StockTradableDF = GetAllDayPriceDataFromSql_save()

    monthlyserries = date_serries(PriceDf, type="monthly")
    IndustryOneHot = GetIndustryOneHotFromApi(
            monthlyserries, indextype="申万行业板块", industry="sw_l1"
            )  # 行业独热编码
    datasavepath = r"E:\Documents\PythonProject\StockProject\StockData"
    IndustryOneHot.to_pickle(datasavepath + r"\classification_one_hot.pkl")


if __name__ == "__main__":
    run()
   
