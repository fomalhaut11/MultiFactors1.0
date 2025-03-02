import os
import numpy as np
import pandas as pd
import pymssql
import h5py
import pickle
import gc


def get_realeaddates_formH5(datapath=None):
    if datapath is None:
        datapath = r"E:\Documents\PythonProject\StockProject\StockData"
    financialfile = os.path.join(datapath, "financial_v2.h5")
    financialdata = h5py.File(financialfile, "r")
    stockcodes_in_financialdata = financialdata["uni_code"][()]
    fzb = financialdata.get("fzb")[()]
    public_dates = fzb[:, :, 0]
    report_due_dates = fzb[:, :, 1]
    qnum = fzb[:, :, -3]
    ynum = fzb[:, :, -2]

    StockCodes = stockcodes_in_financialdata.repeat(np.shape(report_due_dates[0]))
    StockCodes = [code.decode("utf-8") for code in StockCodes]
    index1 = pd.MultiIndex.from_arrays(
        [StockCodes, pd.to_datetime(report_due_dates.reshape(-1), format="%Y%m%d")]
    )
    realesed_dates_df = pd.DataFrame(
        pd.to_datetime(public_dates.reshape(-1), format="%Y%m%d"),
        index=index1,
        columns=["ReleasedDates"],
    )
    qnum_df = pd.DataFrame(qnum.reshape(-1), index=index1, columns=["Quater"])
    ynum_df = pd.DataFrame(ynum.reshape(-1), index=index1, columns=["Year"])
    realesed_dates_df = pd.merge(
        realesed_dates_df, qnum_df, left_index=True, right_index=True
    )
    realesed_dates_df = pd.merge(
        realesed_dates_df, ynum_df, left_index=True, right_index=True
    )
    realesed_dates_df.index.names = ["StockCodes", "ReportDates"]
    return realesed_dates_df


def StockDataDF2Matrix(StockDataDF):
    levels = [StockDataDF["tradingday"].unique(), StockDataDF["code"].unique()]
    StockDataDF0 = StockDataDF.set_index(["tradingday", "code"])
    StockDataDF1 = StockDataDF0.reindex(
        pd.MultiIndex.from_product(levels), fill_value=np.nan
    )
    pricematrix = StockDataDF1.values.reshape(
        (len(levels[0]), len(levels[1]), np.shape(StockDataDF1.values)[1])
    )
    StockData3dMatrix = {
        "TradingDates": levels[0],
        "StockCodes": levels[1],
        "pricematrix": pricematrix,
        "datacolumns": StockDataDF1.columns,
    }
    return StockData3dMatrix


def get_price_data(savepath=None):
    if savepath is None:
        savepath = r"E:\Documents\PythonProject\StockProject\StockData"
    pricefile = os.path.join(savepath, "Price.pkl")
    PriceDF = pd.read_pickle(pricefile)
    tradablefile = os.path.join(savepath, "TradableDF.pkl")
    StockTradableDF = pd.read_pickle(tradablefile)
    PriceDF = PriceDF[~(PriceDF["exchange_id"] == "BJ")]  # 踢出北交所
    StockTradableDF = StockTradableDF[
        ~(StockTradableDF["exchange_id"] == "BJ")
    ]  # 踢出北交所
    PriceDF = PriceDF.join(
        StockTradableDF, how="left", lsuffix="_left", rsuffix="_right"
    )
    PriceDF = PriceDF[~(PriceDF["trade_status"] == "退市")]
    del StockTradableDF
    StockDataDF = PriceDF.copy()
    StockDataDF.index = PriceDF.index.set_names(["tradingday", "code"])
    StockDataDF = StockDataDF.reset_index()
    StockDataDF["tradingday"] = StockDataDF["tradingday"].dt.strftime("%Y%m%d")
    columns = [
        "tradingday",
        "code",
        "o",
        "h",
        "l",
        "c",
        "v",
        "amt",
        "adjfactor",
        "total_shares",
        "free_float_shares",
        "MC",
        "FMC",
        "turnoverrate",
        "vwap",
        "freeturnoverrate",
    ]
    StockDataDF = StockDataDF[columns]
    Stock3d = StockDataDF2Matrix(StockDataDF.copy(deep=True))
    del StockDataDF
    with open(savepath + r"\Stock3d.pkl", "wb") as f:
        pickle.dump(Stock3d, f)
    return PriceDF, Stock3d


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


def logreturndf_dateserries(
            PriceDF,
            datesserries,
            ReturnType="o2o",
            inputtype="unadjusted"
        ):
    next_log_return = []
    # 返回的收益率数据的时间戳为开仓时间
    if inputtype == "unadjusted":

        for i in range(len(datesserries) - 1):

            if ReturnType == "o2o":
                openprice = (
                    PriceDF.loc[datesserries[i], :]["o"]
                    * PriceDF.loc[datesserries[i], :]["adjfactor"]
                )
                closeprice = (
                    PriceDF.loc[datesserries[i + 1], :]["o"]
                    * PriceDF.loc[datesserries[i + 1], :]["adjfactor"]
                )
            if ReturnType == "c2c":
                openprice = (
                    PriceDF.loc[datesserries[i], :]["c"]
                    * PriceDF.loc[datesserries[i], :]["adjfactor"]
                )
                closeprice = (
                    PriceDF.loc[datesserries[i + 1], :]["c"]
                    * PriceDF.loc[datesserries[i + 1], :]["adjfactor"]
                )
            if ReturnType == "vwap":
                openprice = (
                    PriceDF.loc[datesserries[i], :]["vwap"]
                    * PriceDF.loc[datesserries[i], :]["adjfactor"]
                )
                closeprice = (
                    PriceDF.loc[datesserries[i + 1], :]["vwap"]
                    * PriceDF.loc[datesserries[i + 1], :]["adjfactor"]
                )

                pass
            logreturn = np.log(closeprice / openprice)
            logreturn.name = datesserries[i]
            next_log_return.append(logreturn)
        next_log_return = pd.concat(next_log_return, axis=1)
        next_log_return = next_log_return.unstack().to_frame()
        next_log_return.columns = ["LogReturn"]
        next_log_return.index.names = ["TradingDates", "StockCodes"]
        return next_log_return
    if inputtype == "adjusted":
        print("不支持adjusted")
    pass


def released_dates_count(realesed_dates_df, TradingDates):
    # 计算每个交易日距离最近的财报发布日的时间差
    realesed_dates_df.groupby("Quater")

    def row_calc(row, TradingDates):
        indices = np.searchsorted(row, TradingDates.values.flatten(), side="right") - 1
        time_diffs = (
            TradingDates.values.flatten() - row.iloc[indices].values
        ) / np.timedelta64(1, "D")
        time_diffs = pd.Series(time_diffs, index=TradingDates.iloc[:, 0])
        return time_diffs

    for i in range(4):
        print(i)
        quater1data = realesed_dates_df.loc[realesed_dates_df["Quater"] == 1 + i]
        rd = quater1data["ReleasedDates"].to_frame().unstack()
        time_diffs = rd.apply(row_calc, axis=1, args=(TradingDates,))
        df = time_diffs.reset_index()
        df = df.melt(id_vars="StockCodes", var_name="TradingDates")
        df.set_index(["TradingDates", "StockCodes"], inplace=True)
        columnnames = "Quater" + str(i + 1)
        df.columns = [columnnames]
        if i == 0:
            DateCount_df = df
        else:
            DateCount_df = pd.merge(DateCount_df, df, left_index=True, right_index=True)
    return DateCount_df


def run(datasavepath=None):
    if datasavepath is None:
        datasavepath = r"E:\Documents\PythonProject\StockProject\StockData"
    PriceDF, Stock3d = get_price_data()
    dailyserries = date_serries(PriceDF, type="daily")
    LogReturn_daily = logreturndf_dateserries(PriceDF, dailyserries, ReturnType="o2o")
    pd.to_pickle(LogReturn_daily, datasavepath + r"\LogReturn_daily_o2o.pkl")
    LogReturn_daily = logreturndf_dateserries(PriceDF, dailyserries, ReturnType="vwap")
    pd.to_pickle(LogReturn_daily, datasavepath + r"\LogReturn_daily_vwap.pkl")
    LogReturn_daily = None
    weeklyserries = date_serries(PriceDF, type="weekly")
    LogReturn_weekly = logreturndf_dateserries(PriceDF, weeklyserries, ReturnType="o2o")
    pd.to_pickle(LogReturn_weekly, datasavepath + r"\LogReturn_weekly_o2o.pkl")
    LogReturn_weekly = logreturndf_dateserries(
        PriceDF, weeklyserries, ReturnType="vwap"
    )
    pd.to_pickle(LogReturn_weekly, datasavepath + r"\LogReturn_weekly_vwap.pkl")
    LogReturn_weekly = None
    monthlyserries = date_serries(PriceDF, type="monthly")
    LogReturn_monthly = logreturndf_dateserries(
        PriceDF, monthlyserries, ReturnType="o2o"
    )
    pd.to_pickle(LogReturn_monthly, datasavepath + r"\LogReturn_monthly_o2o.pkl")
    LogReturn_monthly = logreturndf_dateserries(
        PriceDF, monthlyserries, ReturnType="vwap"
    )
    pd.to_pickle(LogReturn_monthly, datasavepath + r"\LogReturn_monthly_vwap.pkl")

    LogReturn_monthly = None
    TradingDates0 = PriceDF.index.get_level_values(0).unique().tolist()
    PriceDF = None
    gc.collect()
    realesed_dates_df = get_realeaddates_formH5()
    TradingDates = pd.DataFrame(TradingDates0, columns=["date"])
    realesed_dates_count_df = released_dates_count(realesed_dates_df, TradingDates)
    pd.to_pickle(
        realesed_dates_count_df, datasavepath + r"\realesed_dates_count_df.pkl"
    )
    return


if __name__ == "__main__":
    print("main")
    run()
