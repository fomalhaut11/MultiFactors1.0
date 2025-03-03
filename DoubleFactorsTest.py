import pandas as pd
import numpy as np
from SingleFactorTest import Remove_Outlier, quick_remove_outlier_np, Normlization, single_factor_test_data, half_decay_factor
import matplotlib.pyplot as plt
from risk_model import sequential_orthog_df, dataloader
import factorsmaking as fm


def double_factors_test_2triple(factordata, logreturn, rollingwindow=20, beta1=None, beta2=None):
    
    factordata = factordata.join(logreturn, how="left")
    factordata = factordata.dropna(subset=["LogReturn"])
    
    icdata = factordata.groupby('TradingDates').apply(
        lambda x: x[['data1', 'LogReturn']].corr().loc['data1']['LogReturn']
            ).to_frame('ic1').shift(1)
    icdata['ic1_rolling'] = icdata['ic1'].rolling(rollingwindow).mean()
    icdata['ic2'] = factordata.groupby('TradingDates').apply(
        lambda x: x[['data2', 'LogReturn']].corr().loc['data2']['LogReturn']
            ).to_frame('ic2').shift(1)
    icdata['ic2_rolling'] = icdata['ic2'].rolling(rollingwindow).mean()

    def _indayfunc_proportion(dayslice, icdata, beta1=None, beta2=None):
        current_date = dayslice.index.get_level_values('TradingDates')[0]
        print(current_date)
        icdata_slice = icdata.loc[icdata.index.get_level_values('TradingDates') == current_date]
        if icdata_slice.isna().any().any():
            return np.nan, np.nan, np.nan, np.nan, []
        if beta1 is None:
            beta1 = 1
            if icdata_slice['ic1_rolling'].item() < 0:
                beta1 = -1
        if beta2 is None:
            beta2 = 1
            if icdata_slice['ic2_rolling'].item() < 0:
                beta2 = -1

        dayslice['group1'] = pd.qcut(dayslice['data1']*beta1, q=3, labels=False, duplicates='drop')
        dayslice['group2'] = pd.qcut(dayslice['data2']*beta2, q=3, labels=False, duplicates='drop')
        dayslice['group_logreturn'] = pd.qcut(dayslice['LogReturn'], q=3, labels=False, duplicates='drop')
        data1_group2_slice = dayslice[dayslice['group1'] == 2].copy()
        data1_group2_slice['group_data2'] = pd.qcut(data1_group2_slice['data2'], q=3, labels=False, duplicates='drop')
        data1_group2_data2_group2 = data1_group2_slice[data1_group2_slice['group_data2'] == 2]
        data1_group2_data2_group2_accuracy = (data1_group2_data2_group2['group_logreturn'] ==2).mean()
        data1_group2_slice_accuracy = (data1_group2_slice['group_logreturn'] ==2).mean()
        improve_rate = data1_group2_data2_group2_accuracy - data1_group2_slice_accuracy

        data1_group0_slice = dayslice[dayslice['group1'] == 0].copy()
        data1_group0_slice['group_data2'] = pd.qcut(data1_group0_slice['data2'], q=3, labels=False, duplicates='drop')
        data1_group0_data2_group0 = data1_group0_slice[data1_group0_slice['group_data2'] == 0]
        data1_group0_data2_group0_accuracy = (data1_group0_data2_group0['group_logreturn'] ==0).mean()
        data1_group0_slice_accuracy = (data1_group0_slice['group_logreturn'] ==0).mean()
        improve_rate1 = data1_group0_data2_group0_accuracy - data1_group0_slice_accuracy
 
        newfactor = dayslice['data1'].copy()
        std_facor = newfactor.std()
        mask1 = data1_group2_data2_group2.index
        mask2 = data1_group0_data2_group0.index

        newfactor.loc[mask1] = newfactor.loc[mask1] + 4*std_facor
        newfactor.loc[mask2] = newfactor.loc[mask2] - 4*std_facor

        return improve_rate, improve_rate1, data1_group2_slice_accuracy, data1_group0_slice_accuracy, newfactor.loc[current_date]
    improvedata = factordata.groupby('TradingDates').apply(
        lambda x: _indayfunc_proportion(x, icdata)
        )
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    improvedata.apply(lambda x: x[0]).rolling(rollingwindow).mean().plot(ax=axs[0], label = 'LongImprovedRate')
    improvedata.apply(lambda x: x[1]).rolling(rollingwindow).mean().plot(ax=axs[0], label = 'ShortImprovedRate')
    axs[0].legend()
    improvedata.apply(lambda x: x[2]).rolling(rollingwindow).mean().plot(ax=axs[1], label = 'Factor1_long_accuracy')
    improvedata.apply(lambda x: x[3]).rolling(rollingwindow).mean().plot(ax=axs[1], label = 'Factor1_short_accuracy') 
    axs[1].legend()
    plt.show()
    result_list = []
    for date, group in improvedata.items():
        if group[0] == np.nan:
            continue
        if len(group[4]) == 0:
            continue
        df = group[4].to_frame(name='factor')
        df['TradingDates'] = date
        result_list.append(df)
    adjfactor = pd.concat(result_list)
    adjfactor = adjfactor.reset_index().set_index(['TradingDates', 'StockCodes'])
    return improvedata, adjfactor


if __name__ == "__main__":
    print("main")
    datapath = r"E:\Documents\PythonProject\StockProject\StockData"
    PriceDf = pd.read_pickle(datapath + "\\" + "Price.pkl")
    StockTradableDF = pd.read_pickle(datapath + "\\" + "TradableDF.pkl")
    PriceDf = PriceDf[~(PriceDf["exchange_id"] == "BJ")]  # 踢出北交所
    StockTradableDF = StockTradableDF[
        ~(StockTradableDF["exchange_id"] == "BJ")
    ]  # 踢出北交所

    PriceDf = PriceDf.join(
        StockTradableDF, how="left", lsuffix="_left", rsuffix="_right"
    )
    PriceDf = PriceDf[~(PriceDf["trade_status"] == "退市")]
    logreturn = pd.read_pickle(datapath + "\\" + "LogReturn_daily_o2o.pkl")
    realead_dates_count_df = pd.read_pickle(
            datapath + "\\" + "realesed_dates_count_df.pkl"
            )

    datapath = r"E:\Documents\PythonProject\StockProject\StockData"
    basenames = ['LogMarketCap']
    factornames = ['ROEzscores_4', 'zz2000_252_60_alfa']
    basedata = dataloader(basenames)
    factordata_init = dataloader(factornames)
    factordata_init.columns = ['data1', 'data2']  
    data1 = basedata.join(factordata_init, how="left")
    factordata = sequential_orthog_df(data1)
    improvedata, newfactor  = double_factors_test_2triple(factordata, logreturn, 60)
    test = single_factor_test_data_grouped(newfactor,
                        basenormed, logreturn)
 


    test = single_factor_test_data_grouped(fm.half_decay_factor(factordata['data1'].to_frame(name='factor'),realead_dates_count_df,20),
                    basenormed, logreturn)                    
    test2 = single_factor_test_data_grouped(fm.half_decay_factor(newfactor,realead_dates_count_df,20),
                    basenormed, logreturn)                    

    basenormed = factordata.drop(columns=['data1', 'data2'])
 
    test = single_factor_test_data_grouped(factordata['data1'].to_frame(name='factor'),
                        basenormed, logreturn)