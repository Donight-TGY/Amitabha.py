import numpy as np
import pandas as pd
from PyEMD import CEEMDAN
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

# 不输出警告
warnings.filterwarnings("ignore")
# 输入原始数组
def loadPrices(fn):
    global nt, nInst
    #df=pd.read_csv(fn, sep='\s+', names=cols, header=None, index_col=0)
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    nt, nInst = df.values.shape
    return (df.values).T

def FCES(prcHistSoFar, nt):
    ## 初始化模型的数据--------------------------------------------------------------------------------------------------------
    pricesFile=".\prices_pre.txt"
    prcAll = loadPrices(pricesFile)
    prcAll = prcAll[:,-100:]
    
    test_time = 14 # 训练长度
    day = 2 # 预测长度
    
    times_CEEMD = 1 # CEEMD重构次数

    ## 正式开始预测--------------------------------------------------------------------------------------------------------
    if nt < test_time:
        prc = np.hstack((prcAll[:, -(test_time-nt):], prcHistSoFar))
        
        ## 进行模态分解

        # 获取数据的行数（向量数量）
        num_vectors = prc.shape[0]

        # 用于存储每个向量的IMF
        all_imfs = []

        # 对每一个向量进行CEEMD操作
        for i in range(num_vectors):
            ceemdan = CEEMDAN(trials=times_CEEMD)  # 进行5次CEEMD重构
            c_imfs = ceemdan(prc[i, :])
            new_c_imfs = np.sum(c_imfs[2:], axis=0)
            all_imfs.append(new_c_imfs)

        # 对低频分量进行预测

        forecasts = []

        for data in all_imfs:
            model = ExponentialSmoothing(data, trend="add", seasonal=None)
            model_fit = model.fit(smoothing_level=0.2)
            # 计算指数平均模型的预测
            forecast = model_fit.forecast(steps=day)  # 预测接下来的day天
            forecasts.append(forecast)

    else:
        prc = prcHistSoFar[:, -test_time:]
        
        ## 进行模态分解

        # 获取数据的行数（向量数量）
        num_vectors = prcAll.shape[0]

        # 用于存储每个向量的IMF
        all_imfs = []

        # 对每一个向量进行CEEMD操作
        for i in range(num_vectors):
            ceemdan = CEEMDAN(trials=times_CEEMD)  # 进行5次CEEMD重构
            c_imfs = ceemdan(prcAll[i, :])
            new_c_imfs = np.sum(c_imfs[2:], axis=0)
            all_imfs.append(new_c_imfs)

        # 对低频分量进行预测
        forecasts = []

        for data in all_imfs:

            model = ExponentialSmoothing(data, trend="add", seasonal=None)
            model_fit = model.fit(smoothing_level=0.2)
            # 计算指数平均模型的预测
            forecast = model_fit.forecast(steps=day)  # 预测接下来的day天
            forecasts.append(forecast)
        
    today_imfs = []
    for i in range(50):
        today_imfs.append(all_imfs[i][-2:])
    
    return today_imfs, forecasts