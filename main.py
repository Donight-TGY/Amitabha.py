import numpy as np
from forecast_CEEMD_ExponentialSmoothing import FCES

# Group Name: Amitabha
# Year: 2023
# https://github.com/Donight-TGY/Amitabha.py

nInst=50
currentPos = np.zeros(nInst)

def getMyPosition(prcHistSoFar):
    global currentPos
    pos_Limit = 8000
    limit = 10000

    # 假设 prcHistSoFar 是一个包含历史价格数据的 NumPy 二维数组，行表示不同的金融工具，列表示不同的交易日

    nInst, nt = prcHistSoFar.shape

    # 初始化新的交易仓位（头寸）数组，初始值都为 0
    newPos = np.zeros(nInst)

    # 调用 forecast 函数获取未来三天的价格预测
    today_imfs, forecasted_prices = FCES(prcHistSoFar, nt)

    # 假设这里实现一个交易策略：根据明日价格趋势方向进行头寸调整
    if nt == 1:
        for i in range(50):

            # 当前持有的金融工具的价格
            current_price = prcHistSoFar[i, -1]
            # 预测的未来三天价格
            future_prices = forecasted_prices[i]

            #判断做空还是做多
            if future_prices[0] <= future_prices[1]: #做多
                newPos[i] = int(pos_Limit/ current_price)
            elif future_prices[0] > future_prices[1]:#做空
                newPos[i] = - int(pos_Limit / current_price)
            
    else:
        for i in range(50):

            # 当前持有的金融工具的价格
            current_price = prcHistSoFar[i, -1]
            # 预测的未来三天价格
            future_prices = forecasted_prices[i]

            #判断做空还是做多
            if future_prices[0] <= future_prices[1]: #做多
                option = 1
            elif future_prices[0] > future_prices[1]:#做空
                option = -1
            cur = currentPos[-50:]
            
            poli = cur[i] + option * int((pos_Limit/2) / current_price)
            
            if abs(poli) >= limit:
                if poli > 0:
                    newPos[i] = limit
                else:
                    newPos[i] = -limit
            else:
                newPos[i] = cur[i] + option * int((pos_Limit/2) / current_price)

    currentPos = np.concatenate((currentPos, newPos), axis=0)

    return newPos