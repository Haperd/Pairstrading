#另起炉灶法把之前的拉起来。USA这个是最新的我要实际跑的。

"""每个交易日交易时段内打开运行一次，手动1.更改时间 2.运行 3.更改state."""

# 初始化 =========================================================================

import akshare as ak
import pandas as pd
import numpy as np
import time
import schedule
from datetime import datetime, timedelta
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.common.util.contract_utils import stock_contract
from tigeropen.common.util.order_utils import market_order
from tigeropen.trade.trade_client import TradeClient
from tigeropen.push.push_client import PushClient
from tigeropen.push.pb.QuoteBasicData_pb2 import QuoteBasicData
import ssl  # 这段import得放在开头才不报错
ssl._create_default_https_context = ssl._create_unverified_context


def get_client_config():
    client_config = TigerOpenClientConfig(props_path='')
    return client_config


client_config = get_client_config()
trade_client = TradeClient(client_config)

# 信号 =========================================================================

symbols = ['GPRO', 'MICS']
start_date = "2024-02-01"
end_date = "2024-03-01"
stock_data_all = pd.DataFrame()
for symbol in symbols:
    stock_data = ak.stock_us_daily(symbol=symbol, adjust="")

    # 如果 stock_data 是 None，表示获取数据失败
    if stock_data is not None:
        # 将 "date" 列设置为索引
        stock_data.set_index("date", inplace=True)

        # 保留 "open" 列，使用 symbol 作为列名
        stock_data_all[symbol] = stock_data["open"]
stock_data_all = stock_data_all.loc[start_date:end_date]


def z_test():
    regression_ratio = 0.3016
    stock1 = np.array(stock_data_all["GPRO"])
    stock2 = np.array(stock_data_all["MICS"])
    stable_series = stock2 - regression_ratio * stock1
    series_mean = np.mean(stable_series)
    sigma = np.std(stable_series)
    diff = stable_series[-1] - series_mean
    return diff / sigma


def get_signal():
    z_score = z_test()
    if z_score >= 2:
        return 'highstop'
    elif 1 < z_score < 2:
        # 状态为做多stock1做空stock2
        return 'high'
    # 如果小于负标准差
    elif -2 < z_score < -1:
        # 状态为做多stock2做空stock1
        return 'low'
    elif z_score <= -2:
        return 'lowstop'
    # 如果在正负标准差之间
    elif -1 <= z_score <= 1:
        # 如果差大于0
        if z_score >= 0:
            # 在均值上面
            return 'upside'
        # 反之
        else:
            # 在均值下面
            return 'downside'


new_state = get_signal()

# 下单 =========================================================================


def give_order(symbol, action, quantity):
    '''lmt_order'''
    # 生成股票合约
    contract = stock_contract(symbol=symbol, currency='USD')
    # 生成订单对象
    order = market_order(account=client_config.account, contract=contract, action=action, quantity=quantity)
    # 下单
    oid = trade_client.place_order(order)
    # 打印
    print(order)
    print(order.status)
    print(order.reason)

    return oid


def change_positions(new_state):
    state = 'empty'
    if new_state == 'high'and state != 'high':
        give_order('GPRO', 'BUY', 0.3016*100)
        give_order('MICS', 'SELL', 100)
        state = 'high'
    if new_state == 'low'and state != 'low':
        give_order('GPRO', 'SELL', 0.3016*100)
        give_order('MICS', 'BUY', 100)
        state = 'low'
    if state == 'high' and new_state == 'downside':
        give_order('GPRO', 'SELL', 0.3016*100)
        give_order('MICS', 'BUY', 100)
        state = 'quit'
    if state == 'low' and new_state == 'upside':
        give_order('GPRO', 'BUY', 0.3016*100)
        give_order('MICS', 'SELL', 100)
        state = 'quit'
    if new_state == 'highstop'and state != 'highstop':
        give_order('GPRO', 'SELL', 0.3016*100)
        give_order('MICS', 'BUY', 100)
        state = 'highstop'
    if new_state == 'lowstop'and state != 'lowstop':
        give_order('GPRO', 'BUY', 0.3016*100)
        give_order('MICS', 'SELL', 100)
        state = 'lowstop'
    print(state)


change_positions(new_state)

