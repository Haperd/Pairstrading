import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import akshare as ak


# 输入是一DataFrame，每一列是一支股票在每一日的价格
def find_cointegrated_pairs(dataframe):
    # 得到DataFrame长度
    n = dataframe.shape[1]
    # 初始化p值矩阵
    pvalue_matrix = np.ones((n, n))
    # 抽取列的名称
    keys = dataframe.keys()
    # 初始化强协整组
    pairs = []
    # 对于每一个i
    for i in range(n):
        # 对于大于i的j
        for j in range(i+1, n):
            # 获取相应的两只股票的价格Series
            stock1 = dataframe[keys[i]]
            stock2 = dataframe[keys[j]]
            # 分析它们的协整关系
            result = sm.tsa.stattools.coint(stock1, stock2)
            '''协整检验函数在这里'''
            # 取出并记录p值
            pvalue = result[1]
            pvalue_matrix[i, j] = pvalue
            # 如果p值小于0.05
            if pvalue < 0.05:
                '''为什么用0.05'''
                # 记录股票对和相应的p值
                pairs.append((keys[i], keys[j], pvalue))
    # 返回结果
    return pvalue_matrix, pairs


# 定义多个 symbols
symbols = ['NIO', 'XPEV', 'MNSO', 'TIGR', 'SOHU', 'LI', 'JD', 'BZ', 'BABA', 'NTES','BIDU','YUMC','VIPS','EDU','KRKR','PDD','IQ','BILI']
"暂时手动输入吧，先确定行业"

# 定义时间范围
start_date = "2024-01-01"
end_date = "2024-03-16"

# 存储所有股票数据的 DataFrame
stock_price = pd.DataFrame()

# 获取每支股票的数据并添加到 DataFrame
for symbol in symbols:
    stock_data = ak.stock_us_daily(symbol=symbol, adjust="")

    # 如果 stock_data 是 None，表示获取数据失败
    if stock_data is not None:
        # 将 "date" 列设置为索引
        stock_data.set_index("date", inplace=True)

        # 保留 "open" 列，使用 symbol 作为列名
        stock_price[symbol] = stock_data["open"]

# 使用切片选择特定时间范围的数据
stock_price = stock_price.loc[start_date:end_date]

# 打印结果
print(stock_price)

pvalues, pairs = find_cointegrated_pairs(stock_price)

plt.figure(figsize=(10, 8))
sns.heatmap(1-pvalues, xticklabels=symbols, yticklabels=symbols, cmap='RdYlGn_r', mask = (pvalues == 1))
print(pairs)
plt.show()

stock_df1 = stock_price['JD']
stock_df2 = stock_price["BILI"]
'''这里加一个自动传入'''
plt.plot(stock_df1); plt.plot(stock_df2)
plt.xlabel("Time"); plt.ylabel("Price")
plt.legend(["JD", "BILI"],loc='best')
plt.show()

x = stock_df1
y = stock_df2
X = sm.add_constant(x)
result = (sm.OLS(y,X)).fit()
print(result.summary())

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x, y, 'o', label="data")
ax.plot(x, result.fittedvalues, 'r', label="OLS")
ax.legend(loc='best')
plt.show()

plt.plot(0.5120*stock_df1-stock_df2)
plt.axhline((0.5120*stock_df1-stock_df2).mean(), color="red", linestyle="--")
plt.xlabel("Time"); plt.ylabel("Stationary Series")
plt.legend(["Stationary Series", "Mean"])
plt.show()


def zscore(series):
    return (series - series.mean()) / np.std(series)


plt.plot(zscore(0.5120*stock_df1-stock_df2))
plt.axhline(zscore(0.5120*stock_df1-stock_df2).mean(), color="black")
plt.axhline(1.0, color="red", linestyle="--")
plt.axhline(-1.0, color="green", linestyle="--")
plt.legend(["z-score", "mean", "+1", "-1"])
plt.show()

#入场放在从外面回来之后怎么样？