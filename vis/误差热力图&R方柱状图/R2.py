# -*- coding: gbk -*-
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator

# 设置全局字体和字体大小
plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['axes.labelsize'] = 18  # x和y标签的字体大小
plt.rcParams['xtick.labelsize'] = 20  # x轴刻度的字体大小
plt.rcParams['ytick.labelsize'] = 20  # y轴刻度的字体大小
plt.rcParams['legend.fontsize'] = 20  # 图例的字体大小

# 更新数据框架，包括两个指标MSE和R2
df = pd.DataFrame({
    'Model': ['(EMD)-SVR', '(EMD)-LSTM', '(EMD)-GRU', '(EMD)-TCN', '(EMD)-TAE',
              '(EEMD)-SVR', '(EEMD)-LSTM', '(EEMD)-GRU', '(EEMD)-TCN', '(EEMD)-TAE',
              '(CEEMDAN)-SVR', '(CEEMDAN)-LSTM', '(CEEMDAN)-GRU', '(CEEMDAN)-TCN', '(CEEMDAN)-TAE'] * 2,
    'Metric': ['MAPE']*15 + ['R2']*15,
    'Value': [
        2.5388, 1.5027, 1.5325, 1.1729, 0.6752,
        2.6684, 1.4482, 1.5087, 1.1985, 0.6505,
        2.4700, 1.1754, 1.3681, 1.1227, 0.5323,
        0.3000, 0.6072, 0.5915, 0.7607, 0.9207,
        0.2268, 0.6352, 0.6040, 0.7501, 0.9264,
        0.3375, 0.7597, 0.6744, 0.7807, 0.9507
    ]
})

# 绘制条形图
plt.figure(figsize=(15, 10))
barplot = sns.barplot(x="Model", y="Value", hue="Metric", data=df, palette=["#1B3C73", "#FF407D"],
                      edgecolor="black", linewidth=1.5)

# 去掉图例标题
plt.legend().set_title(None)

# 设置标题和轴标签
plt.title('')
plt.xlabel('')
plt.ylabel('')

# 旋转X轴标签以便它们更容易阅读
# 旋转X轴标签以便它们更容易阅读
x_ticks = barplot.get_xticks()
barplot.xaxis.set_major_locator(FixedLocator(x_ticks))
barplot.set_xticklabels(barplot.get_xticklabels(), rotation=45, horizontalalignment='right')
# # 设置柱状的宽度
# for patch in barplot.patches:
#     current_width = patch.get_width()
#     new_width = current_width * 0.6
#     patch.set_width(new_width)
#     # We also recenter the bar
#     patch.set_x(patch.get_x() + current_width * 0.2)

# 去掉网格线
plt.grid(False)

# 展示图形
plt.tight_layout()
plt.show()


