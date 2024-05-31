# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import pandas as pd


def load_data(filepath):
    df = pd.read_csv(filepath, encoding='ISO-8859-1', parse_dates=['date'], index_col='date')
    total_size = len(df)
    test_size = int(total_size * 0.1)
    val_size = int(total_size * 0.1)
    train_size = total_size - val_size - test_size
    return df['close'], train_size, val_size, test_size


# Load data
df, train_size, val_size, test_size = load_data('../../dataset/tanjiaoyi/sh_carbon.csv')

plt.figure(figsize=(12, 6))

plt.rcParams.update({
        'font.size': 28,  # 修改字体大小
        'font.family': 'Times New Roman',
        'xtick.major.size': 5,  # x轴刻度大小
        'xtick.major.width': 1,  # x轴刻度宽度
        'ytick.major.size': 5,  # y轴刻度大小
        'ytick.major.width': 1,  # y轴刻度宽度
        'axes.linewidth': 1.5,  # 轴线宽度
    })

# 绘制训练集部分
plt.plot(df.iloc[:train_size], label="Training Set", color='#FF7C7C')

# 绘制验证集部分
plt.plot(df.iloc[train_size:train_size + val_size], label="Validation Set", color='#9382EA')

# 绘制测试集部分
plt.plot(df.iloc[-test_size:], label="Test Set", color='#4AA3F6')

plt.xlabel('Time', fontdict={'family': 'Times New Roman', 'size': 28})
plt.ylabel('Carbon Price', fontdict={'family': 'Times New Roman', 'size': 28})
# 设置 x 轴和 y 轴刻度朝内
plt.tick_params(axis='both', direction='in', labelsize=24)
plt.legend(loc='lower left')
plt.tight_layout()
# 去掉网格
plt.grid(False)

# 如果需要保存图像，取消注释以下行并指定保存路径
# plt.savefig('../vis/sqlit_hb.png', dpi=300)

plt.show()
