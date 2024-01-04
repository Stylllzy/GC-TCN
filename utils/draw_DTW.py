import pandas as pd
from tslearn.metrics import dtw_path
import numpy as np
import matplotlib.pyplot as plt

from utils.util import moving_average

data = pd.read_csv('../dataset/tanjiaoyi/gd_carbon.csv', encoding='GBK', parse_dates=['date'],
                   index_col='date').dropna()

# 对数据进行标准化
data = (data - data.mean()) / data.std()

# close列为标签列, 选取前100个数据作为序列A
sequence_a = data['close'].values[800:900]
sequence_b = data['deal_EUA'].values[800:900]

# 使序列平滑
sequence_a = moving_average(sequence_a, 10)
sequence_b = moving_average(sequence_b, 10)

# 使用tslearn的dtw_path计算DTW对齐路径
path, sim = dtw_path(sequence_a.reshape(-1, 1), sequence_b.reshape(-1, 1))

# 绘制两个序列和连接线在同一个子图上
plt.figure(figsize=(12, 6))

# 绘制序列A和序列B
plt.plot(sequence_a, label='Carbon Price', color='#000000')
plt.plot(sequence_b + max(sequence_a), label='influence factor', color='#CF0A0A')
# 去掉坐标轴的值
plt.xticks([])
plt.yticks([])
plt.xlabel('Time', fontsize=18)
plt.ylabel('Value', fontsize=18)

# 调整字体大小以及xlabel和ylabel的字体大小
plt.rcParams.update({'font.size': 16})

# 绘制连接线，考虑到序列B的偏移
offset = max(sequence_a)
for (i, j) in path:
    plt.plot([i, j], [sequence_a[i], sequence_b[j] + offset], c='grey', alpha=0.5)

# 添加图例和标题
plt.legend()
# plt.title('DTW Alignment of Two Sequences')
plt.savefig('DTW_path.png', dpi=300)
plt.show()