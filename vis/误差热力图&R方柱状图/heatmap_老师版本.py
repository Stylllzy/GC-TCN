import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = {
    'Market': ['Shanghai', 'Shanghai', 'Shanghai', 'Shanghai', 'Shanghai', 'Shanghai', 'Guangdong', 'Guangdong',
               'Guangdong', 'Guangdong', 'Guangdong', 'Guangdong', 'Hubei', 'Hubei', 'Hubei', 'Hubei', 'Hubei',
               'Hubei'],
    'Model': ['SVR', 'LSTM', 'GRU', 'TCN', 'TCN-Seq2Seq', 'DGCPN', 'SVR', 'LSTM', 'GRU', 'TCN', 'TCN-Seq2Seq', 'DGCPN',
              'SVR', 'LSTM', 'GRU', 'TCN', 'TCN-Seq2Seq', 'DGCPN'],
    'MAE': [1.7231, 0.5129, 0.6597, 0.2718, 0.2878, 0.2301, 1.3909, 1.7348, 0.7863, 0.7732, 0.7712, 0.3770, 1.0091,
            1.2606, 0.4926, 0.1781, 0.1451, 0.0857],
    'RMSE': [2.0445, 0.7756, 0.8364, 0.2282, 0.2953, 0.2000, 1.7764, 2.0679, 1.0085, 0.7970, 1.0015, 0.4792, 1.3389,
             1.4098, 0.6212, 0.2070, 0.1571, 0.0794],
    'MAPE': [0.0414, 0.0130, 0.0163, 0.0144, 0.0123, 0.0058, 0.0475, 0.0603, 0.0275, 0.0268, 0.0253, 0.0129, 0.0349,
             0.0443, 0.0171, 0.0092, 0.0116, 0.0066],
    'R2': [0.5461, 0.8954, 0.8783, 0.9509, 0.9734, 0.9943, 0.3189, 0.4435, 0.7725, 0.8579, 0.8010, 0.9515, 0.4992,
           0.4851, 0.9000, 0.9889, 0.9892, 0.9938]
}
df = pd.DataFrame(data)
# 指定Model和Market的顺序
model_order = ["SVR", "LSTM", "GRU", "TCN", "TCN-Seq2Seq", "DGCPN"]
market_order = ["Shanghai", "Guangdong", "Hubei"]

# 重排DataFrame以匹配指定顺序
df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
df["Market"] = pd.Categorical(df["Market"], categories=market_order, ordered=True)

# 排序
df_sorted = df.sort_values(by=["Model", "Market"])

sns.set(font_scale=1.5)
plt.rc('font', family='Times New Roman')
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
metrics = ['MAE', 'RMSE', 'MAPE', 'R2']

for i in range(2):
    for j in range(2):
        metric = metrics[i * 2 + j]
        ax = axes[i, j]
        metric_data = df_sorted.pivot(index='Model', columns='Market', values=metric)

        if metric == 'R2':
            cmap = 'RdYlBu'
        else:
            cmap = 'RdYlBu_r'

        sns.heatmap(metric_data, annot=True, cmap=cmap, fmt=".4f", ax=ax,
                    annot_kws={'size': 14, 'weight': 'bold'},
                    cbar_kws={'label': '', 'orientation': 'vertical', 'pad': 0.05},
                    linewidths=.5, linecolor='white')
        ax.set_xlabel(f"{metric}", fontsize=16, labelpad=10)
        ax.set_ylabel('')
        ax.tick_params(labelsize=16)
        ax.xaxis.tick_top()

plt.tight_layout()
# plt.savefig('图8.pdf', dpi=600, bbox_inches='tight')
plt.show()
