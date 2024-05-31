import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Provided data in a structured format
data = {
    'Models': ['(EMD)-SVR', '(EMD)-LSTM', '(EMD)-GRU', '(EMD)-TCN', '(EMD)-TAE',
               '(EEMD)-SVR', '(EEMD)-LSTM', '(EEMD)-GRU', '(EEMD)-TCN', '(EEMD)-TAE',
               '(CEEMDAN)-SVR', '(CEEMDAN)-LSTM', '(CEEMDAN)-GRU', '(CEEMDAN)-TCN', '(CEEMDAN)-TAE'],
    'MSE': [
        6.4457, 2.2581, 2.3484, 1.3757, 0.4559,
        7.2377, 2.0972, 2.2762, 1.4363, 0.4231,
        6.1011, 1.3816, 1.8718, 1.2605, 0.2833,
    ],
    'MAE': [
        2.0361, 1.1617, 1.1947, 0.9512, 0.4999,
        2.1958, 1.1248, 1.2063, 0.9119, 0.5047,
        1.9472, 0.9192, 1.1168, 0.9028, 0.4073
    ],
    'RMSE': [
        2.5388, 1.5027, 1.5325, 1.1729, 0.6752,
        2.6684, 1.4482, 1.5087, 1.1985, 0.6505,
        2.4700, 1.1754, 1.3681, 1.1227, 0.5323
    ],
    'MAPE': [
        0.0485, 0.0294, 0.0299, 0.0235, 0.0123,
        0.0518, 0.0281, 0.0302, 0.0226, 0.0125,
        0.0464, 0.0232, 0.0282, 0.0225, 0.0101
    ]
}

# Creating DataFrame
df = pd.DataFrame(data)
# Creating the heatmap with specified adjustments
sns.set(font_scale=1.5)
plt.rc('font', family='Times New Roman', size=12)
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(df.set_index('Models'), annot=True, fmt=".4f",
                      cmap="RdBu_r", center=4,
                      annot_kws={'size': 13, 'weight': 'bold'},
                      linewidths=.5, linecolor='white',
                      cbar_kws={'orientation': 'vertical', 'format': '%.4f'},
                      )

#
cbar = heatmap.collections[0].colorbar
cbar.ax.yaxis.set_ticks_position('right')
cbar.ax.yaxis.set_label_position('right')

# Removing the axis labels but leaving the ticks
plt.xlabel('')
plt.ylabel('')
plt.xticks(rotation=0, fontsize=15)
plt.yticks(rotation=0, fontsize=15)

# Applying the tight layout
plt.tight_layout()

# Save the figure to a file
# plt.savefig('heatmap.pdf', dpi=600, bbox_inches='tight')
plt.show()
