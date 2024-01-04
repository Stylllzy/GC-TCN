"""
    LIME 工具包
"""
import numpy as np
import pandas as pd
import xgboost
from lime import lime_tabular
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

df = pd.read_csv('../dataset/tanjiaoyi/gd_carbon.csv', encoding='GBK', parse_dates=['date'],
                 index_col='date').dropna()
feature_names = df.columns  # 提取特征名

# 选择数据
X = df.drop(df.columns[0], axis=1).values
y = df['close'].values
# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = RandomForestRegressor().fit(X, y)
model = xgboost.XGBRegressor().fit(X_train, y_train)

# 生成解释器
explainer = lime_tabular.LimeTabularExplainer(
    X_train,
    mode='regression',
    feature_names=None,
    categorical_features=None,
    verbose=True,
    class_names=None)

# 对局部点的解释
i = np.random.randint(0, X_test.shape[0])  # 随机选择一个测试样本
exp = explainer.explain_instance(X_test[i], model, num_features=33)
# 显示详细信息图
exp.show_in_notebook(show_table=True, show_all=True)
# 显示权重图
exp.as_pyplot_figure()
