"""
    shap工具包
"""

import pandas as pd
import xgboost
import shap
from matplotlib import pyplot as plt

df = pd.read_csv('../dataset/tanjiaoyi/gd_carbon.csv', encoding='GBK', parse_dates=['date'],
                 index_col='date').dropna()

X = df.drop(df.columns[0], axis=1).values
y = df['close'].values

# train an XGBoost model
model = xgboost.XGBRegressor().fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X)

# visualize the first prediction's explanation
# shap.plots.waterfall(shap_values[0])
shap.plots.heatmap(shap_values)