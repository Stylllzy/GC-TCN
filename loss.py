import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 创建一些样本数据
np.random.seed(0)
X = np.random.rand(100, 1) * 2 - 1  # 100个样本，1个特征
y = 3 * X.squeeze() + np.random.randn(100) * 0.5

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# 初始化SGDRegressor
model = SGDRegressor(max_iter=1, warm_start=True, penalty=None, learning_rate='constant', eta0=0.01)

# 记录损失和R²值
train_losses = []
val_losses = []
train_r2_scores = []
val_r2_scores = []

# 迭代训练模型
for _ in range(100):
    model.partial_fit(X_train, y_train)

    # 计算训练损失和R²
    train_predictions = model.predict(X_train)
    train_losses.append(mean_squared_error(y_train, train_predictions))
    train_r2_scores.append(r2_score(y_train, train_predictions))

    # 计算验证损失和R²
    val_predictions = model.predict(X_val)
    val_losses.append(mean_squared_error(y_val, val_predictions))
    val_r2_scores.append(r2_score(y_val, val_predictions))

# 绘制损失和R²值图表
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', c='red')
plt.plot(val_losses, label='Validation Loss', c='blue')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_r2_scores, label='Train R² Score', c='red')
plt.plot(val_r2_scores, label='Validation R² Score', c='blue')
plt.title('R² Score over Epochs')
plt.xlabel('Epoch')
plt.ylabel('R² Score')
plt.legend()

plt.tight_layout()
plt.show()
