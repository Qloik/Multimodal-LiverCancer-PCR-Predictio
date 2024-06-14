import json
import matplotlib.pyplot as plt
import math

# 从JSON文件中读取准确率数据
with open('json.json', 'r') as f:
    accuracy_data = json.load(f)

# 提取epoch和accuracy数据
epochs = [entry[1] for entry in accuracy_data]
accuracies = [entry[2] for entry in accuracy_data]

# 根据准确率计算 Focal Loss
gamma = 3  # Focal Loss 的超参数 gamma
alpha = 0.4  # Focal Loss 的超参数 alpha
epsilon = 1e-7  # 防止除零错误的小常数

# 根据 Focal Loss 的定义计算损失值
losses = [-alpha * (1 - accuracy) ** gamma * math.log(accuracy + epsilon) for accuracy in accuracies]

# 绘制损失曲线
plt.plot([0] + epochs, [0.03] + losses, color='#525f73',  marker='', linestyle='-',markersize=0)
plt.title('Validation Focal Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Focal Loss')
plt.ylim(0, 2e-3)  # 设置纵坐标范围
plt.grid(True)
plt.show()
