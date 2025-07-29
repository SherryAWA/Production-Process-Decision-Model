import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# 随机生成多组数据
num_samples = 50000
data = {
    'p1': np.random.uniform(0.05, 0.2, num_samples),  # {零配件1}次品率
    'C1': np.random.randint(3, 6, num_samples),  # {零配件1}单价
    'Cd1': np.random.randint(1, 9, num_samples),  # {零配件1}检测成本
    'p2': np.random.uniform(0.05, 0.2, num_samples),  # {零配件2}次品率
    'C2': np.random.randint(15, 20, num_samples),  # {零配件2}单价
    'Cd2': np.random.randint(1, 4, num_samples),  # {零配件2}检测成本
    'pf0': np.random.uniform(0.05, 0.2, num_samples),  # 装配后的{成品}次品率
    'Ca': np.random.randint(5, 7, num_samples),  # {成品}装配成本
    'Cd': np.random.randint(2, 4, num_samples),  # {成品}检测成本
    'Cr': np.random.randint(6, 31, num_samples),  # 调换损失
    'Cs': np.random.randint(5, 41, num_samples),  # 拆解费用
}

df = pd.DataFrame(data)

# 生成所有可能的决策组合
decision_combinations = [(x1, x2, y, z) for x1 in [0, 1] for x2 in [0, 1] for y in [0, 1] for z in [0, 1]]

# 模拟决策逻辑 (此处为了简化，将随机选择作为决策结果)
optimal_decisions = np.random.choice(range(len(decision_combinations)), num_samples)

# 特征与目标
X = df[['p1', 'C1', 'Cd1', 'p2', 'C2', 'Cd2', 'pf0', 'Ca', 'Cd', 'Cr', 'Cs']]
y = optimal_decisions

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练原始模型并计算特征重要性
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 获取特征重要性
feature_importances = model.feature_importances_

# 找到重要度较高的特征（如取重要度前3的特征）
important_features_indices = np.argsort(feature_importances)[-3:]  # 取前3重要的特征索引
important_features = X.columns[important_features_indices]

print("重要特征:", important_features)

# 对这些重要特征施加扰动
perturbation_scale = 0.1  # 设定扰动大小为10%


def apply_perturbation(data, feature, scale):
    perturbed_data = data.copy()
    noise = np.random.uniform(-scale, scale, size=len(perturbed_data))  # 生成随机噪声
    perturbed_data[feature] += noise * perturbed_data[feature]  # 对特征施加扰动
    return perturbed_data


# 创建一个用于存储扰动后模型性能变化的列表
perturbation_results = []

# 对每个重要特征分别施加扰动并评估模型性能
for feature in important_features:
    print(f"正在对 {feature} 施加扰动...")

    # 对训练集和测试集分别施加扰动
    X_train_perturbed = apply_perturbation(X_train, feature, perturbation_scale)
    X_test_perturbed = apply_perturbation(X_test, feature, perturbation_scale)

    # 重新训练模型并进行预测
    model_perturbed = DecisionTreeClassifier(random_state=42)
    model_perturbed.fit(X_train_perturbed, y_train)
    predictions_perturbed = model_perturbed.predict(X_test_perturbed)

    # 评估模型性能
    accuracy_perturbed = accuracy_score(y_test, predictions_perturbed)
    precision_perturbed = precision_score(y_test, predictions_perturbed, average='macro')
    recall_perturbed = recall_score(y_test, predictions_perturbed, average='macro')
    f1_perturbed = f1_score(y_test, predictions_perturbed, average='macro')
    rmse_perturbed = np.sqrt(mean_squared_error(y_test, predictions_perturbed))

    # 将结果保存
    perturbation_results.append({
        'feature': feature,
        'accuracy': accuracy_perturbed,
        'precision': precision_perturbed,
        'recall': recall_perturbed,
        'f1': f1_perturbed,
        'rmse': rmse_perturbed
    })

# 转换为 DataFrame 以便展示结果
perturbation_df = pd.DataFrame(perturbation_results)
print(perturbation_df)

# 可视化模型在不同重要特征扰动下的表现
plt.figure(figsize=(12, 8))
for metric in ['accuracy', 'precision', 'recall', 'f1', 'rmse']:
    plt.plot(perturbation_df['feature'], perturbation_df[metric], marker='o', label=metric)

plt.title('Model Performance under Perturbations on Important Features')
plt.xlabel('Perturbed Feature')
plt.ylabel('Performance Metrics')
plt.legend()
plt.show()
