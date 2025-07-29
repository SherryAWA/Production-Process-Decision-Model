
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
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

# 动态计算成品次品率
def calculate_dynamic_pf(row, x1, x2):
    pf_dynamic = 1 - (1 - (1 - x1) * row['p1']) * (1 - (1 - x2) * row['p2']) * (1 - row['pf0'])
    return pf_dynamic

# 计算回收收益
def calculate_recovery(row, pf_dynamic):
    recovery_benefit_1 = pf_dynamic * (1 - row['p1']) * row['C1']
    recovery_benefit_2 = pf_dynamic * (1 - row['p2']) * row['C2']
    return recovery_benefit_1 + recovery_benefit_2

# 计算总成本
def calculate_cost(row, x1, x2, y, z):
    purchase_cost = row['C1'] + row['C2']
    detection_cost = x1 * row['Cd1'] + x2 * row['Cd2']
    discard_part_cost = x1 * row['p1'] * row['C1'] + x2 * row['p2'] * row['C2']

    if (x1 == 0 or (x1 == 1 and row['p1'] < 1)) and (x2 == 0 or (x2 == 1 and row['p2'] < 1)):
        assembly_cost = row['Ca']
    else:
        assembly_cost = 0

    pf_dynamic = calculate_dynamic_pf(row, x1, x2)

    discard_product_cost = 0
    dismantle_cost = 0
    recovery_benefit = 0
    exchange_loss = 0
    detection_cost_product = 0

    if y == 1:
        detection_cost_product = row['Cd']
        if z:
            dismantle_cost = pf_dynamic * row['Cs']
            recovery_benefit = calculate_recovery(row, pf_dynamic)
        else:
            discard_product_cost = pf_dynamic * (assembly_cost + x1 * row['Cd1'] + x2 * row['Cd2'])
    else:
        exchange_loss = pf_dynamic * row['Cr']
        if z:
            dismantle_cost = pf_dynamic * row['Cs']
            recovery_benefit = calculate_recovery(row, pf_dynamic)
        else:
            discard_product_cost = pf_dynamic * (assembly_cost + x1 * row['Cd1'] + x2 * row['Cd2'])

    total_cost = (purchase_cost + detection_cost + discard_part_cost + discard_product_cost +
                  assembly_cost + dismantle_cost + exchange_loss + detection_cost_product - recovery_benefit)

    return total_cost

# 生成所有可能的决策组合
decision_combinations = [(x1, x2, y, z) for x1 in [0, 1] for x2 in [0, 1] for y in [0, 1] for z in [0, 1]]

# 为每种情况计算成本
costs = []
for index, row in df.iterrows():
    cost_row = []
    for comb in decision_combinations:
        cost = calculate_cost(row, *comb)
        cost_row.append(cost)
    costs.append(cost_row)

# 将成本转换为DataFrame格式
cost_df = pd.DataFrame(costs, columns=decision_combinations)

# 寻找最优决策组合
optimal_decisions = cost_df.idxmin(axis=1)

# 使用决策树模型验证决策
X = df[['p1', 'C1', 'Cd1', 'p2', 'C2', 'Cd2', 'pf0', 'Ca', 'Cd', 'Cr', 'Cs']]
y = optimal_decisions.map(lambda x: decision_combinations.index(x))

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用分类模型
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 计算分类指标
conf_matrix = confusion_matrix(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='macro')
recall = recall_score(y_test, predictions, average='macro')  # 灵敏度
f1 = f1_score(y_test, predictions, average='macro')

# R² 和 均方根误差
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mse)

print("混淆矩阵:\n", conf_matrix)
print("准确率 (ACC):", accuracy)
print("灵敏度 (Recall):", recall)
print("特异度 (Specificity):", (conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])) if conf_matrix.shape[0] > 1 else 'N/A')
print("F1值:", f1)
print("R²:", r2)
print("均方根误差 (RMSE):", rmse)

# 可视化混淆矩阵
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# KFold 交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

acc_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
r2_scores = []
rmse_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    acc_scores.append(accuracy_score(y_test, predictions))
    precision_scores.append(precision_score(y_test, predictions, average='macro'))
    recall_scores.append(recall_score(y_test, predictions, average='macro'))
    f1_scores.append(f1_score(y_test, predictions, average='macro'))

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mse)

    r2_scores.append(r2)
    rmse_scores.append(rmse)

# 输出最终平均分类结果和回归指标
print("平均准确率 (ACC):", np.mean(acc_scores))
print("平均灵敏度 (Recall):", np.mean(recall_scores))
print("平均 F1 值:", np.mean(f1_scores))
print("平均 R²:", np.mean(r2_scores))
print("平均均方根误差 (RMSE):", np.mean(rmse_scores))
