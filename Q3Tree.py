import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 初始化标量值
scalar_values = {
    '装配错误率': 0.1,
    '半成品检测成本': 4,
    '半成品和成品装配成本': 8,
    '半成品拆解费用': 6,
    '成品拆解费用': 10,
    '成品调换损失': 40
}

data = {
    '零件次品率': [0.1] * 8,  # 8个零件的{次品率}
    '零件单价': [2, 8, 12, 2, 8, 12, 8, 12],  # 8个零件的单价
    '零件检测成本': [1, 1, 2, 1, 1, 2, 1, 2],  # 8个零件的检测成本
    '零件购买成本': [2, 8, 12, 2, 8, 12, 8, 12]  # 8个零件的购买成本
}

df = pd.DataFrame([data])

# 定义计算函数
def calculate_failure_rate(component_failure_rates, assembly_error_rate=0.1, detection_decisions=None):
    if detection_decisions is None:
        detection_decisions = [0] * len(component_failure_rates)
    adjusted_failure_rates = [(0 if detection_decisions[i] == 1 else component_failure_rates[i]) for i in range(len(component_failure_rates))]
    product_failure_rate = 1 - np.prod([1 - rate for rate in adjusted_failure_rates]) * (1 - assembly_error_rate)
    return product_failure_rate

def calculate_half_product_failure_rates(components_failure_rates, detection_decisions):
    half_product_1 = calculate_failure_rate(components_failure_rates[:3], assembly_error_rate=scalar_values['装配错误率'], detection_decisions=detection_decisions[:3])
    half_product_2 = calculate_failure_rate(components_failure_rates[3:6], assembly_error_rate=scalar_values['装配错误率'], detection_decisions=detection_decisions[3:6])
    half_product_3 = calculate_failure_rate(components_failure_rates[6:], assembly_error_rate=scalar_values['装配错误率'], detection_decisions=detection_decisions[6:])
    return [half_product_1, half_product_2, half_product_3]

def calculate_final_product_failure_rate(half_product_failure_rates, detection_decisions):
    return calculate_failure_rate(half_product_failure_rates, assembly_error_rate=scalar_values['装配错误率'], detection_decisions=detection_decisions[:3])

def calculate_discard_cost(component_failure_rates, component_costs, assembly_costs, detection_decisions, discard_decisions):
    total_discard_cost = 0
    for i in range(8):
        total_discard_cost += component_costs[i]
    half_product_failure_rates = calculate_half_product_failure_rates(component_failure_rates, detection_decisions[:8])
    for j in range(3):
        if j < 2:
            indices = range(j * 3, (j + 1) * 3)
        else:
            indices = range(6, 8)
        if detection_decisions[8 + j] == 1:
            half_product_cost = assembly_costs[j] + sum(component_costs[k] for k in indices)
            total_discard_cost += half_product_cost
    final_product_failure_rate = calculate_final_product_failure_rate(half_product_failure_rates, detection_decisions[8:12])
    if detection_decisions[11] == 1:
        final_product_cost = assembly_costs[0]
        total_discard_cost += final_product_cost
    return total_discard_cost

def calculate_recovery_benefit(component_failure_rates, component_costs, detection_decisions, product_failure_rate):
    recovery_benefit = product_failure_rate * sum((1 - component_failure_rates[i]) * component_costs[i] for i in range(len(component_failure_rates)) if detection_decisions[i] == 1)
    return recovery_benefit

def calculate_total_cost(components_failure_rates, component_costs, detection_costs, assembly_costs, disassemble_costs, decision_variables):
    total_cost = 0
    for i in range(8):
        total_cost += component_costs[i]
        if decision_variables[i] == 1:
            total_cost += detection_costs[i]
    half_product_failure_rates = calculate_half_product_failure_rates(components_failure_rates, decision_variables[:8])
    for j in range(3):
        if decision_variables[8 + j] == 1:
            total_cost += assembly_costs[j] + half_product_failure_rates[j] * disassemble_costs[j]
        else:
            total_cost += assembly_costs[j]
        if decision_variables[12 + j] == 1 and half_product_failure_rates[j] > 0:
            if j < 2:
                indices = range(j * 3, (j + 1) * 3)
            else:
                indices = range(6, 8)
            recovery_benefit = calculate_recovery_benefit(
                [components_failure_rates[k] for k in indices],
                [component_costs[k] for k in indices],
                decision_variables[:8],
                half_product_failure_rates[j]
            )
            total_cost -= recovery_benefit
    final_product_failure_rate = calculate_final_product_failure_rate(half_product_failure_rates, decision_variables[8:12])
    if decision_variables[11] == 1:
        total_cost += assembly_costs[0] + final_product_failure_rate * disassemble_costs[0]
    else:
        total_cost += assembly_costs[0]
    if decision_variables[15] == 1 and final_product_failure_rate > 0:
        recovery_benefit = calculate_recovery_benefit(components_failure_rates, component_costs, decision_variables[:8], final_product_failure_rate)
        total_cost -= recovery_benefit
    total_cost += final_product_failure_rate * scalar_values['成品调换损失']
    discard_cost = calculate_discard_cost(components_failure_rates, component_costs, assembly_costs, decision_variables[:12], decision_variables[12:])
    total_cost += discard_cost
    return total_cost

# 生成所有可能的16个决策变量组合
decision_combinations = [(x1, x2, x3, x4, x5, x6, x7, x8, y1, y2, y3, y4, z1, z2, z3, z4)
                         for x1 in [0, 1] for x2 in [0, 1] for x3 in [0, 1] for x4 in [0, 1]
                         for x5 in [0, 1] for x6 in [0, 1] for x7 in [0, 1] for x8 in [0, 1]
                         for y1 in [0, 1] for y2 in [0, 1] for y3 in [0, 1] for y4 in [0, 1]
                         for z1 in [0, 1] for z2 in [0, 1] for z3 in [0, 1] for z4 in [0, 1]]

# 模拟每种组合的{总成本}
def simulate_costs_for_combinations(df):
    costs = []
    decision_data = []
    for index, row in df.iterrows():
        cost_row = []
        for comb in decision_combinations:
            total_cost = calculate_total_cost(
                row['零件次品率'],
                row['零件单价'],
                row['零件检测成本'],
                [scalar_values['半成品和成品装配成本']] * 3,
                [scalar_values['成品拆解费用']] * 4,
                comb
            )
            cost_row.append(total_cost)
            decision_data.append(comb)
        costs.append(cost_row)
    return pd.DataFrame(costs, columns=decision_combinations), decision_data

cost_df, decision_data = simulate_costs_for_combinations(df)

# 准备{决策树}数据
X = np.array(decision_data)
y = cost_df.values.flatten()

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练{决策树}模型
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# 预测并寻找最佳方案
predicted_costs = model.predict(X)
optimal_index = np.argmin(predicted_costs)
optimal_decision = decision_combinations[optimal_index]

print(f"最佳决策方案: {optimal_decision}")
print(f"最优方案的总成本: {predicted_costs[optimal_index]}")

# 回归指标
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mse)

# 转换为分类任务以计算F1值
threshold = np.mean(y)  # 使用{平均成本}作为阈值
# 使用拆分后的 y_test 而不是整个 y
y_test_labels = (y_test < threshold).astype(int)

# 预测转换为分类任务以计算 F1 值
predictions_labels = (predictions < threshold).astype(int)

# 计算F1值
f1 = f1_score(y_test_labels, predictions_labels)

# 特征重要性
feature_importances = model.feature_importances_

# 将特征重要性与特征名称配对
importance_df = pd.DataFrame({'Feature': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8',
                                          'y1', 'y2', 'y3', 'y4', 'z1', 'z2', 'z3', 'z4'],
                              'Importance': feature_importances})

# 按重要性排序
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("特征的重要性:")
print(importance_df)

# KFold 交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

r2_scores = []
rmse_scores = []
f1_scores = []
accuracies = []
sensitivities = []

for train_index, test_index in kf.split(X):
    X_train_kf, X_test_kf = X[train_index], X[test_index]
    y_train_kf, y_test_kf = y[train_index], y[test_index]

    model_kf = DecisionTreeRegressor()
    model_kf.fit(X_train_kf, y_train_kf)
    predictions_kf = model_kf.predict(X_test_kf)

    mse_kf = mean_squared_error(y_test_kf, predictions_kf)
    r2_kf = r2_score(y_test_kf, predictions_kf)
    rmse_kf = np.sqrt(mse_kf)

    # 转换为分类标签
    y_test_kf_labels = (y_test_kf < threshold).astype(int)
    predictions_kf_labels = (predictions_kf < threshold).astype(int)

    # 计算F1值
    f1_kf = f1_score(y_test_kf_labels, predictions_kf_labels)

    # 计算混淆矩阵，并根据混淆矩阵计算准确率和灵敏度（召回率）
    cm = confusion_matrix(y_test_kf_labels, predictions_kf_labels)
    accuracy_kf = accuracy_score(y_test_kf_labels, predictions_kf_labels)
    recall_kf = recall_score(y_test_kf_labels, predictions_kf_labels)

    r2_scores.append(r2_kf)
    rmse_scores.append(rmse_kf)
    f1_scores.append(f1_kf)
    accuracies.append(accuracy_kf)
    sensitivities.append(recall_kf)

# 输出最终平均回归指标和F1值
print("平均 R²:", np.mean(r2_scores))
print("平均均方根误差 (RMSE):", np.mean(rmse_scores))
print("平均 F1 值:", np.mean(f1_scores))
print(f"平均准确率 (ACC): {np.mean(accuracies)}")
print(f"平均灵敏度 (Sens): {np.mean(sensitivities)}")

# 绘制特征重要性的柱状图
plt.figure(figsize=(10, 6))
plt.bar(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.axhline(y=importance_df['Importance'].mean(), color='red', linestyle='--', label='平均重要性')
plt.title('特征重要性柱状图')
plt.xlabel('决策变量')
plt.ylabel('重要性')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
# 显示图像
plt.show()


