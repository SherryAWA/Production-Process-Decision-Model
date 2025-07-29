import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
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

# 定义初始数据
data = {
    '零件次品率': [0.1] * 8,
    '零件单价': [2, 8, 12, 2, 8, 12, 8, 12],
    '零件检测成本': [1, 1, 2, 1, 1, 2, 1, 2],
    '零件购买成本': [2, 8, 12, 2, 8, 12, 8, 12]
}

df = pd.DataFrame([data])

# 添加装配错误率和半成品检测成本列
df['装配错误率'] = [scalar_values['装配错误率']] * len(df)
df['半成品检测成本'] = [scalar_values['半成品检测成本']] * len(df)

# 生成所有可能的16个决策变量组合
decision_combinations = [(x1, x2, x3, x4, x5, x6, x7, x8, y1, y2, y3, y4, z1, z2, z3, z4)
                         for x1 in [0, 1] for x2 in [0, 1] for x3 in [0, 1] for x4 in [0, 1]
                         for x5 in [0, 1] for x6 in [0, 1] for x7 in [0, 1] for x8 in [0, 1]
                         for y1 in [0, 1] for y2 in [0, 1] for y3 in [0, 1] for y4 in [0, 1]
                         for z1 in [0, 1] for z2 in [0, 1] for z3 in [0, 1] for z4 in [0, 1]]


# 计算总成本的函数
def calculate_total_cost(components_failure_rates, component_costs, detection_costs, assembly_costs, disassemble_costs,
                         decision_variables):
    total_cost = 0
    for i in range(8):
        total_cost += component_costs[i]
        if decision_variables[i] == 1:
            total_cost += detection_costs[i]
    total_cost += np.sum(components_failure_rates) * scalar_values['成品调换损失']
    return total_cost


# 模拟每种组合的总成本
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

# 准备数据
X = np.array(decision_data)
y = cost_df.values.flatten()

# 转换为分类任务
threshold = np.mean(y)  # 使用平均成本作为阈值
y_labels = (y < threshold).astype(int)

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42)


# 训练和预测各个模型的ROC曲线
def train_and_plot_roc(model, model_name, X_train, X_test, y_train, y_test, plot=True):
    model.fit(X_train, y_train)
    y_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc_score = roc_auc_score(y_test, y_scores)
    if plot:
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')
    return auc_score


# 模型初始化
models = {
    "决策树": DecisionTreeClassifier(),
    "随机森林": RandomForestClassifier(n_estimators=50, random_state=42),
    "梯度上升决策树": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

# 原始数据上的模型性能评估
original_auc_scores = {}
for name, model in models.items():
    auc_score = train_and_plot_roc(model, name, X_train, X_test, y_train, y_test, plot=False)
    original_auc_scores[name] = auc_score

# 对选定的三个变量进行扰动
perturbation_scale = 0.1  # 设置扰动大小
perturbation_columns = ['装配错误率', '半成品检测成本', '零件次品率']  # 选定的三个变量
perturbation_results = []


# 定义扰动函数
def apply_perturbation(df, column, scale):
    perturbed_df = df.copy()

    # 判断列中元素是否是序列类型
    if isinstance(perturbed_df[column].iloc[0], (list, np.ndarray)):
        # 如果是序列类型，对序列中的每个元素进行扰动
        perturbed_df[column] = perturbed_df[column].apply(
            lambda x: [v + np.random.uniform(-scale, scale) * v for v in x]
        )
    else:
        # 否则直接对标量进行扰动
        noise = np.random.uniform(-scale, scale)
        perturbed_df[column] += noise * perturbed_df[column]

    return perturbed_df


# 对每个变量进行扰动并评估模型
for col in perturbation_columns:
    print(f"对 {col} 进行扰动...")

    # 对数据进行扰动
    perturbed_df = apply_perturbation(df, col, perturbation_scale)

    # 重新计算成本和决策数据
    perturbed_cost_df, perturbed_decision_data = simulate_costs_for_combinations(perturbed_df)
    perturbed_X = np.array(perturbed_decision_data)
    perturbed_y = (perturbed_cost_df.values.flatten() < threshold).astype(int)

    # 重新划分数据集
    X_train_perturbed, X_test_perturbed, y_train_perturbed, y_test_perturbed = train_test_split(perturbed_X,
                                                                                                perturbed_y,
                                                                                                test_size=0.2,
                                                                                                random_state=42)

    # 评估模型在扰动数据上的性能
    for name, model in models.items():
        perturbed_auc_score = train_and_plot_roc(model, name, X_train_perturbed, X_test_perturbed, y_train_perturbed,
                                                 y_test_perturbed, plot=False)

        # 计算AUC变化
        auc_change = (perturbed_auc_score - original_auc_scores[name]) / original_auc_scores[name] * 100
        perturbation_results.append({
            'variable': col,
            'model': name,
            'original_auc': original_auc_scores[name],
            'perturbed_auc': perturbed_auc_score,
            'auc_change': auc_change
        })

# 转换为 DataFrame 以便展示结果
perturbation_df = pd.DataFrame(perturbation_results)
print(perturbation_df)

# 可视化扰动前后AUC变化
plt.figure(figsize=(12, 8))
for name in models.keys():
    model_data = perturbation_df[perturbation_df['model'] == name]
    plt.plot(model_data['variable'], model_data['auc_change'], marker='o', label=name)

plt.title('AUC Change After Perturbation for Selected Variables')
plt.xlabel('Perturbed Variable')
plt.ylabel('AUC Change (%)')
plt.legend(loc='best')
plt.show()
