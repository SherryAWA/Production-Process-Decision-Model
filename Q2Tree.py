import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree

data = {
    '情况': [1, 2, 3, 4, 5, 6],
    'p1': [0.0814,0.1752,0.0814,0.1752,0.0814,0.0365],  # 零配件1次品率
    'C1': [4, 4, 4, 4, 4, 4],  # 零配件1单价
    'Cd1': [2, 2, 2, 1, 8, 2],  # 零配件1检测成本
    'p2': [0.0814, 0.1752, 0.0814, 0.1752, 0.1752, 0.0365],  # 零配件2次品率
    'C2': [18, 18, 18, 18, 18, 18],  # 零配件2单价
    'Cd2': [3, 3, 3, 1, 1, 3],  # 零配件2检测成本
    'pf0': [0.814, 0.1752, 0.0814, 0.1752, 0.0814, 0.0365],  # 装配后的成品次品率
    'Ca': [6, 6, 6, 6, 6, 6],  # 成品装配成本
    'Cd': [3, 3, 3, 2, 2, 3],  # 成品检测成本
    'Cr': [6, 6, 30, 30, 10, 10],  # 调换损失
    'Cs': [5, 5, 5, 5, 5, 40],  # 拆解费用
}

df = pd.DataFrame(data)

def calculate_dynamic_pf(row, x1, x2):
    pf_dynamic = 1 - (1 - (1 - x1) * row['p1']) * (1 - (1 - x2) * row['p2']) * (1 - row['pf0'])
    return pf_dynamic

def calculate_recovery(row, pf_dynamic):
    recovery_benefit_1 = pf_dynamic * (1 - row['p1']) * row['C1']
    recovery_benefit_2 = pf_dynamic * (1 - row['p2']) * row['C2']
    return recovery_benefit_1 + recovery_benefit_2

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

def plot_custom_tree(decision_tree, feature_names):
    fig, ax = plt.subplots(figsize=(20, 10))
    tree.plot_tree(decision_tree, feature_names=feature_names, filled=True, ax=ax)
    for i, pair in enumerate(zip(ax.lines[::2], ax.lines[1::2])):
        x1, y1 = pair[0].get_xdata()[1], pair[0].get_ydata()[1]
        x2, y2 = pair[1].get_xdata()[1], pair[1].get_ydata()[1]
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2
        decision = "是" if i % 2 == 0 else "否"
        ax.text(xm, ym, decision, fontsize=12, va='center', ha='center', backgroundcolor='w')

for situation in df['情况'].unique():
    subset_df = df[df['情况'] == situation]

    results = []
    for index, row in subset_df.iterrows():
        for x1 in [0, 1]:
            for x2 in [0, 1]:
                for y in [0, 1]:
                    for z in [0, 1]:
                        total_cost = calculate_cost(row, x1, x2, y, z)
                        results.append({
                            'x1': x1,
                            'x2': x2,
                            'y': y,
                            'z': z,
                            '总成本': total_cost
                        })

    results_df = pd.DataFrame(results)

    X = results_df[['x1', 'x2', 'y', 'z']]
    y = results_df['总成本']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = DecisionTreeRegressor(random_state=0)
    model.fit(X_train, y_train)

    min_cost_index = y.idxmin()
    optimal_decision = results_df.iloc[min_cost_index]
    print(f"情况 {situation} 最佳决策参数:", optimal_decision)

    plot_custom_tree(model, feature_names=['零配件1检测', '零配件2检测', '成品检测', '是否拆解'])
    plt.title(f"情况 {situation} 决策树")
    plt.show()

# 针对情况一，输出每种决策组合下的成本
situation1_df = df[df['情况'] == 1]

results = []
for index, row in situation1_df.iterrows():
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            for y in [0, 1]:
                for z in [0, 1]:
                    total_cost = calculate_cost(row, x1, x2, y, z)
                    results.append({
                        'x1': x1,
                        'x2': x2,
                        'y': y,
                        'z': z,
                        '总成本': total_cost
                    })

results_df = pd.DataFrame(results)
print("情况 1 决策组合及成本：")
print(results_df)
