import math

# 给定的次品率数据
data = {
    'p1': [0.1, 0.2, 0.1, 0.2, 0.1, 0.05],  # 零配件1次品率
    'p2': [0.1, 0.2, 0.1, 0.2, 0.2, 0.05],  # 零配件2次品率
    'pf0': [0.1, 0.2, 0.1, 0.2, 0.1, 0.05],  # 装配后的成品次品率
}

# 假设的样本量 n
n = 1000
# Z值 (95%信度下为1.96)
z_95 = 1.96

# 计算置信区间的函数
def calculate_confidence_interval(p_hat, n, z_value):
    """
    计算给定样本次品率的95%置信区间
    :param p_hat: 样本次品率
    :param n: 样本量
    :param z_value: 对应的 z 值 (1.96 for 95% confidence)
    :return: (置信区间下限, 置信区间上限)
    """
    se = math.sqrt(p_hat * (1 - p_hat) / n)  # 标准误差
    lower_bound = p_hat - z_value * se
    upper_bound = p_hat + z_value * se
    return max(0, lower_bound), min(1, upper_bound)  # 保证概率范围在 [0,1]

# 计算所有次品率的置信区间
def calculate_all_confidence_intervals(data, n, z_value):
    confidence_intervals = {}
    for key, defect_rates in data.items():
        confidence_intervals[key] = []
        for p_hat in defect_rates:
            ci_lower, ci_upper = calculate_confidence_interval(p_hat, n, z_value)
            confidence_intervals[key].append((ci_lower, ci_upper))
    return confidence_intervals

# 计算置信区间
confidence_intervals = calculate_all_confidence_intervals(data, n, z_95)

# 输出每个次品率的置信区间上下限
for key, intervals in confidence_intervals.items():
    print(f"{key}的95%置信区间:")
    for i, (lower, upper) in enumerate(intervals):
        print(f"  情况 {i+1}: [{lower:.4f}, {upper:.4f}]")
