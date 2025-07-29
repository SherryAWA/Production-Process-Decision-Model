import math
import numpy as np
import matplotlib.pyplot as plt

# 初始化参数
p0 = 0.10  # 标称次品率
p1 = 0.15  # 假设超过标称值的次品率
alpha = 0.05  # 第一类错误概率（拒收）
beta = 0.10  # 第二类错误概率（接收）

# 计算上下限的对数值
lnA = math.log((1 - beta) / alpha)
lnB = math.log(beta / (1 - alpha))

# 定义函数计算对数似然比
def log_likelihood(x, n, p0, p1):
    return x * math.log(p1 / p0) + (n - x) * math.log((1 - p1) / (1 - p0))

# SPRT算法实现
def sprt_test(p_true, p0, p1, alpha, beta, max_samples=2000):
    n = 0  # 样本总数
    x = 0  # 不合格品数量

    for i in range(max_samples):
        n += 1
        # 随机生成一个样本是否不合格（0表示合格，1表示不合格）
        sample = np.random.binomial(1, p_true)
        x += sample  # 更新不合格品数量

        # 计算当前对数似然比
        lnL = log_likelihood(x, n, p0, p1)

        # 决策阶段
        if lnL >= lnA:
            return False, n  # 拒绝该批次，返回拒收并且记录样本数量
        elif lnL <= lnB:
            return True, n  # 接受该批次，返回接受并且记录样本数量

    return None, n  # 达到最大样本数但未决策

# 模拟OC和ASN曲线
def simulate_oc_asn(p0, p1, alpha, beta, p_range, num_simulations=5000):
    oc_curve = []
    asn_curve = []

    for p_true in p_range:
        accept_count = 0
        total_samples = 0

        for _ in range(num_simulations):
            accept, samples = sprt_test(p_true, p0, p1, alpha, beta)
            if accept is not None:  # 仅统计有效结果
                accept_count += int(accept)
            total_samples += samples

        # 计算接受批次的比例 (OC曲线)
        oc_curve.append(accept_count / num_simulations)
        # 计算平均样本数量 (ASN曲线)
        asn_curve.append(total_samples / num_simulations)

    return oc_curve, asn_curve

# 定义次品率的范围
p_range = np.linspace(0.05, 0.30, 26)

# 运行模拟以获取OC和ASN曲线
oc_curve, asn_curve = simulate_oc_asn(p0, p1, alpha, beta, p_range)

# 绘制OC曲线
plt.figure(figsize=(10, 6))
plt.plot(p_range, oc_curve, label='OC曲线', color='blue', linewidth=2)
plt.axvline(x=p0, color='red', linestyle='--', label='p0 = 0.10')
plt.axvline(x=p1, color='red', linestyle='--', label='p1 = 0.20')
plt.title('操作特性（OC）曲线', fontsize=14)
plt.xlabel('真实次品率', fontsize=12)
plt.ylabel('接受概率', fontsize=12)
plt.xticks(np.arange(0.05, 0.31, 0.05))  # 设置横坐标的间隔为 0.05
plt.grid(True)
plt.legend(loc='best')
plt.show()

# 绘制ASN曲线
plt.figure(figsize=(10, 6))
plt.plot(p_range, asn_curve, label='ASN曲线', color='green', linewidth=2)
plt.axvline(x=p0, color='red', linestyle='--', label='p0 = 0.10')
plt.axvline(x=p1, color='red', linestyle='--', label='p1 = 0.20')
plt.title('平均样本量（ASN）曲线', fontsize=14)
plt.xlabel('真实次品率', fontsize=12)
plt.ylabel('平均样本量', fontsize=12)
plt.xticks(np.arange(0.05, 0.31, 0.05))  # 设置横坐标的间隔为 0.05
plt.grid(True)
plt.legend(loc='best')
plt.show()
