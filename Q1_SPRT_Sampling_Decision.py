import math
import numpy as np
import matplotlib.pyplot as plt

# 初始化参数
p0 = 0.10  # 标称{次品率}
p1 = 0.15  # 假设超过标称值的{次品率}
alpha = 0.05  # {第一类错误概率}（拒收）
beta = 0.10  # {第二类错误概率}（接收）

# 计算上下限的对数值
lnA = math.log((1 - beta) / alpha)
lnB = math.log(beta / (1 - alpha))

# 定义函数计算{对数似然比}
def log_likelihood(x, n, p0, p1):
    return x * math.log(p1 / p0) + (n - x) * math.log((1 - p1) / (1 - p0))

# {SPRT算法}实现
def sprt_test(p_true, p0, p1, alpha, beta, max_samples=2000):
    n = 0  # 样本总数
    x = 0  # 不合格品数量

    for _ in range(max_samples):
        n += 1
        sample = np.random.binomial(1, p_true)
        x += sample

        lnL = log_likelihood(x, n, p0, p1)

        if lnL >= lnA:
            return False, n
        elif lnL <= lnB:
            return True, n

    return None, n

# 运行模拟，并平均化结果
def average_sprt_test(p_true, p0, p1, alpha, beta, num_simulations=5000, max_samples=2000):
    total_accept = 0
    total_samples = 0
    for _ in range(num_simulations):
        result, samples = sprt_test(p_true, p0, p1, alpha, beta, max_samples)
        total_accept += int(result is True)
        total_samples += samples

    avg_accept_rate = total_accept / num_simulations
    avg_samples = total_samples / num_simulations

    return avg_accept_rate, avg_samples

# 计算拒收和接收概率及平均样本量
accept_rate_exceed, avg_samples_exceed = average_sprt_test(p1, p0, p1, alpha, beta)
accept_rate_within, avg_samples_within = average_sprt_test(p0, p0, p1, alpha, beta)

print(f"95% 信度下拒收概率为：{1-accept_rate_exceed}, 平均样本量为: {avg_samples_exceed:.2f}")
print(f"90% 信度下接收概率为：{accept_rate_within}, 平均样本量为: {avg_samples_within:.2f}")

# 模拟{OC}和{ASN曲线}
def simulate_oc_asn(p0, p1, alpha, beta, p_range, num_simulations=5000):
    oc_curve = []
    asn_curve = []

    for p_true in p_range:
        accept_count = 0
        total_samples = 0

        for _ in range(num_simulations):
            accept, samples = sprt_test(p_true, p0, p1, alpha, beta)
            if accept is not None:
                accept_count += int(accept)
            total_samples += samples

        oc_curve.append(accept_count / num_simulations)
        asn_curve.append(total_samples / num_simulations)

    return oc_curve, asn_curve

# 定义次品率的范围
p_range = np.linspace(0.05, 0.30, 26)

# 运行模拟以获取{OC}和{ASN曲线}
oc_curve, asn_curve = simulate_oc_asn(p0, p1, alpha, beta, p_range)

# 绘制{OC曲线}
plt.figure(figsize=(10, 6))
plt.plot(p_range, oc_curve, label='OC曲线', color='blue', linewidth=2)
plt.axvline(x=p0, color='red', linestyle='--', label='p0 = 0.10')
plt.axvline(x=p1, color='red', linestyle='--', label='p1 = 0.15')
plt.title('操作特性（OC）曲线', fontsize=14)
plt.xlabel('真实次品率', fontsize=12)
plt.ylabel('接受概率', fontsize=12)
plt.xticks(np.arange(0.05, 0.31, 0.05))
plt.grid(True)
plt.legend(loc='best')
plt.show()

# 绘制{ASN曲线}
plt.figure(figsize=(10, 6))
plt.plot(p_range, asn_curve, label='ASN曲线', color='green', linewidth=2)
plt.axvline(x=p0, color='red', linestyle='--', label='p0 = 0.10')
plt.axvline(x=p1, color='red', linestyle='--', label='p1 = 0.15')
plt.title('平均样本量（ASN）曲线', fontsize=14)
plt.xlabel('真实次品率', fontsize=12)
plt.ylabel('平均样本量', fontsize=12)
plt.xticks(np.arange(0.05, 0.31, 0.05))
plt.grid(True)
plt.legend(loc='best')
plt.show()

# 可视化拒收和接收概率随{次品率}变化
reject_probabilities = [1 - simulate_oc_asn(p0, p1, alpha, beta, [p])[0][0] for p in p_range]
accept_probabilities = [simulate_oc_asn(p0, p1, alpha, beta, [p])[0][0] for p in p_range]

plt.figure(figsize=(10, 6))
plt.plot(p_range, reject_probabilities, label='95%信度下拒收概率曲线', color='#FF6600', linewidth=2)
plt.plot(p_range, accept_probabilities, label='90%信度下接收概率曲线', color='#009966', linewidth=2)
plt.axvline(x=p1, color='red', linestyle='--', label='拒收临界值 p1 = 0.15')
plt.axvline(x=p0, color='orange', linestyle='--', label='接收临界值 p0 = 0.10')

# 添加水平线标记
plt.axhline(y=1-accept_rate_exceed, color='purple', linestyle='--', label=f'拒收概率 = {1-accept_rate_exceed:.2f}')
plt.axhline(y=accept_rate_within, color='cyan', linestyle='--', label=f'接收概率 = {accept_rate_within:.2f}')

plt.title('随次品率变化的拒收和接收概率', fontsize=14)
plt.xlabel('次品率', fontsize=12)
plt.ylabel('概率', fontsize=12)
plt.grid(True)
plt.legend(loc='best')
plt.show()
