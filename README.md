# 🧪 生产过程决策模型

> 面向制造业质量-成本协同优化的开源决策支持系统  
> 支持 **序贯概率比检验（SPRT）**、**决策树**、**多工序-多零配件泛化**、**置信区间鲁棒分析** 四大核心功能。

## 🚀 快速上手
```
bash
git clone https://github.com/yourname/production-decision-tree.git
cd production-decision-tree

# 生成问题1 OC/ASN 曲线
python Q1SPRT.py

# 生成问题2 决策树与交叉验证
python Q2Tree.py && python Q2Test.py

# 生成问题3 16 维决策树
python Q3Tree.py && python Q3Test.py

# 生成问题4 置信区间
python Q4.py
```
## 生产流程图
<img width="1314" height="599" alt="image" src="https://github.com/user-attachments/assets/5f2e4de4-2a23-467e-86d5-2c9cb6db01c4" />

## 模型效果
<img width="1093" height="854" alt="image" src="https://github.com/user-attachments/assets/fc97a994-c97b-4a39-89b2-974725f75c94" />

## 参考文献
[1] 徐浩. 决策树算法改进及其在中小企业成本控制上的应用[D]. [出版地不详]: 江西理工大学, 2019.  
[2] 方茂达. 基于动态规划方法的指数分布截尾序贯最优检验研究[D]. [出版地不详]: 贵州大学, 2023.  
[3] 闫涛, 茹乐, 杜兴民. 一种基于折线逼近的对数似然比简化算法[J]. 电子与信息学报, 2008(08):1832-1835.  
[4] 郭立力, 赵春江. 十折交叉检验的支持向量机参数优化算法[J]. 计算机工程与应用, 2009, 45(08):55-57.  
[5] 戚长松, 余忠华, 侯智, 等. 基于CART 决策树的复杂生产过程质量预测方法研究[J]. 组合机床与自动化加工技术, 2010(03):94-97+102.
