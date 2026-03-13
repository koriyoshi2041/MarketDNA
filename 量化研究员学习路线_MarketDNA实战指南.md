# 量化研究员学习路线 — 从零到实战

> 以 MarketDNA 项目为例，自底向上构建量化研究的核心知识体系。
> 每一个概念都有「为什么要学这个」的动机，和「代码在哪里」的实例。

---

## 目录

1. [第零层：心态与方法论](#第零层心态与方法论)
2. [第一层：数据基础](#第一层数据基础)
3. [第二层：收益率分布——打破正态幻觉](#第二层收益率分布打破正态幻觉)
4. [第三层：波动率建模——赚"可预测性"的钱](#第三层波动率建模赚可预测性的钱)
5. [第四层：相关性与协整——找真CP](#第四层相关性与协整找真cp)
6. [第五层：Regime检测——市场不是一成不变的](#第五层regime检测市场不是一成不变的)
7. [第六层：Random Matrix Theory——从噪声中提炼信号](#第六层random-matrix-theory从噪声中提炼信号)
8. [第七层：信号构建——把分析变成策略](#第七层信号构建把分析变成策略)
9. [第八层：回测与风险管理](#第八层回测与风险管理)
10. [第九层：工程实践](#第九层工程实践)
11. [附录：推荐资源](#附录推荐资源)

---

## 第零层：心态与方法论

### 量化研究员到底做什么？

量化研究员的核心工作是：**在数据中发现可重复的统计规律，并将其转化为可执行的交易策略**。

这需要三根支柱：
1. **统计学/数学** — 理解数据背后的分布、检验、模型
2. **金融直觉** — 知道哪些统计规律有经济学意义（不是所有统计显著的东西都能赚钱）
3. **编程能力** — 把想法高效实现，处理脏数据，做回测

### MarketDNA 为什么是个好项目？

它不是一个"玩具项目"。它覆盖了量化研究的核心 pipeline：

```
数据获取 → 统计分析 → 信号生成 → 回测 → 可视化
```

每个模块都对应一个真实的量化研究技能。做完这个项目，你就具备了：
- 独立分析一只股票的统计特性
- 判断两只股票是否适合做配对交易
- 用 GARCH 做波动率预测和仓位管理
- 用 HMM 检测市场状态
- 用 RMT 从噪声中提取真实因子

---

## 第一层：数据基础

### 动机：没有干净的数据，一切分析都是空中楼阁

量化研究 90% 的时间花在数据上。学会正确获取和清洗数据，是一切的起点。

### 核心概念

#### 1.1 复权价格（Adjusted Close）

**为什么不能用原始收盘价？**

假设某股票 100 元时进行 1:2 拆股，第二天开盘价变成 50 元。
如果用原始价格计算收益率，会得到 -50% 的"假跌幅"。
复权价格（Adjusted Close）修正了分拆、股息等公司行为的影响，反映了"真实的投资回报"。

> **代码**: `marketdna/data/fetcher.py:67-68`
> ```python
> close_col = "Adj Close" if "Adj Close" in raw.columns else "Close"
> adj_close = raw[close_col].squeeze()
> ```

#### 1.2 Log Return vs Simple Return

| | Log Return `ln(P_t/P_{t-1})` | Simple Return `(P_t-P_{t-1})/P_{t-1}` |
|---|---|---|
| **可加性** | ✅ 多期 log return = 单期之和 | ❌ 不可简单相加 |
| **对称性** | ✅ +10% 和 -10% 等距 | ❌ 涨10%再跌10% ≠ 0 |
| **统计分析** | ✅ 更接近正态分布 | 偏态更明显 |
| **实际盈亏** | ❌ 需要 `exp(r)-1` 转换 | ✅ 就是你赚/亏的比例 |

**量化研究中几乎都用 log return 做分析**，但报告给投资者时用 simple return（因为更直观）。

> **代码**: `marketdna/data/fetcher.py:70-73`
> ```python
> log_ret = np.log(adj_close / adj_close.shift(1)).dropna()
> simple_ret = adj_close.pct_change().dropna()
> ```

#### 1.3 不可变数据容器

为什么用 `frozen=True` 的 dataclass？

金融数据一旦获取，不应该被意外修改。如果你在分析过程中不小心改了原始数据，所有后续分析都是错的，而且很难 debug。`frozen=True` 强制不可变，任何修改都会报错。

> **代码**: `marketdna/data/fetcher.py:19-21`
> ```python
> @dataclass(frozen=True)
> class MarketData:
>     """不可变的市场数据容器"""
> ```

### 实践建议

- 永远检查数据的起止日期、缺失值、异常值
- 对任何新数据源，先画出来看看是否合理
- 注意时区（美股用 Eastern Time，A股用北京时间）

---

## 第二层：收益率分布——打破正态幻觉

### 动机：为什么 VaR 模型在 2008 年金融危机中集体失灵？

几乎所有经典金融模型（Black-Scholes、CAPM、VaR）都假设收益率服从正态分布。
但真实市场的日收益率分布有两个关键特征：

1. **厚尾（Fat Tails）**：极端事件（暴跌/暴涨）发生的频率远高于正态分布的预测
2. **负偏（Negative Skew）**：跌得比涨得猛（恐慌比贪婪来得快）

**SPY 的真实数据告诉我们**（Demo 输出）：
- 超额峰度 = 14.58（正态分布是 0，越大尾巴越厚）
- 超过 3σ 的天数是正态预测的 5.1 倍
- Student-t 拟合自由度 ν = 2.7（<5 意味着极厚尾）
- 最大单日跌幅 -11.59%（在正态假设下，这是几百万年一遇）

### 核心概念

#### 2.1 矩（Moments）

| 矩 | 含义 | 量化用途 |
|---|---|---|
| 1st 均值 | 平均收益 | 策略的期望回报 |
| 2nd 标准差 | 波动率 | 风险度量 |
| 3rd 偏度 | 分布不对称性 | <0 说明左尾更厚（跌得猛）|
| 4th 峰度 | 尾部厚度 | >0 说明极端事件更频繁 |

> **代码**: `marketdna/analysis/distribution.py:81-84`
> ```python
> mean_daily = float(np.mean(r))
> std_daily = float(np.std(r, ddof=1))
> skew = float(stats.skew(r))
> kurt = float(stats.kurtosis(r))  # scipy默认就是excess kurtosis
> ```

#### 2.2 正态性检验

为什么不能"看一眼觉得像正态就当正态用"？需要统计检验。

**Jarque-Bera 检验**：
- 基于偏度和峰度构造统计量
- H₀：数据来自正态分布
- p < 0.05 → 拒绝正态假设
- 优点：对大样本效力强
- 缺点：对小样本不敏感

**Shapiro-Wilk 检验**：
- 基于 order statistics
- 更适合中小样本（<5000）
- 实践中通常比 JB 更敏感

> **代码**: `marketdna/analysis/distribution.py:87-91`

#### 2.3 Student-t 分布拟合

既然不是正态，那用什么？Student-t 分布是比正态分布更好的近似：
- 当自由度 ν → ∞ 时，t 分布退化为正态分布
- ν 越小，尾部越厚
- 金融收益率典型的 ν = 3~7

SPY 的 ν = 2.7，说明它的尾部比正态分布厚得多。

> **代码**: `marketdna/analysis/distribution.py:100`
> ```python
> t_df, t_loc, t_scale = stats.t.fit(r)  # MLE 拟合
> ```

#### 2.4 QQ-Plot — 用图直观看厚尾

QQ-Plot 是对比两个分布的标准工具：
- X 轴：理论正态分布的分位数
- Y 轴：实际数据的分位数
- 如果完美正态 → 点落在对角线上
- 尾部翘起 → 厚尾（这是你在几乎所有股票数据上都会看到的）

> **代码**: `marketdna/viz/plots.py:60-77` (`plot_qq`)

### 为什么这对量化很重要？

1. **风险低估**：如果你用正态分布做 VaR，会严重低估尾部风险
2. **期权定价**：Black-Scholes 假设正态 → 实际中深度虚值期权价格远高于 BS 模型预测（vol smile）
3. **压力测试**：知道真实的尾部分布，才能做有意义的压力测试

---

## 第三层：波动率建模——赚"可预测性"的钱

### 动机：波动率是金融市场最稳健的可预测量

收益率本身几乎不可预测（如果可预测，所有人都会去套利直到消失）。
但波动率（收益率的方差）高度可预测！这叫 **波动率聚类（Volatility Clustering）**：

> "大波动后面跟着大波动，小波动后面跟着小波动"

这违反了收益率 i.i.d.（独立同分布）的假设，但它是金融市场最 robust 的统计规律之一。

### 核心概念

#### 3.1 波动率聚类检验

如何证明波动率聚类存在？
1. 计算 r²（收益率的平方，波动率的代理变量）
2. 对 r² 做自相关检验（Ljung-Box）
3. 如果 r² 有显著自相关 → 波动率有"记忆"→ 波动率聚类

**SPY 实测**：
- Ljung-Box p 值 = 0.00（极显著）
- r² 自相关 lag1 = 0.421（强聚类）

> **代码**: `marketdna/analysis/volatility.py:62-68`

#### 3.2 GARCH(1,1) 模型

GARCH = Generalized AutoRegressive Conditional Heteroskedasticity（广义自回归条件异方差）

核心方程：

```
σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}
```

| 参数 | 含义 | SPY 实测值 |
|---|---|---|
| ω (omega) | 长期均值方差 | 很小 |
| α (alpha) | 昨天冲击的影响（"news impact"）| 0.177 |
| β (beta) | 昨天波动率的惯性（"memory"）| 0.790 |
| α+β | 持续性（越接近1，记忆越长）| 0.967 |

**直觉理解**：
- α 大 → 对新消息反应剧烈
- β 大 → 波动率变化缓慢
- α+β 接近 1 → 波动率冲击需要很长时间才消退

**半衰期** = log(0.5) / log(α+β)：波动率冲击衰减到一半需要多少天。
SPY 的半衰期 ≈ 21 天（约一个月），说明一次市场恐慌的影响会持续一个月。

> **代码**: `marketdna/analysis/volatility.py:74-91`

#### 3.3 杠杆效应（Leverage Effect）

**"坏消息比好消息引起更大的波动"**

当股价下跌时，公司的债务/股权比率上升 → 杠杆增大 → 未来不确定性增加 → 波动率上升。
这解释了为什么 VIX（恐慌指数）在市场下跌时飙升。

SPY 实测：收益-未来波动率相关 = -0.129（负相关 = 下跌引起更大波动）

> **代码**: `marketdna/analysis/volatility.py:94-109`

### 为什么这对量化很重要？

1. **波动率择时（Vol Timing）**：高波时减仓、低波时加仓 → 改善 Sharpe
2. **动态 VaR**：比固定 VaR 更准确的风险度量
3. **期权交易**：波动率预测是期权定价的核心

---

## 第四层：相关性与协整——找真CP

### 动机：为什么"高相关"的两只股票不一定能做配对交易？

这是量化面试的经典问题。答案是：**相关性和协整是完全不同的概念**。

| | 相关性（Correlation）| 协整（Cointegration）|
|---|---|---|
| 测什么 | 收益率方向一致性 | 价格差是否回归均值 |
| 数学 | corr(Δlog P_A, Δlog P_B) | spread = P_A - β×P_B 平稳？ |
| 交易含义 | "今天一起涨一起跌" | "价差偏了会回来" |
| 配对交易 | ❌ 不充分 | ✅ 核心条件 |

**GLD/GDX 实例**（Demo 输出）：
- 收益率相关 = 0.766（高相关！）
- 协整 p 值 = 0.217（❌ 不协整！）
- **结论**：高相关但不协整 = "假 CP"，不适合做配对交易

### 核心概念

#### 4.1 Engle-Granger 协整检验

两步法：
1. 对 P_A 和 P_B 做 OLS 回归，得到 hedge ratio β 和残差 spread = P_A - β×P_B
2. 对 spread 做 ADF 单位根检验：如果 spread 是平稳的 → 协整

> **代码**: `marketdna/analysis/correlation.py:93-98`

#### 4.2 Spread 的均值回复半衰期

如果两只股票协整，spread 会围绕均值波动。但"多快回来"很关键：
- 半衰期 < 20 天 → 适合高频交易
- 半衰期 20-60 天 → 适合日频/周频策略
- 半衰期 > 60 天 → 回归太慢，风险大

估计方法：对 spread 拟合 AR(1) 模型 `s_t = ρ × s_{t-1} + ε`，则半衰期 = -ln(2) / ln(ρ)

> **代码**: `marketdna/analysis/correlation.py:113-126`

#### 4.3 滚动相关的稳定性

相关性不是恒定的！它会随时间变化。滚动相关的标准差告诉你这种关系有多不稳定。
如果两只股票的相关性在 0.3 ~ 0.9 之间大幅波动，做配对交易的风险就很大。

> **代码**: `marketdna/analysis/correlation.py:87-90`

---

## 第五层：Regime检测——市场不是一成不变的

### 动机：为什么同一个策略有时赚钱有时亏钱？

市场在不同的"状态（Regime）"之间切换：

| Regime | 特征 | 适合的策略 |
|---|---|---|
| 低波牛市 | 稳定上涨，波动小 | 动量/趋势跟踪 |
| 高波熊市 | 急跌，恐慌，波动大 | 对冲/做空/降仓 |
| 震荡市 | 方向不明，来回波动 | 均值回复/配对交易 |

**SPY Regime 实测**（Demo 输出）：
- Regime 0（低波平静）：年化 +35.4%，波动 11.3%
- Regime 1（中波震荡）：年化 +14.7%，波动 13.2%
- Regime 2（高波恐慌）：年化 -47.6%，波动 34.8%，平均持续 30 天

如果你能在 Regime 2 开始时减仓，在 Regime 0 开始时加仓，绩效将大幅改善。

### 核心概念

#### 5.1 Hidden Markov Model (HMM)

**核心思想**：市场有 K 个你"看不见"的隐藏状态，每个状态下收益率服从不同的正态分布。

```
隐藏状态:    S₁ → S₂ → S₂ → S₃ → S₁ → S₁ ...
                ↓      ↓      ↓      ↓      ↓      ↓
观测收益率:  r₁    r₂    r₃    r₄    r₅    r₆ ...
```

HMM 的任务：从你能观察到的收益率序列，反推每天最可能处于哪个状态。

三个核心参数：
1. **初始状态概率** π：一开始在哪个状态
2. **转移矩阵** A：从状态 i 切换到状态 j 的概率
3. **发射分布** B：每个状态下收益率的均值和方差

> **代码**: `marketdna/analysis/regime.py:84-91`
> ```python
> model = GaussianHMM(
>     n_components=n_regimes,
>     covariance_type="full",
>     n_iter=200,
>     random_state=42,
> )
> model.fit(X)
> ```

#### 5.2 转移矩阵的含义

```
           → 低波  → 震荡  → 恐慌
从低波:    95%     5%      0%     ← 低波状态很稳定
从震荡:    2%      96%     2%
从恐慌:    1%      4%      95%    ← 恐慌一旦开始很难结束
```

这告诉你：
- 低波状态有 5% 的概率切换到震荡，但几乎不会直接跳到恐慌
- 恐慌状态有 95% 的概率持续（这解释了"恐慌是有惯性的"）
- 状态切换不是对称的（从平静到恐慌比从恐慌到平静更难）

---

## 第六层：Random Matrix Theory——从噪声中提炼信号

### 动机：为什么 500 只股票的相关矩阵 80% 是噪声？

假设你有 500 只股票、3 年日度数据（~750 天）：
- 相关矩阵有 500×500 = 250,000 个参数
- 但你只有 750×500 = 375,000 个数据点

**参数和数据的数量级差不多**，意味着大部分"相关性"只是统计噪声。

如果你用这个充满噪声的相关矩阵做投资组合优化（Markowitz），噪声会被放大 → 优化结果极不稳定。

### 核心概念

#### 6.1 Marchenko-Pastur 定律

如果收益率**完全随机**（没有任何真实相关性），样本相关矩阵的特征值应该服从 Marchenko-Pastur 分布：

```
λ_max = (1 + √(N/T))²
λ_min = (1 - √(N/T))²
```

其中 N = 股票数，T = 观测天数。

**任何超过 λ_max 的特征值才是"真实信号"**，其余都是噪声。

**科技股 RMT 实测**（10 只股票 × 1551 天）：
- λ_max = 1.167（噪声上界）
- 信号特征值 = 1 个（λ₁ = 5.68，市场因子）
- 噪声特征值 = 9 个
- **43% 的相关矩阵信息是纯噪声**

> **代码**: `marketdna/analysis/rmt.py:94-101`
> ```python
> mp_max = sigma2 * (1 + np.sqrt(q)) ** 2
> signal_mask = eigenvalues > mp_max
> ```

#### 6.2 去噪方法

1. 特征值分解：C = VΛV^T
2. 把噪声特征值替换为它们的均值（保持 trace 不变）
3. 重构：C_denoised = V Λ_denoised V^T
4. 强制对角线 = 1（保持相关矩阵性质）

> **代码**: `marketdna/analysis/rmt.py:106-119`

#### 6.3 条件数（Condition Number）

条件数 = 最大特征值 / 最小特征值。
- 条件数大 → 矩阵近似奇异 → 求逆不稳定 → 优化结果垃圾
- 去噪后条件数减小 → 矩阵更稳定 → 优化更可靠

实测：去噪前 24 → 去噪后 12（改善 2 倍）

---

## 第七层：信号构建——把分析变成策略

### 动机：分析再漂亮，不能赚钱就是零

前面所有的分析最终都要变成可执行的交易信号。这一步是从"研究员"到"PM/trader"的桥梁。

### 7.1 波动率择时信号

**核心逻辑**：用 GARCH 预测明天的波动率，然后调整仓位：

```
w_t = σ_target / σ_predicted_t
```

- 预测波动率高 → 降低仓位（承担更少风险）
- 预测波动率低 → 增加仓位（同样的风险预算放更多头寸）
- 效果：组合的已实现波动率接近恒定（volatility targeting）

**SPY Vol Timing 实测**（Demo 输出）：
- 原始波动率 17.8% → 策略波动率 10.2%（接近目标 10%）
- Sharpe：0.693 → 0.742（+0.049）
- 最大回撤：-35.7% → -14.7%（改善 21%！）

> **代码**: `marketdna/signals/vol_timing.py:82-88`
> ```python
> target_daily = target_vol / np.sqrt(252)
> weights = target_daily / cond_vol_daily
> weights = weights.clip(upper=max_leverage)
> # 策略收益 = 昨天的权重 × 今天的收益（避免前瞻偏差）
> strategy_ret = (weights.shift(1) * r).dropna()
> ```

**关键细节**：`weights.shift(1)` — 用昨天的权重乘以今天的收益。
如果用今天的权重乘以今天的收益 → **前瞻偏差（Look-Ahead Bias）**，回测赚钱但实盘做不到。
这是量化新手最常犯的错误之一。

### 7.2 均值回复配对交易信号

**核心逻辑**：找到协整的两只股票，当价差偏离均值过远时下注价差会回来。

```
z_t = (spread_t - mean) / std

z > +2.0  → 做空 spread（做空A、做多B）
z < -2.0  → 做多 spread（做多A、做空B）
|z| < 0.5 → 平仓
|z| > 4.0 → 止损
```

> **代码**: `marketdna/signals/mean_reversion.py:104-121`

**止损为什么重要？**
协整关系可能破裂。如果两只股票的经济关系发生了根本变化（例如一家被收购），
价差可能永远不会回来。止损 z = 4.0 就是在说："如果偏离到这种程度，可能不是暂时的错误定价了。"

---

## 第八层：回测与风险管理

### 动机：回测是验证策略的唯一科学方法

"回测里赚钱 ≠ 实盘会赚钱"，但"回测都亏钱 → 实盘基本也亏钱"。

### 核心概念

#### 8.1 Sharpe Ratio

```
Sharpe = (年化收益 - 无风险利率) / 年化波动率
```

| Sharpe | 含义 |
|---|---|
| < 0.5 | 差，不值得交易 |
| 0.5-1.0 | 一般，需要结合其他指标 |
| 1.0-2.0 | 好策略 |
| > 2.0 | 非常优秀或者你的回测有 bug |

> **代码**: `marketdna/signals/vol_timing.py:98-99`
> ```python
> strat_sharpe = strat_mean / strat_annual_vol if strat_annual_vol > 0 else 0
> ```

#### 8.2 最大回撤（Max Drawdown）

从历史最高点到最低点的最大跌幅。它衡量的是"最糟糕的时候你会亏多少"。

```python
peak = cumulative.cummax()
drawdown = (cumulative - peak) / peak
max_dd = drawdown.min()
```

**心理学意义**：投资者更在乎"最多亏多少"而非"平均赚多少"。
一个 Sharpe 1.5 但 max drawdown -60% 的策略，很少有人能坚持执行。

> **代码**: `marketdna/signals/vol_timing.py:46-50`

#### 8.3 前瞻偏差（Look-Ahead Bias）

回测中最危险的错误：用了"未来才知道"的信息做决策。

常见形式：
- 用今天的数据做信号，今天就交易（应该用昨天的信号）
- 用全样本估计参数，然后在全样本上回测（应该用走出样本）
- 复权价格用了后续的分红信息

MarketDNA 中如何避免：
- `weights.shift(1) * r` → 用昨天的权重
- `positions.shift(1) * spread_ret` → 用昨天的仓位

---

## 第九层：工程实践

### 9.1 项目结构

```
marketdna/
├── data/           # 数据层：获取和清洗
│   └── fetcher.py
├── analysis/       # 分析层：统计指纹提取
│   ├── distribution.py
│   ├── volatility.py
│   ├── correlation.py
│   ├── regime.py
│   └── rmt.py
├── signals/        # 信号层：把分析转为交易信号
│   ├── vol_timing.py
│   └── mean_reversion.py
├── viz/            # 可视化层
│   └── plots.py
├── tests/          # 测试
│   └── test_core.py
└── scan.py         # 主入口
```

这种分层结构（data → analysis → signals → viz）是量化系统的标准架构。

### 9.2 测试的重要性

金融代码中的 bug 代价极高。一个符号错误可能意味着"做多"变"做空"。
合成数据测试（synthetic data tests）是量化中的标准做法：
- 生成已知特性的数据
- 验证分析代码能否恢复这些特性
- 不依赖网络，运行速度快

> **代码**: `marketdna/tests/test_core.py`
> 使用 Student-t(4) 合成数据测试分布分析，用构造的协整对测试相关性分析。

### 9.3 数据类型选择

| 类型 | 用途 | 项目示例 |
|---|---|---|
| `frozen dataclass` | 不可变结果容器 | 所有 Fingerprint 类 |
| `pd.Series` | 带时间索引的一维数据 | 收益率、仓位 |
| `pd.DataFrame` | 多列数据 | OHLCV 价格 |
| `np.ndarray` | 矩阵运算 | 相关矩阵、特征向量 |

---

## 附录：推荐资源

### 必读书

| 书名 | 为什么 | 对应 MarketDNA 模块 |
|---|---|---|
| *Advances in Financial Machine Learning* (Marcos López de Prado) | 现代量化的圣经，涵盖 RMT、特征重要性、策略评估 | RMT, 分布分析 |
| *Quantitative Trading* (Ernest Chan) | 入门配对交易和均值回复 | 配对交易信号 |
| *Analysis of Financial Time Series* (Ruey Tsay) | 金融时间序列分析教科书 | GARCH, 波动率 |
| *Options, Futures, and Other Derivatives* (John Hull) | 衍生品定价基础 | 波动率概念 |

### 核心 Python 库

| 库 | 用途 | MarketDNA 中的使用 |
|---|---|---|
| `numpy` | 数值计算 | 到处都是 |
| `pandas` | 时间序列处理 | 数据对齐、滚动窗口 |
| `scipy.stats` | 统计检验和分布拟合 | JB/Shapiro 检验、Student-t 拟合 |
| `statsmodels` | 计量经济学 | Ljung-Box、ADF、协整检验 |
| `arch` | GARCH 建模 | 波动率预测 |
| `hmmlearn` | Hidden Markov Model | Regime 检测 |
| `matplotlib` | 可视化 | 所有图表 |
| `yfinance` | 数据获取 | 市场数据 |

### 关键概念速查

| 概念 | 一句话解释 | MarketDNA 位置 |
|---|---|---|
| Log return | ln(P_t/P_{t-1})，可加性好 | fetcher.py |
| Excess kurtosis | >0 = 厚尾 | distribution.py |
| Jarque-Bera | 检验是否正态 | distribution.py |
| GARCH(1,1) | σ² = ω + αε² + βσ² | volatility.py |
| Vol clustering | 大波动接大波动 | volatility.py |
| Leverage effect | 跌比涨引起更大波动 | volatility.py |
| Cointegration | 价差平稳（可回归） | correlation.py |
| Spread half-life | 价差回归到均值的速度 | correlation.py |
| HMM | 隐藏状态 + 观测序列 | regime.py |
| Marchenko-Pastur | 噪声特征值的理论分布 | rmt.py |
| Vol targeting | w = σ_target / σ_pred | vol_timing.py |
| Z-score | (x - μ) / σ | mean_reversion.py |
| Look-ahead bias | 用未来数据做决策 | 所有 signal |
| Max drawdown | 历史最大回撤 | 绩效评估 |

---

## 运行项目

```bash
cd quantresearch
source .venv/bin/activate

# 运行完整 Demo
python run_demo.py

# 运行测试
python -m pytest marketdna/tests/test_core.py -v

# 在 Python 中使用
python -c "from marketdna.scan import scan; scan('AAPL')"

# 深度扫描（含 Regime + Vol Timing）
python -c "from marketdna.scan import scan_deep; scan_deep('SPY')"
```

---

> **记住**：量化研究不是关于找到"圣杯策略"。
> 它是关于理解市场的统计特性，在可预测的地方下注，
> 并严格管理那些不可预测的风险。
>
> MarketDNA 教你的不是一个赚钱的策略，
> 而是分析任何策略所需要的统计工具箱。
