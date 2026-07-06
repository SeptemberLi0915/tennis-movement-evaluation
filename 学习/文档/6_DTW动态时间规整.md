# DTW 动态时间规整（Dynamic Time Warping）

## 为什么需要 DTW？

欧氏距离要求两个序列**等长且对齐**——但正手动作的速度、节奏因人而异。
同一个人打快了 0.5 秒和慢了 0.5 秒，动作轨迹几乎一样，欧氏距离却会很大。

DTW 允许序列在时间轴上**弹性伸缩**，找到最优对齐方式，再计算距离。

```
标准:  ──/──\──       DTW 可以把"快版"的 / 拉长，
你的:  ─/\─           让两条曲线对齐后再比较
```

## 递推公式

给定序列 `s1[1..n]` 和 `s2[1..m]`，构造代价矩阵 `D[i,j]`：

```
D[0, 0] = 0
D[i, 0] = D[0, j] = ∞   (边界)

D[i, j] = cost(s1[i], s2[j]) + min(
    D[i-1, j],     ← 插入（s1 拉伸）
    D[i, j-1],     ← 删除（s2 拉伸）
    D[i-1, j-1]    ← 匹配
)
```

其中 `cost(a, b) = ‖a - b‖`（欧氏距离）。

最终 DTW 距离 = `D[n, m]`。

## 复杂度

- **时间**：O(n × m)  
- **空间**：O(n × m)，可优化为 O(m)（只保留两行）

对于本项目：n = m = 40（重采样后），所以 40×40 = 1600 次运算，非常快。

## 项目中的用法

### 1. 重采样

正手段落长度不一（有的 50 帧，有的 80 帧），先统一重采样到 40 帧：

```python
def resample(seq, n=40):
    x_old = np.linspace(0, 1, len(seq))
    x_new = np.linspace(0, 1, n)
    return np.column_stack([np.interp(x_new, x_old, seq[:, c]) for c in range(2)])
```

### 2. DTW 计算

```python
def dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
    d = np.full((n+1, m+1), np.inf)
    d[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.linalg.norm(s1[i-1] - s2[j-1])
            d[i, j] = cost + min(d[i-1, j], d[i, j-1], d[i-1, j-1])
    return d[n, m]
```

每个元素是 2D 向量 `[肘角度, 肩角度]`，所以 cost 是二维欧氏距离。

### 3. DTW → 评分

```python
DTW_BEST = 500      # 完美动作的 DTW 距离
DTW_WORST = 2500     # 最差动作的 DTW 距离

def dtw_to_score(dtw_dist):
    ratio = clamp((DTW_WORST - dtw_dist) / (DTW_WORST - DTW_BEST), 0, 1)
    return int(100 * ratio ** 0.5)   # √曲线，对中间范围更宽容
```

为什么用 √ 曲线？线性映射会让中等水平的人分数太低（50→60），√ 曲线把中间段拉高，评分更符合直觉。

```
线性:  DTW=1500 → 50分
√曲线: DTW=1500 → 71分  ← 更合理
```

## FastDTW 加速（可选扩展）

标准 DTW 是 O(nm)。对于长序列（上千帧），可以用 FastDTW：

- **思路**：先在低分辨率上算粗略路径，再逐步细化
- **复杂度**：O(n)，线性近似
- **Python 库**：`pip install fastdtw`

本项目重采样到 40 帧后不需要加速，但如果将来做全视频级别的 DTW 搜索会用到。

## 关键理解

| 概念 | 说明 |
|------|------|
| **对齐路径** | D 矩阵中从 (0,0) 到 (n,m) 的最短路径 |
| **弹性伸缩** | 同一帧可以匹配对方的多帧（拉伸）|
| **全局约束** | Sakoe-Chiba Band 限制伸缩幅度，防止过度变形 |
| **归一化** | 除以路径长度可消除序列长度影响（本项目已重采样，不需要）|

## 思考题

1. 如果两个序列完全相同，DTW 距离是多少？（0）
2. DTW 距离满足三角不等式吗？（满足，可做度量）
3. 把 √ 曲线换成 log 曲线会怎样？（低分段更敏感，高分段更扁平）
