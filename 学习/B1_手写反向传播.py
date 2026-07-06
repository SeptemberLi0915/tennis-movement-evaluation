"""
B1：纯 numpy 手写一个 2 层 MLP + 反向传播
=========================================
规则：不准用 torch.autograd / loss.backward()。每一行梯度自己写。
目标：分类正手 vs 反手（单帧，34维特征）。

完成顺序：
  1. 先读懂 forward()（我已写好，当范例）
  2. 填 backward() 里的 TODO（核心，这才是你要学的）
  3. 跑 gradient_check()：用 PyTorch autograd 验证你手算的梯度对不对
  4. 跑 train()：训练并看准确率

数学参考（见聊天里的推导）：
  dZ2 = (P - Y_onehot) / N
  dW2 = A1.T @ dZ2          db2 = dZ2.sum(0)
  dA1 = dZ2 @ W2.T          dZ1 = dA1 * (Z1 > 0)
  dW1 = X.T @ dZ1           db1 = dZ1.sum(0)
"""

import numpy as np

np.random.seed(0)


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────
def relu(z):
    return np.maximum(0, z)


def softmax(z):
    # 减去每行最大值，防止 exp 溢出（数值稳定）
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(P, y):
    """P:(N,2) 概率, y:(N,) 整数标签。返回平均损失。"""
    N = len(y)
    # 取出每个样本"正确类别"的概率，加 1e-9 防 log(0)
    correct_logp = -np.log(P[np.arange(N), y] + 1e-9)
    return correct_logp.mean()


# ──────────────────────────────────────────────
# 参数初始化
# ──────────────────────────────────────────────
def init_params(in_dim=34, hidden=64, out_dim=2):
    # He 初始化（适合 ReLU）：方差 2/in_dim
    params = {
        'W1': np.random.randn(in_dim, hidden) * np.sqrt(2 / in_dim),
        'b1': np.zeros(hidden),
        'W2': np.random.randn(hidden, out_dim) * np.sqrt(2 / hidden),
        'b2': np.zeros(out_dim),
    }
    return params


# ──────────────────────────────────────────────
# 前向传播（已写好，当范例研究）
# ──────────────────────────────────────────────
def forward(X, params):
    """返回 (P, cache)。cache 存反向要用的中间量。"""
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']

    Z1 = X @ W1 + b1        # (N, H)
    A1 = relu(Z1)           # (N, H)
    Z2 = A1 @ W2 + b2       # (N, 2)
    P = softmax(Z2)         # (N, 2)

    cache = {'X': X, 'Z1': Z1, 'A1': A1, 'P': P}
    return P, cache


# ──────────────────────────────────────────────
# 反向传播（★你来填★）
# ──────────────────────────────────────────────
def backward(cache, y, params):
    """
    输入: cache(前向的中间量), y(真实标签 (N,)), params(含 W2 用于 dA1)
    输出: grads 字典 {'W1','b1','W2','b2'}，每个形状和对应参数一致
    """
    # 取出前向存下来的中间量。每个变量的形状（N=样本数, H=隐藏层大小=64）：
    #   X:(N,34)   Z1:(N,H)   A1:(N,H)   P:(N,2)
    X, Z1, A1, P = cache['X'], cache['Z1'], cache['A1'], cache['P']
    W2 = params['W2']           # (H, 2)
    N = len(y)

    # 把整数标签 y 变成 one-hot。例如 y=1 → [0,1]，y=0 → [1,0]。形状 (N, 2)
    # 后面要算 P - Y，所以 Y 必须和 P 同形状。
    Y = np.zeros_like(P)
    Y[np.arange(N), y] = 1

    # ===== 反向传播 = 从损失 L 出发，倒着用链式法则，一层一层求"每个量对 L 的影响" =====
    # 记号约定：dXxx 表示 ∂L/∂Xxx，即"损失对 Xxx 的梯度"。

    # --- 第1步：输出层 dZ2 = ∂L/∂Z2 ---------------------------------
    # 这是整个反向传播的起点。softmax + 交叉熵 合起来求导，结果出奇地简洁：
    #   ∂L/∂Z2 = P - Y   （模型预测概率 减 真实答案）
    # 直觉：预测对了(P≈Y)梯度≈0不用改；预测错得越离谱(P离Y越远)梯度越大改得越狠。
    # 除以 N：因为损失 L 用的是 N 个样本的"平均"(mean)，求导后这个 1/N 要带上。
    dZ2 = (P - Y) / N          # 形状 (N, 2)

    # --- 第2步：第二层参数梯度 -------------------------------------
    # 通用规律（记住它，CNN/Transformer 全通用）：
    #   某层权重的梯度 = (该层的输入).T @ (该层输出的梯度)
    # 第二层的输入是 A1，输出的梯度是 dZ2。
    # 形状核对：A1.T 是 (H,N)，dZ2 是 (N,2)，相乘得到 (H,2) —— 正好和 W2 同形状 ✓
    dW2 = A1.T @ dZ2           # (H, 2)
    # 偏置 b2 对每个样本都加了一次，所以它的梯度是把 N 个样本的 dZ2 沿样本维(axis=0)加起来。
    db2 = dZ2.sum(axis=0)      # (2,)

    # --- 第3步：把梯度"传回"隐藏层 --------------------------------
    # dA1：损失对 A1 的梯度。Z2 = A1 @ W2，所以梯度往回传要乘 W2 的转置。
    # 形状核对：dZ2 是 (N,2)，W2.T 是 (2,H)，相乘得到 (N,H) —— 和 A1 同形状 ✓
    dA1 = dZ2 @ W2.T           # (N, H)
    # dZ1：穿过 ReLU 往回。ReLU 前向是 max(0,Z1)：Z1>0 时原样通过(导数1)，Z1≤0 时被压成0(导数0)。
    # 所以反向时，Z1>0 的位置梯度照常通过，Z1≤0 的位置梯度被"掐断"为0。
    # 注意用 Z1 判断正负(不是 A1)，因为 ReLU 的导数是按"输入 Z1"的正负决定的。
    dZ1 = dA1 * (Z1 > 0)       # (N, H)，(Z1>0) 是布尔数组，True=1 False=0

    # --- 第4步：第一层参数梯度（同第2步的规律）--------------------
    # 第一层的输入是 X，输出的梯度是 dZ1。
    # 形状核对：X.T 是 (34,N)，dZ1 是 (N,H)，相乘得到 (34,H) —— 和 W1 同形状 ✓
    dW1 = X.T @ dZ1            # (34, H)
    db1 = dZ1.sum(axis=0)      # (H,)

    # 返回每个参数的梯度，形状必须和参数本身完全一致（外面会用 W -= lr*dW 更新）
    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}


# ──────────────────────────────────────────────
# 梯度校验：用 PyTorch autograd 对答案
# ──────────────────────────────────────────────
def gradient_check():
    """造一点假数据，比较你手写的梯度 和 PyTorch 自动求的梯度。"""
    import torch

    N, D, H, C = 8, 34, 64, 2
    X = np.random.randn(N, D)
    y = np.random.randint(0, C, size=N)
    params = init_params(D, H, C)

    # --- 你的 numpy 版 ---
    P, cache = forward(X, params)
    grads = backward(cache, y, params)

    if grads['W1'] is None:
        print("⚠️ backward() 还没填完，先把 TODO 填了再来校验。")
        return

    # --- PyTorch 版（自动求导当标准答案）---
    Xt = torch.tensor(X, requires_grad=False)
    yt = torch.tensor(y, dtype=torch.long)  # cross_entropy 要求标签是 Long(int64)
    W1 = torch.tensor(params['W1'], requires_grad=True)
    b1 = torch.tensor(params['b1'], requires_grad=True)
    W2 = torch.tensor(params['W2'], requires_grad=True)
    b2 = torch.tensor(params['b2'], requires_grad=True)

    Z1 = Xt @ W1 + b1
    A1 = torch.relu(Z1)
    Z2 = A1 @ W2 + b2
    loss = torch.nn.functional.cross_entropy(Z2, yt)
    loss.backward()

    # --- 对比 ---
    print("===== 梯度校验（越接近0越好）=====")
    for name, torch_grad in [('W1', W1.grad), ('b1', b1.grad),
                             ('W2', W2.grad), ('b2', b2.grad)]:
        diff = np.abs(grads[name] - torch_grad.numpy()).max()
        flag = "✅" if diff < 1e-6 else "❌ 差太多，检查这个梯度"
        print(f"  {name}: 最大误差 = {diff:.2e}  {flag}")


# ──────────────────────────────────────────────
# 训练（用你真实的网球数据）
# ──────────────────────────────────────────────
def load_data():
    """读关键点 CSV，髋部归一化，返回 X(N,34), y(N,)。"""
    import pandas as pd
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df_f = pd.read_csv(os.path.join(root, 'data', 'keypoints_forehand_all.csv'))
    back_path = os.path.join(root, 'keypoints', 'keypoints_backhand.csv')
    df_b = pd.read_csv(back_path)

    def normalize(df):
        df = df.copy()
        hx = (df['kp11_x'] + df['kp12_x']) / 2
        hy = (df['kp11_y'] + df['kp12_y']) / 2
        for i in range(17):
            df[f'kp{i}_x'] -= hx
            df[f'kp{i}_y'] -= hy
        return df

    cols = [f'kp{i}_{c}' for i in range(17) for c in ['x', 'y']]
    Xf = normalize(df_f)[cols].values
    Xb = normalize(df_b)[cols].values
    X = np.vstack([Xf, Xb]).astype(np.float64)
    y = np.array([1] * len(Xf) + [0] * len(Xb))  # forehand=1, backhand=0

    # 打乱
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def train(epochs=300, lr=0.05):
    X, y = load_data()
    # 简单标准化输入（每列减均值除标准差），帮助收敛
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    n_tr = int(len(X) * 0.8)
    Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]

    params = init_params()
    for ep in range(epochs):
        P, cache = forward(Xtr, params)
        loss = cross_entropy(P, ytr)
        grads = backward(cache, ytr, params)

        if grads['W1'] is None:
            print("⚠️ backward() 还没填完。")
            return

        # 梯度下降更新
        for k in params:
            params[k] -= lr * grads[k]

        if ep % 50 == 0:
            acc = (forward(Xte, params)[0].argmax(1) == yte).mean()
            print(f"  epoch {ep:3d} | loss {loss:.4f} | 测试准确率 {acc:.2%}")

    acc = (forward(Xte, params)[0].argmax(1) == yte).mean()
    print(f"最终测试准确率: {acc:.2%}")


if __name__ == '__main__':
    # 第一步：填完 backward() 后，先跑这个校验梯度
    gradient_check()

    # 第二步：梯度校验通过后，取消下面这行注释来训练
    train()
