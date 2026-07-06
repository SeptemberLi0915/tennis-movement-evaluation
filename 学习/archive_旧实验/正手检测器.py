"""
正手检测器（二分类：正手 vs 非正手）
=====================================
目标：单帧判断"这一帧球员是不是在打正手"。
  正手(1)   = keypoints_forehand_all.csv
  非正手(0) = 反手 + idle 合并

为什么单帧：后面"滑动窗口检测击球边界"需要逐帧的正手概率，单帧分类器最直接。

⚠️ 已知隐患：正手数据来自17个视频的所有帧，混入了走位/等球等"伪正手"帧（标签噪声）。
   先训一版看效果，不行再清洗。

输出：forehand_detector.pth (模型) + forehand_detector_norm.npz (标准化参数，推理时要用)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

torch.manual_seed(0)
np.random.seed(0)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KP_COLS = [f'kp{i}_{c}' for i in range(17) for c in ['x', 'y']]
N_EACH = 2000   # 每类采样数（平衡），解决正手8万、非正手2千的不平衡


# ──────────────────────────────────────────────
# 数据
# ──────────────────────────────────────────────
def normalize_hip(df):
    """髋部中心归一化：消除球员在画面中的位置/远近影响"""
    df = df.copy()
    hx = (df['kp11_x'] + df['kp12_x']) / 2
    hy = (df['kp11_y'] + df['kp12_y']) / 2
    for i in range(17):
        df[f'kp{i}_x'] -= hx
        df[f'kp{i}_y'] -= hy
    return df


def load_xy():
    fore = pd.read_csv(os.path.join(ROOT, 'keypoints_forehand_all.csv'))
    back = pd.read_csv(os.path.join(ROOT, 'keypoints', 'keypoints_backhand.csv'))
    idle = pd.read_csv(os.path.join(ROOT, 'keypoints_idle_all.csv'))

    fore = normalize_hip(fore)[KP_COLS].values
    back = normalize_hip(back)[KP_COLS].values
    idle = normalize_hip(idle)[KP_COLS].values

    non_fore = np.vstack([back, idle])   # 非正手 = 反手 + idle

    # 平衡采样：每类取 N_EACH 个
    def sample(arr, n):
        idx = np.random.choice(len(arr), size=min(n, len(arr)), replace=False)
        return arr[idx]

    Xf = sample(fore, N_EACH)
    Xn = sample(non_fore, N_EACH)

    X = np.vstack([Xf, Xn]).astype(np.float32)
    y = np.array([1] * len(Xf) + [0] * len(Xn), dtype=np.int64)
    print(f"正手样本 {len(Xf)} | 非正手样本 {len(Xn)}（反手{len(back)}+idle{len(idle)}）")
    return X, y


# ──────────────────────────────────────────────
# 模型（单帧 MLP，和你 B1 学的同款）
# ──────────────────────────────────────────────
class ForehandDetector(nn.Module):
    def __init__(self, in_dim=34, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────
# 训练
# ──────────────────────────────────────────────
def train(epochs=300, lr=1e-3):
    X, y = load_xy()

    # 标准化输入（每列减均值除标准差），并保存参数供推理用
    mean, std = X.mean(0), X.std(0) + 1e-8
    X = (X - mean) / std

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    Xtr = torch.tensor(Xtr); ytr = torch.tensor(ytr)
    Xte = torch.tensor(Xte); yte = torch.tensor(yte)

    model = ForehandDetector()
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    print("\n开始训练...")
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        total = 0
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]
            pred = model(Xtr[idx])
            loss = loss_fn(pred, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if ep % 50 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(Xte).argmax(1) == yte).float().mean()
            print(f"  epoch {ep:3d} | loss {total:.3f} | 测试准确率 {acc:.2%}")

    # 最终报告
    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(1)
    print("\n===== 最终测试结果 =====")
    print(classification_report(yte.numpy(), pred.numpy(),
                                target_names=['非正手', '正手']))

    # 保存
    torch.save(model.state_dict(), os.path.join(ROOT, 'forehand_detector.pth'))
    np.savez(os.path.join(ROOT, 'forehand_detector_norm.npz'), mean=mean, std=std)
    print("已保存: forehand_detector.pth + forehand_detector_norm.npz")


if __name__ == '__main__':
    train()
