"""
滑动窗口边界检测：自动找出每次正手的"起止帧"
================================================
核心思路（Temporal Action Detection）：
  1. 用单帧正手检测器，对整段视频逐帧输出"正手概率"
  2. 移动平均平滑概率曲线（去抖）
  3. 阈值穿越找边界：概率上穿=动作开始，下穿=动作结束
  4. 过滤太短的段（误检）

验证：在 keypoints.csv（视频1）上跑，和已知的真实击球时间对答案。
输出：一张概率曲线图，标出检测到的正手段 + 真实击球点。
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KP_COLS = [f'kp{i}_{c}' for i in range(17) for c in ['x', 'y']]
FPS = 29.15

# 检测超参（可调）
SMOOTH_K = 7        # 平滑窗口（帧）
PEAK_THRESH = 0.55  # 峰值要超过这个才算一次击球
EXPAND_LOW = 0.2    # 从峰向两边扩展，概率掉到这个值以下就停（动作边界）
MIN_GAP = 30        # 两次击球峰之间最小间距（帧），约1秒，去重
MAX_HALF = 35       # 单侧最多扩展帧数，防止粘连

# 视频1的真实正手击球时间（秒），来自 archive/tennis_classifier.py 的 labels_train
TRUE_FOREHAND_SEC = [1, 5, 8, 11, 15, 18, 21, 33, 44, 47, 50, 52, 55]


# ── 模型（和训练时同款）──────────────────────────────
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


def normalize_hip(df):
    df = df.copy()
    hx = (df['kp11_x'] + df['kp12_x']) / 2
    hy = (df['kp11_y'] + df['kp12_y']) / 2
    for i in range(17):
        df[f'kp{i}_x'] -= hx
        df[f'kp{i}_y'] -= hy
    return df


def moving_average(x, k):
    return np.convolve(x, np.ones(k) / k, mode='same')


def detect_segments(probs):
    """
    峰值检测 + 向两边扩展，找出 [(start, peak, end), ...]
    1. 找所有超过 PEAK_THRESH 的局部峰，按概率从高到低，用 MIN_GAP 去重
    2. 每个峰向左右扩展，直到概率掉到 EXPAND_LOW 以下（或超过 MAX_HALF）
    """
    n = len(probs)
    # 1. 候选峰：局部最大值且超阈值
    cand = [i for i in range(1, n - 1)
            if probs[i] > PEAK_THRESH and probs[i] >= probs[i-1] and probs[i] >= probs[i+1]]
    # 按概率从高到低排序，贪心去重（保证两峰间距 >= MIN_GAP）
    cand.sort(key=lambda i: -probs[i])
    peaks = []
    for i in cand:
        if all(abs(i - p) >= MIN_GAP for p in peaks):
            peaks.append(i)
    peaks.sort()

    # 2. 每个峰向两边扩展到动作边界
    segments = []
    for pk in peaks:
        s = pk
        while s > 0 and probs[s] > EXPAND_LOW and pk - s < MAX_HALF:
            s -= 1
        e = pk
        while e < n - 1 and probs[e] > EXPAND_LOW and e - pk < MAX_HALF:
            e += 1
        segments.append((s, pk, e))
    return segments


def main():
    # 1. 加载模型 + 标准化参数
    model = ForehandDetector()
    model.load_state_dict(torch.load(os.path.join(ROOT, 'forehand_detector.pth'),
                                     map_location='cpu'))
    model.eval()
    norm = np.load(os.path.join(ROOT, 'forehand_detector_norm.npz'))
    mean, std = norm['mean'], norm['std']

    # 2. 逐帧出概率
    df = normalize_hip(pd.read_csv(os.path.join(ROOT, 'keypoints', 'keypoints.csv')))
    X = df[KP_COLS].values.astype(np.float32)
    X = (X - mean) / std
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(X)), dim=1)[:, 1].numpy()

    # 3. 平滑
    probs_smooth = moving_average(probs, SMOOTH_K)

    # 4. 找边界
    segments = detect_segments(probs_smooth)

    print(f"检测到 {len(segments)} 段正手（真实有 {len(TRUE_FOREHAND_SEC)} 次）")
    print("\n检测到的正手段：")
    for rank, (s, pk, e) in enumerate(segments, 1):
        print(f"  第{rank:2d}次: 帧 {s:4d}~{e:4d} (峰{pk})  ≈ 击球{pk/FPS:4.1f}秒  时长 {(e-s)/FPS:.2f}秒")

    # 5. 画图：概率曲线 + 检测段 + 真实击球点
    plt.figure(figsize=(15, 4))
    plt.plot(probs, color='lightgray', lw=0.8, label='原始正手概率')
    plt.plot(probs_smooth, color='blue', lw=1.5, label='平滑后')
    plt.axhline(PEAK_THRESH, color='green', ls='--', lw=1, label=f'峰值阈值 {PEAK_THRESH}')
    plt.axhline(EXPAND_LOW, color='orange', ls='--', lw=1, label=f'边界阈值 {EXPAND_LOW}')
    for s, pk, e in segments:
        plt.axvspan(s, e, color='blue', alpha=0.15)
        plt.plot(pk, probs_smooth[pk], 'b^', ms=6)
    for sec in TRUE_FOREHAND_SEC:
        plt.axvline(sec * FPS, color='red', ls=':', lw=1.2)
    plt.plot([], [], color='red', ls=':', label='真实正手(标注)')
    plt.xlabel('帧'); plt.ylabel('正手概率')
    plt.title('滑动窗口正手检测 —— 蓝色区间=检测到的正手段, 红虚线=真实击球')
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    out = os.path.join(ROOT, '学习', '滑窗检测结果.png')
    plt.savefig(out, dpi=130)
    print(f"\n图已保存: {out}")


if __name__ == '__main__':
    main()
