"""
滑动窗口边界检测 —— LSTM 版
============================
用 LSTM 序列检测器逐帧出"正手概率"，再峰值检测找每次正手的起止。
对比单帧版：LSTM 看连续30帧轨迹，概率应更平滑、漏检/误检更少。

在 keypoints.csv（视频1，有真实标注）上验证。
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
SEQ = 30

# 检测超参
SMOOTH_K = 5
PEAK_THRESH = 0.55
EXPAND_LOW = 0.2
MIN_GAP = 30
MAX_HALF = 35

TRUE_FOREHAND_SEC = [1, 5, 8, 11, 15, 18, 21, 33, 44, 47, 50, 52, 55]


class ForehandLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=34, hidden_size=64,
                            num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


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
    n = len(probs)
    cand = [i for i in range(1, n - 1)
            if probs[i] > PEAK_THRESH and probs[i] >= probs[i-1] and probs[i] >= probs[i+1]]
    cand.sort(key=lambda i: -probs[i])
    peaks = []
    for i in cand:
        if all(abs(i - p) >= MIN_GAP for p in peaks):
            peaks.append(i)
    peaks.sort()
    raw = []
    for pk in peaks:
        s = pk
        while s > 0 and probs[s] > EXPAND_LOW and pk - s < MAX_HALF:
            s -= 1
        e = pk
        while e < n - 1 and probs[e] > EXPAND_LOW and e - pk < MAX_HALF:
            e += 1
        raw.append((s, pk, e))

    # 合并重叠的段（同一次正手被切成多段时合并，保留概率最高的峰）
    raw.sort(key=lambda x: x[0])
    segments = []
    for s, pk, e in raw:
        if segments and s <= segments[-1][2]:   # 和上一段重叠
            ps, ppk, pe = segments[-1]
            best_pk = pk if probs[pk] > probs[ppk] else ppk
            segments[-1] = (min(ps, s), best_pk, max(pe, e))
        else:
            segments.append((s, pk, e))
    return segments


def main():
    model = ForehandLSTM()
    model.load_state_dict(torch.load(os.path.join(ROOT, 'models', 'forehand_lstm.pth'),
                                     map_location='cpu'))
    model.eval()
    norm = np.load(os.path.join(ROOT, 'models', 'forehand_lstm_norm.npz'))
    mean, std = norm['mean'], norm['std']

    df = normalize_hip(pd.read_csv(os.path.join(ROOT, 'keypoints', 'keypoints.csv')))
    feats = df[KP_COLS].values.astype(np.float32)
    feats = (feats - mean) / std
    n = len(feats)

    # 逐帧概率：每个位置取它结尾的30帧窗口，prob 赋给窗口中心帧
    probs = np.zeros(n)
    windows, centers = [], []
    for i in range(0, n - SEQ + 1):
        windows.append(feats[i:i + SEQ])
        centers.append(i + SEQ // 2)
    windows = torch.tensor(np.array(windows, dtype=np.float32))
    with torch.no_grad():
        # 分批跑，防止显存/内存峰值
        out = []
        for b in range(0, len(windows), 512):
            out.append(torch.softmax(model(windows[b:b+512]), dim=1)[:, 1])
        p = torch.cat(out).numpy()
    for c, pv in zip(centers, p):
        probs[c] = pv

    probs_smooth = moving_average(probs, SMOOTH_K)
    segments = detect_segments(probs_smooth)

    print(f"检测到 {len(segments)} 段正手（真实有 {len(TRUE_FOREHAND_SEC)} 次）\n")
    for rank, (s, pk, e) in enumerate(segments, 1):
        print(f"  第{rank:2d}次: 帧 {s:4d}~{e:4d} (峰{pk})  ≈ 击球{pk/FPS:4.1f}秒  时长 {(e-s)/FPS:.2f}秒")

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
    plt.title('LSTM 正手检测 —— 蓝区间=检测段, 红虚线=真实击球')
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(ROOT, '学习', '输出图', '滑窗检测结果_LSTM.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=130)
    print(f"\n图已保存: {out_path}")


if __name__ == '__main__':
    main()
