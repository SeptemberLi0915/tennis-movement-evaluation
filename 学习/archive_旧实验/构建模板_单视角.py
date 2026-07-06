"""
单视角正手模板（DBA：DTW对齐后平均）
======================================
教训：forehand_angles.npy 来自17个不同视角的视频，跨视角平均角度=直线(无效)。
修正：只在【单个视频(单视角)】内部，把多次正手用 DTW 对齐后平均。

步骤：
  1. forehand_all.csv 按帧号断成多段(每段=一个视频)
  2. 每段跑检测器数正手，挑正手最多的那段
  3. 段内以击球峰为锚截取每次正手
  4. (a)普通平均 vs (b)DTW对齐平均，对比看谁能还原挥拍弧线
输出：forehand_template_sv.npy + 对比图
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
SEQ = 30
HALF = 25            # 击球峰前后各25帧 → 每段50帧
PEAK_THRESH = 0.7
MIN_GAP = 30


class ForehandLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=34, hidden_size=64,
                            num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def angle_3pts(A, B, C):
    BA = A - B; BC = C - B
    cos = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def normalize_hip_arr(row):
    xs = row.reshape(17, 2)
    hip = (xs[11] + xs[12]) / 2
    return (xs - hip).reshape(-1)


def runs_of(frames):
    runs, start = [], 0
    for i in range(1, len(frames)):
        if frames[i] != frames[i-1] + 1:
            runs.append((start, i)); start = i
    runs.append((start, len(frames)))
    return runs


def find_peaks(prob):
    cand = [i for i in range(1, len(prob)-1)
            if prob[i] > PEAK_THRESH and prob[i] >= prob[i-1] and prob[i] >= prob[i+1]]
    cand.sort(key=lambda i: -prob[i])
    peaks = []
    for i in cand:
        if all(abs(i-p) >= MIN_GAP for p in peaks):
            peaks.append(i)
    return sorted(peaks)


def dtw_path(s1, s2):
    """返回把 s2 对齐到 s1 的规整路径 (i,j) 列表"""
    n, m = len(s1), len(s2)
    D = np.full((n+1, m+1), np.inf); D[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.linalg.norm(s1[i-1] - s2[j-1])
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    # 回溯
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        step = np.argmin([D[i-1, j-1], D[i-1, j], D[i, j-1]])
        if step == 0: i, j = i-1, j-1
        elif step == 1: i -= 1
        else: j -= 1
    return path[::-1]


def dba_average(samples, ref):
    """把每个样本 DTW 对齐到 ref 的时间轴上，再平均（一轮 DBA）"""
    L = len(ref)
    acc = np.zeros((L, 2)); cnt = np.zeros((L, 1))
    for s in samples:
        for (i, j) in dtw_path(ref, s):
            acc[i] += s[j]; cnt[i] += 1
    return acc / np.maximum(cnt, 1)


def main():
    model = ForehandLSTM()
    model.load_state_dict(torch.load(os.path.join(ROOT, 'forehand_lstm.pth'), map_location='cpu'))
    model.eval()
    norm = np.load(os.path.join(ROOT, 'forehand_lstm_norm.npz'))
    mean, std = norm['mean'], norm['std']

    df = pd.read_csv(os.path.join(ROOT, 'keypoints_forehand_all.csv'))
    raw = df[KP_COLS].values.astype(np.float32)
    frames = df['frame'].values
    n = len(raw)

    angles = np.full((n, 2), np.nan)
    for f in range(n):
        xs = raw[f].reshape(17, 2)
        rs, re, rw, rh = xs[6], xs[8], xs[10], xs[12]
        if np.linalg.norm(rs-re) > 15 and np.linalg.norm(re-rw) > 15:
            e = angle_3pts(rs, re, rw); s = angle_3pts(rh, rs, re)
            if 30 < e < 175 and 10 < s < 160:
                angles[f] = [e, s]
    feats = ((np.array([normalize_hip_arr(raw[f]) for f in range(n)], dtype=np.float32)) - mean) / std

    # 每段(视频)找正手，统计
    run_strokes = []   # (run_idx, [stroke_angle_arrays])
    runs = runs_of(frames)
    for ri, (a, b) in enumerate(runs):
        run = feats[a:b]
        if len(run) < SEQ + 2*HALF:
            run_strokes.append((ri, [])); continue
        wins = np.array([run[i:i+SEQ] for i in range(len(run)-SEQ+1)], dtype=np.float32)
        with torch.no_grad():
            probs = torch.softmax(model(torch.tensor(wins)), dim=1)[:, 1].numpy()
        prob_full = np.zeros(len(run))
        for i, pv in enumerate(probs):
            prob_full[i + SEQ//2] = pv
        strokes = []
        for pk in find_peaks(prob_full):
            g = a + pk
            if g-HALF < a or g+HALF >= b:
                continue
            seg = angles[g-HALF:g+HALF]
            if not np.isnan(seg).any():
                strokes.append(seg)
        run_strokes.append((ri, strokes))

    # 挑正手最多的视频
    run_strokes.sort(key=lambda x: -len(x[1]))
    best_ri, samples = run_strokes[0]
    a, b = runs[best_ri]
    print(f"各视频正手数: {[len(s) for _, s in sorted(run_strokes, key=lambda x:x[0])]}")
    print(f"选中视频#{best_ri} (帧{frames[a]}~{frames[b-1]}), 含 {len(samples)} 次干净正手")
    if len(samples) < 3:
        print("该视频正手太少，模板不可靠"); return

    samples = np.array(samples)                 # (K, 50, 2)
    plain = samples.mean(axis=0)                # 普通平均
    ref = samples[len(samples)//2]              # 取中间一段当对齐参考
    dba = dba_average(samples, ref)             # DTW对齐平均

    np.save(os.path.join(ROOT, 'forehand_template_sv.npy'), dba)
    print(f"模板已保存 forehand_template_sv.npy  肘{dba[:,0].mean():.1f}° 肩{dba[:,1].mean():.1f}°")

    # 对比图
    x = np.linspace(0, 1, 2*HALF)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    fig.suptitle(f'单视角(视频#{best_ri}, {len(samples)}段)  普通平均 vs DTW对齐平均', fontsize=12)
    for ax, c, name in [(axes[0], 0, '肘关节'), (axes[1], 1, '肩关节')]:
        for s in samples:
            ax.plot(x, s[:, c], color='gray', alpha=0.25)
        ax.plot(x, plain[:, c], 'b--', lw=2, label='普通平均')
        ax.plot(x, dba[:, c], 'r-', lw=2.5, label='DTW对齐平均(DBA)')
        ax.axvline(0.5, color='green', ls=':', lw=1, label='击球锚点')
        ax.set_title(name); ax.set_ylabel('角度(°)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    axes[1].set_xlabel('动作进程')
    plt.tight_layout()
    out = os.path.join(ROOT, '学习', '模板对比_单视角.png')
    plt.savefig(out, dpi=120); plt.close()
    print(f"对比图已保存 {out}")


if __name__ == '__main__':
    main()
