"""
构建高质量正手模板（多段平均 + 相位对齐）
==========================================
问题：旧模板是手取的单段40帧，带噪声、相位还对不齐。
方法：
  1. 在职业正手数据(forehand_all.csv)上跑 LSTM 检测器，找出每次正手的"击球峰值"
  2. 以击球峰值为锚点，截 [峰-HALF, 峰+HALF] 的角度序列 → 所有样本相位对齐
  3. 对齐后的多段取平均 → 平滑的"典型正手"模板，同时记录标准差(可接受范围)

输出：forehand_template.npy (HALF*2, 2)  + 对比图
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
HALF = 20             # 击球峰前后各取20帧 → 模板长度40
PEAK_THRESH = 0.7     # 模板要干净，峰值阈值取高一点
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


def contiguous_runs(frames):
    """返回连续帧段的 (start,end) 索引区间列表"""
    runs = []
    start = 0
    for i in range(1, len(frames)):
        if frames[i] != frames[i-1] + 1:
            runs.append((start, i)); start = i
    runs.append((start, len(frames)))
    return runs


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

    # 每帧角度
    angles = np.full((n, 2), np.nan)
    for f in range(n):
        xs = raw[f].reshape(17, 2)
        rs, re, rw, rh = xs[6], xs[8], xs[10], xs[12]
        if np.linalg.norm(rs-re) > 15 and np.linalg.norm(re-rw) > 15:
            e = angle_3pts(rs, re, rw); s = angle_3pts(rh, rs, re)
            if 30 < e < 175 and 10 < s < 160:
                angles[f] = [e, s]

    # 归一化特征
    feats = ((np.array([normalize_hip_arr(raw[f]) for f in range(n)], dtype=np.float32)) - mean) / std

    # 逐视频(连续段)跑检测，找击球峰
    samples = []
    for (rs_i, re_i) in contiguous_runs(frames):
        run = feats[rs_i:re_i]
        if len(run) < SEQ:
            continue
        wins = np.array([run[i:i+SEQ] for i in range(len(run)-SEQ+1)], dtype=np.float32)
        with torch.no_grad():
            probs = torch.softmax(model(torch.tensor(wins)), dim=1)[:, 1].numpy()
        # 概率赋到窗口中心(全局索引)
        prob_full = np.zeros(len(run))
        for i, pv in enumerate(probs):
            prob_full[i + SEQ//2] = pv
        # 找峰
        cand = [i for i in range(1, len(prob_full)-1)
                if prob_full[i] > PEAK_THRESH and prob_full[i] >= prob_full[i-1] and prob_full[i] >= prob_full[i+1]]
        cand.sort(key=lambda i: -prob_full[i])
        peaks = []
        for i in cand:
            if all(abs(i-p) >= MIN_GAP for p in peaks):
                peaks.append(i)
        # 以峰为中心截角度
        for pk in peaks:
            g = rs_i + pk      # 全局帧索引
            if g-HALF < rs_i or g+HALF >= re_i:
                continue
            seg = angles[g-HALF:g+HALF]
            if not np.isnan(seg).any():     # 整段角度都有效才要
                samples.append(seg)

    samples = np.array(samples)             # (K, 40, 2)
    print(f"采集到 {len(samples)} 段相位对齐的干净正手")

    template = samples.mean(axis=0)         # (40, 2)
    template_std = samples.std(axis=0)
    np.save(os.path.join(ROOT, 'forehand_template.npy'), template)
    print(f"模板已保存 forehand_template.npy  肘均值{template[:,0].mean():.1f}° 肩均值{template[:,1].mean():.1f}°")

    # 对比图：旧模板 vs 新模板(带±1std带)
    old = np.load(os.path.join(ROOT, 'forehand_angles.npy'))[58010:58050]
    x = np.linspace(0, 1, HALF*2)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7))
    fig.suptitle(f'新模板(red, {len(samples)}段平均, 击球对齐在中点) vs 旧模板(gray, 单段手取)', fontsize=12)
    for ax, c, name in [(a1, 0, '肘关节'), (a2, 1, '肩关节')]:
        for s in samples:
            ax.plot(x, s[:, c], color='red', alpha=0.04)
        ax.plot(x, template[:, c], 'r-', lw=2.5, label='新模板(平均)')
        ax.fill_between(x, template[:, c]-template_std[:, c], template[:, c]+template_std[:, c],
                        color='red', alpha=0.15, label='±1标准差')
        ax.plot(np.linspace(0, 1, len(old)), old[:, c], 'k--', lw=1.5, label='旧模板')
        ax.axvline(0.5, color='blue', ls=':', lw=1, label='击球瞬间(锚点)')
        ax.set_title(name); ax.set_ylabel('角度(°)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    a2.set_xlabel('动作进程')
    plt.tight_layout()
    out = os.path.join(ROOT, '学习', '模板对比.png')
    plt.savefig(out, dpi=120); plt.close()
    print(f"对比图已保存 {out}")


if __name__ == '__main__':
    main()
