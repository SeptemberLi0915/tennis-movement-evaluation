"""
从干净视频构建正手模板（单机位 → 高质量 DBA 模板）
====================================================
用法：
  python 学习/从视频建模板.py "路径/你的正手视频.mp4"
  （视频要求：单机位、侧面、单人、多次正手；剪辑越少越好）

流程：YOLO提关键点 → LSTM检测每次正手 → 以击球峰锚定截取角度
      → DTW对齐平均(DBA) → 保存 forehand_template_clean.npy + 对比图

为什么这次能成：单机位 → 同一次正手在不同样本里角度可比 → 平均才有意义。
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import cv2
from ultralytics import YOLO
import matplotlib
import matplotlib.pyplot as plt
from pose_norm import normalize_kp

matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQ = 30
HALF = 25
PEAK_THRESH = 0.6
MIN_GAP = 30
CUT_JUMP = 80     # 髋部位移超过此值视为切镜头，截断连续段


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
    n, m = len(s1), len(s2)
    D = np.full((n+1, m+1), np.inf); D[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.linalg.norm(s1[i-1] - s2[j-1])
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        step = np.argmin([D[i-1, j-1], D[i-1, j], D[i, j-1]])
        if step == 0: i, j = i-1, j-1
        elif step == 1: i -= 1
        else: j -= 1
    return path[::-1]


def dba_average(samples, ref):
    L = len(ref)
    acc = np.zeros((L, 2)); cnt = np.zeros((L, 1))
    for s in samples:
        for (i, j) in dtw_path(ref, s):
            acc[i] += s[j]; cnt[i] += 1
    return acc / np.maximum(cnt, 1)


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else None
    if not video or not os.path.exists(video):
        print("用法: python 学习/从视频建模板.py \"你的正手视频.mp4\"")
        return

    model = ForehandLSTM()
    model.load_state_dict(torch.load(os.path.join(ROOT, 'models', 'forehand_lstm.pth'), map_location='cpu'))
    model.eval()
    norm = np.load(os.path.join(ROOT, 'models', 'forehand_lstm_norm.npz'))
    mean, std = norm['mean'], norm['std']

    # 1. YOLO 逐帧提关键点 → 角度 + 归一化特征（带缓存，避免反复跑YOLO）
    cache = os.path.join(ROOT, '学习', '_cache_' +
                         os.path.splitext(os.path.basename(video))[0] + '.npz')
    if os.path.exists(cache):
        print(f"用缓存 {cache}")
        d = np.load(cache)
        feats_arr, angles, hip = d['feats'], d['angles'], d['hip']
        valid = d['valid']
        n = len(feats_arr)
    else:
        yolo = YOLO(os.path.join(ROOT, 'models', 'yolov8n-pose.pt'))
        cap = cv2.VideoCapture(video)
        feats_l, angle_l, hip_l, valid_l = [], [], [], []
        print("提取关键点中...")
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            r = yolo.predict(frame, conf=0.5, imgsz=640, classes=[0], verbose=False)
            feat = np.zeros(34); ang = [np.nan, np.nan]; hipc = [np.nan, np.nan]; ok_kp = False
            if r[0].keypoints is not None and len(r[0].keypoints.xy) > 0:
                kp = r[0].keypoints.xy[0].cpu().numpy()
                if len(kp) == 17 and kp[11][0] > 0 and kp[12][0] > 0:
                    h = (kp[11] + kp[12]) / 2
                    feat, ok_kp = normalize_kp(kp); hipc = h.tolist()
                    rs, re, rw, rh = kp[6], kp[8], kp[10], kp[12]
                    if np.linalg.norm(rs-re) > 15 and np.linalg.norm(re-rw) > 15:
                        e = angle_3pts(rs, re, rw); s = angle_3pts(rh, rs, re)
                        if 30 < e < 175 and 10 < s < 160:
                            ang = [e, s]
            feats_l.append(feat); angle_l.append(ang); hip_l.append(hipc); valid_l.append(ok_kp)
        cap.release()
        feats_arr = np.array(feats_l, dtype=np.float32)
        angles = np.array(angle_l, dtype=float)
        hip = np.array(hip_l, dtype=float)
        valid = np.array(valid_l, dtype=bool)
        n = len(feats_arr)
        np.savez(cache, feats=feats_arr, angles=angles, hip=hip, valid=valid)
    print(f"总帧数 {n} | 有人体检出 {int(valid.sum())} | 角度有效 {int((~np.isnan(angles[:,0])).sum())}")

    # 诊断：整段视频跑检测，看检测器在该视角下到底响不响应
    full = (feats_arr - mean) / std
    if n >= SEQ:
        wins_all = np.array([full[i:i+SEQ] for i in range(n-SEQ+1)], dtype=np.float32)
        with torch.no_grad():
            pall = []
            for bb in range(0, len(wins_all), 512):
                pall.append(torch.softmax(model(torch.tensor(wins_all[bb:bb+512])), dim=1)[:, 1])
            pall = torch.cat(pall).numpy()
        jumps = np.linalg.norm(np.diff(hip, axis=0), axis=1)
        print(f"诊断: 检测概率 max={pall.max():.2f} 均值={pall.mean():.2f} | "
              f">{PEAK_THRESH}的帧数={int((pall>PEAK_THRESH).sum())}")
        print(f"诊断: 髋部跳变 中位={np.nanmedian(jumps):.1f} >CUT_JUMP({CUT_JUMP})的次数={int(np.nansum(jumps>CUT_JUMP))}")

    # 漏检帧(valid=False)不当作切镜头，用线性插值补上，保证序列连续
    idx = np.arange(n)
    good = np.where(valid)[0]
    for c in range(34):
        feats_arr[:, c] = np.interp(idx, good, feats_arr[good, c])
    for c in range(2):
        hip[:, c] = np.interp(idx, good, hip[good, c])

    # 角度的 NaN 也插值补上（后视角下手臂重叠常测不出），同时记录原始有效性
    ang_valid = ~np.isnan(angles[:, 0])
    ga = np.where(ang_valid)[0]
    angles_filled = angles.copy()
    for c in range(2):
        angles_filled[:, c] = np.interp(idx, ga, angles[ga, c])

    # 2. 只在真正"切镜头"(髋部突跳)处断开
    segs, start = [], 0
    for i in range(1, n):
        if np.linalg.norm(hip[i] - hip[i-1]) > CUT_JUMP:
            segs.append((start, i)); start = i
    segs.append((start, n))
    print(f"切镜头分段: {len(segs)} 段, 最长 {max(b-a for a,b in segs)} 帧")

    # 3. 每个连续段跑检测 → 以击球峰截取角度
    samples = []
    for (a, b) in segs:
        if b - a < SEQ + 2*HALF:
            continue
        run = (feats_arr[a:b] - mean) / std
        wins = np.array([run[i:i+SEQ] for i in range(len(run)-SEQ+1)], dtype=np.float32)
        with torch.no_grad():
            probs = torch.softmax(model(torch.tensor(wins)), dim=1)[:, 1].numpy()
        prob_full = np.zeros(len(run))
        for i, pv in enumerate(probs):
            prob_full[i + SEQ//2] = pv
        pks = find_peaks(prob_full)
        print(f"  段[{a}:{b}] 检测到 {len(pks)} 个击球峰")
        for pk in pks:
            g = a + pk
            if g-HALF < a or g+HALF >= b:
                continue
            # 窗口原始角度有效率 >60% 才采用（避免整窗都是插值）
            if ang_valid[g-HALF:g+HALF].mean() > 0.6:
                samples.append(angles_filled[g-HALF:g+HALF])

    print(f"连续段 {len(segs)} 个，提取到 {len(samples)} 次干净正手")
    if len(samples) < 3:
        print("正手太少，换一个正手更多的视频")
        return

    samples = np.array(samples)
    plain = samples.mean(axis=0)
    ref = plain                      # 用普通平均当参考(更稳)，再DTW对齐迭代一轮
    dba = dba_average(samples, ref)
    np.save(os.path.join(ROOT, 'models', 'forehand_template_clean.npy'), dba)
    np.save(os.path.join(ROOT, 'models', 'forehand_template_clean_std.npy'), samples.std(axis=0))
    print(f"模板已保存 forehand_template_clean.npy(+_std)  肘{dba[:,0].mean():.1f}° 肩{dba[:,1].mean():.1f}°")

    x = np.linspace(0, 1, 2*HALF)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    fig.suptitle(f'干净视频模板  {len(samples)}次正手  普通平均 vs DTW对齐(DBA)', fontsize=12)
    for ax, c, name in [(axes[0], 0, '肘关节'), (axes[1], 1, '肩关节')]:
        for s in samples:
            ax.plot(x, s[:, c], color='gray', alpha=0.2)
        ax.plot(x, plain[:, c], 'b--', lw=2, label='普通平均')
        ax.plot(x, dba[:, c], 'r-', lw=2.5, label='DBA模板')
        ax.axvline(0.5, color='green', ls=':', lw=1, label='击球锚点')
        ax.set_title(name); ax.set_ylabel('角度(°)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    axes[1].set_xlabel('动作进程')
    plt.tight_layout()
    out = os.path.join(ROOT, '学习', '输出图', '模板_干净视频.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=120); plt.close()
    print(f"对比图已保存 {out}")


if __name__ == '__main__':
    main()
