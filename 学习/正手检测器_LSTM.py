"""  
正手检测器 v3：LSTM 序列模型（二分类 正手 vs 非正手）
========================================================
v3 改进：
  - 增加 idle 数据（keypoints_idle_all.csv）
  - 水平翻转数据增强（左右关键点互换）
  - 全量数据 + class_weight（替代截断平衡）
  - CosineAnnealingLR 学习率调度
  - 早停（patience=15）
  - 梯度裁剪

输出：forehand_lstm.pth + forehand_lstm_norm.npz
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pose_norm import normalize_pose_df

torch.manual_seed(0)
np.random.seed(0)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KP_COLS = [f'kp{i}_{c}' for i in range(17) for c in ['x', 'y']]
SEQ = 30          # 序列长度（帧）
STRIDE = 3        # 滑窗步长
FPS = 29.15
MARGIN = 30       # idle 窗口要离任何击球这么多帧

# 两个有标注视频的"所有击球"时间（秒），正手+反手都算（idle 要避开全部）
VIDEO_HITS = {
    os.path.join(ROOT, 'keypoints', 'keypoints.csv'):
        [1, 5, 8, 11, 15, 18, 21, 24, 29, 33, 40, 44, 47, 50, 52, 55],
    os.path.join(ROOT, 'keypoints', 'keypoints_test.csv'):
        [0, 2, 4.5, 7, 10, 12, 14, 17, 19, 21, 27, 39, 45, 48, 50, 57, 59, 61, 64],
}

# COCO 左右关键点对（水平翻转增强用）
SWAP_PAIRS = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]


def flip_features(windows):
    """水平翻转：x取反 + 左右关键点互换"""
    flipped = np.array(windows, copy=True)
    flipped[..., ::2] *= -1   # 所有 x 取反
    for a, b in SWAP_PAIRS:
        tmp = flipped[..., [2*a, 2*a+1]].copy()
        flipped[..., [2*a, 2*a+1]] = flipped[..., [2*b, 2*b+1]]
        flipped[..., [2*b, 2*b+1]] = tmp
    return flipped


def contiguous_runs(df):
    """按帧号连续性把 df 切成多段，返回 [ndarray(帧数,34), ...]"""
    frames = df['frame'].values
    feats = df[KP_COLS].values.astype(np.float32)
    runs = []
    start = 0
    for i in range(1, len(frames)):
        if frames[i] != frames[i-1] + 1:   # 帧号跳变 → 断开
            runs.append(feats[start:i])
            start = i
    runs.append(feats[start:])
    return runs


def windows_from_runs(runs):
    """对每段连续帧滑窗，返回所有 (SEQ,34) 窗口"""
    out = []
    for run in runs:
        for i in range(0, len(run) - SEQ + 1, STRIDE):
            out.append(run[i:i + SEQ])
    return out


def build_forehand():
    df = normalize_pose_df(pd.read_csv(os.path.join(ROOT, 'data', 'keypoints_forehand_all.csv')), KP_COLS)
    return windows_from_runs(contiguous_runs(df))


def build_backhand():
    df = normalize_pose_df(pd.read_csv(os.path.join(ROOT, 'keypoints', 'keypoints_backhand.csv')), KP_COLS)
    return windows_from_runs(contiguous_runs(df))


def build_idle():
    """从整段视频滑窗，窗口完全不碰任何击球(±MARGIN)才收为 idle"""
    out = []
    for path, hits in VIDEO_HITS.items():
        df = normalize_pose_df(pd.read_csv(path), KP_COLS)
        feats = df[KP_COLS].values.astype(np.float32)
        frames = df['frame'].values
        hit_frames = [int(s * FPS) for s in hits]
        for i in range(0, len(feats) - SEQ + 1, STRIDE):
            w_lo, w_hi = frames[i], frames[i + SEQ - 1]
            if all(w_hi < h - MARGIN or w_lo > h + MARGIN for h in hit_frames):
                out.append(feats[i:i + SEQ])
    return out


def build_idle_csv():
    """从 keypoints_idle_all.csv 构造额外 idle 窗口"""
    path = os.path.join(ROOT, 'data', 'keypoints_idle_all.csv')
    if not os.path.exists(path):
        return []
    df = normalize_pose_df(pd.read_csv(path), KP_COLS)
    return windows_from_runs(contiguous_runs(df))


# ── 模型（和 train_forehand.py 同款）──────────────────
class ForehandLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=34, hidden_size=64,
                            num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def main():
    print("构造序列...")
    fore = build_forehand()
    back = build_backhand()
    idle = build_idle()
    idle_csv = build_idle_csv()
    non_fore = back + idle + idle_csv
    print(f"  正手窗口 {len(fore)} | 非正手窗口 {len(non_fore)}")
    print(f"    反手{len(back)} + idle标注{len(idle)} + idle_csv{len(idle_csv)}")

    # ── 翻转增强 ──
    fore_arr = np.array(fore, dtype=np.float32)
    non_arr = np.array(non_fore, dtype=np.float32)
    fore_flip = flip_features(fore_arr)
    non_flip = flip_features(non_arr)
    X_fore = np.concatenate([fore_arr, fore_flip], axis=0)
    X_non = np.concatenate([non_arr, non_flip], axis=0)
    print(f"  翻转后: 正手{len(X_fore)} | 非正手{len(X_non)}")

    # ── 全量数据 + class weight ──
    X = np.concatenate([X_fore, X_non], axis=0)
    y = np.concatenate([np.ones(len(X_fore)), np.zeros(len(X_non))]).astype(np.int64)

    # 标准化（按特征维，用所有帧统计），保存供推理
    flat = X.reshape(-1, 34)
    mean, std = flat.mean(0), flat.std(0) + 1e-8
    X = (X - mean) / std

    # 类别权重（处理不平衡）
    n_pos, n_neg = len(X_fore), len(X_non)
    w_pos = len(X) / (2 * n_pos)
    w_neg = len(X) / (2 * n_neg)
    class_w = torch.tensor([w_neg, w_pos], dtype=torch.float32)
    print(f"  class_weight: 非正手={w_neg:.2f}, 正手={w_pos:.2f}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)
    Xtr = torch.tensor(Xtr); ytr = torch.tensor(ytr)
    Xte = torch.tensor(Xte); yte = torch.tensor(yte)

    model = ForehandLSTM()
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150, eta_min=1e-5)

    EPOCHS = 150
    PATIENCE = 15
    best_acc = 0.0
    wait = 0
    best_state = None

    print(f"\n开始训练（{EPOCHS} epochs, 早停patience={PATIENCE}）...")
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        total = 0
        for i in range(0, len(Xtr), 128):
            idx = perm[i:i + 128]
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()

        if ep % 5 == 0 or ep == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                pred_te = model(Xte).argmax(1)
                acc = (pred_te == yte).float().mean().item()
            lr_now = opt.param_groups[0]['lr']
            print(f"  epoch {ep:3d} | loss {total:.3f} | acc {acc:.2%} | lr {lr_now:.1e}")

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 5
            if wait >= PATIENCE:
                print(f"  早停 @ epoch {ep}（best acc = {best_acc:.2%}）")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(1)
    print(f"\n===== 最终测试结果 (best acc={best_acc:.2%}) =====")
    print(classification_report(yte.numpy(), pred.numpy(),
                                target_names=['非正手', '正手']))

    torch.save(model.state_dict(), os.path.join(ROOT, 'models', 'forehand_lstm.pth'))
    np.savez(os.path.join(ROOT, 'models', 'forehand_lstm_norm.npz'), mean=mean, std=std)
    print("已保存: forehand_lstm.pth + forehand_lstm_norm.npz")


if __name__ == '__main__':
    main()
