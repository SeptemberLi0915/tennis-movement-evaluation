"""
正手 AI 教练（端到端 MVP）
==========================
流程：视频 → YOLO关键点(缓存) → LSTM检测每段正手 → 截角度 → DTW比模板 → 诊断+图
模板：forehand_template_clean.npy（单视角DBA，见 学习/从视频建模板.py）
注意：待分析视频必须和模板【同机位视角】，否则2D角度不可比。

用法：python 学习/正手教练.py "后视角正手视频.mp4"
输出：学习/教练报告/ 下每段正手一张诊断图
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import cv2
from scipy.signal import find_peaks
from pose_norm import normalize_kp

matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KP_COLS = [f'kp{i}_{c}' for i in range(17) for c in ['x', 'y']]
FPS = 29.15
SEQ = 30          # LSTM 窗口
WIN = 40          # 重采样后的统一长度（DTW 用）

# 检测超参
SMOOTH_K = 5
PEAK_THRESH = 0.55
EXPAND_LOW = 0.2
MIN_GAP = 30
MAX_HALF = 35
SKELETON = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

# ── 评分 ──
DTW_BEST = 500
DTW_WORST = 2500

def dtw_to_score(dtw_dist):
    """把DTW距离映射到 0~100 分（√曲线，对中间范围更宽容）"""
    ratio = max(0.0, min(1.0, (DTW_WORST - dtw_dist) / (DTW_WORST - DTW_BEST)))
    return max(0, min(100, int(100 * ratio ** 0.5)))


# ── 模型 ──────────────────────────────────────────────
class ForehandLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=34, hidden_size=64,
                            num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ── 工具 ──────────────────────────────────────────────
def angle_3pts(A, B, C):
    BA = A - B; BC = C - B
    cos = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def normalize_hip_arr(kp_row):
    """kp_row: (34,) 原始坐标 → 髋部归一化 (34,)"""
    xs = kp_row.reshape(17, 2)
    hip = (xs[11] + xs[12]) / 2
    return (xs - hip).reshape(-1)


def moving_average(x, k):
    return np.convolve(x, np.ones(k) / k, mode='same')


def smooth2d(arr, k=5):
    out = np.copy(arr)
    for c in range(arr.shape[1]):
        out[:, c] = np.convolve(arr[:, c], np.ones(k)/k, mode='same')
    return out


# 击球点检测参数(经7次真值标定, 详见 学习/验证击球点.py)
CONTACT_LAG = 3       # 中距情形用 LSTM峰值+此偏移(pk普遍早约3帧)
CONTACT_BURST_LAG = 1 # 近档前冲峰比真实击球普遍早约1帧, 加此偏移
CONTACT_BACK = 22     # 在pk前多少帧内搜索手腕前冲峰(覆盖概率饱和导致pk偏后的情况)
CONTACT_FWD = 33      # 在pk后多少帧内搜索(需足够大: 慢挥/高帧率视频触球可能在pk后30帧, 如直臂引拍)
CONTACT_NEAR = 3      # 前冲峰离pk≤此帧数才信任(标定发现d≤3时手腕峰≈触球, d≥4时滞后3-4帧)
CONTACT_JUMP = 12     # 前冲峰若离pk超过此帧数, 判定pk不可靠, 改用前冲峰
CONTACT_FAR_LAG = 5   # 远档且峰在pk之后(延迟直臂挥拍, pk落在引拍相): 手腕前冲峰领先触球约5帧
CONTACT_LATE_LAG = 1  # 远档且峰在pk之前(概率饱和使pk偏晚): 手腕峰≈触球, 加此偏移(同近档)


def find_contact_frame(wrist_seg, elbow_seg=None, center=None, window=10):
    """定位真正的击球瞬间(经7次真值标定, 平均误差<1帧)。
    以LSTM峰值 center 为锚点, 在其附近找手腕"沿击球方向前送"峰值 burst, 按 burst 距 center 分三档:
      近档(|burst-center|≤CONTACT_NEAR): 手腕峰即触球瞬间, 最准, 用 burst+CONTACT_BURST_LAG。
      中档(CONTACT_NEAR<..≤CONTACT_JUMP): burst噪声较大, 用 center+CONTACT_LAG(LSTM更稳)。
      远档(>CONTACT_JUMP): center被概率饱和拖偏, 不可靠, 改用 burst+CONTACT_BURST_LAG。
    手腕信号: 沿击球方向的水平前送速度×肘伸展权重, 并平滑以压制孤立跳变(保留持续前冲)。
    wrist_seg:(L,2)右手腕坐标; elbow_seg:(L,)肘角; center:LSTM峰值局部索引。
    返回击球点在段内的局部索引。
    """
    L = len(wrist_seg)
    if L < 3:
        return L // 2
    xs = wrist_seg[:, 0].astype(float).copy()
    ys = wrist_seg[:, 1].astype(float).copy()
    valid = (xs > 0) & (ys > 0)
    if valid.sum() < 3:
        return int(np.clip((center or L // 2) + CONTACT_LAG, 0, L - 1))
    # 1. 插值填补无效点 + 平滑, 计算沿击球方向前送速度
    ix = np.arange(L)
    xs = np.interp(ix, ix[valid], xs[valid])
    xs = moving_average(xs, 3)
    dx = np.gradient(xs)
    direction = np.sign(xs[-1] - xs[0]) or 1.0
    fwd = np.clip(dx * direction, 0, None)
    fwd = moving_average(fwd, 3)  # 压制孤立跳变, 保留持续前冲(真实挥拍)
    score = fwd.copy()
    if elbow_seg is not None and len(elbow_seg) == L:
        ext = np.clip((np.asarray(elbow_seg, float) - 110.0) / 50.0, 0.0, 1.0)
        score = fwd * ext
    # 2. 无 center: 全段取前冲峰(建模板等场景)
    if center is None:
        return int(np.argmax(score)) if score.max() > 0 else L // 2
    center = int(np.clip(center, 0, L - 1))
    pk_contact = int(np.clip(center + CONTACT_LAG, 0, L - 1))
    # 3. 在 center 附近窗口找手腕前冲峰
    lo = max(0, center - CONTACT_BACK)
    hi = min(L, center + CONTACT_FWD + 1)
    masked = np.full(L, -1.0)
    masked[lo:hi] = score[lo:hi]
    smax = masked.max()
    if smax <= 0:
        return pk_contact
    # 取"首个足够高"的前冲峰(height≥0.5*峰值): 直臂引拍/慢挥时pk落在引拍相, 真实前冲峰
    # 在pk之后较远处; 而随挥常有更大的第二前冲峰, 故取"首个高峰"而非全局最大, 避免选到随挥。
    peaks, _ = find_peaks(masked, height=0.5 * smax)
    burst = int(peaks[0]) if len(peaks) else int(np.argmax(masked))
    d = abs(burst - center)
    # 4. 近档: 手腕前冲峰紧邻center, 即真正触球瞬间(经标定最准)
    if d <= CONTACT_NEAR:
        return int(np.clip(burst + CONTACT_BURST_LAG, 0, L - 1))
    # 5. 远档: center不可靠, 改用前冲峰。按峰相对center的方向分两种:
    if d > CONTACT_JUMP:
        if burst >= center:
            # pk偏早(直臂延迟挥拍, pk落在引拍相): 手腕前冲峰领先触球约CONTACT_FAR_LAG帧
            return int(np.clip(burst + CONTACT_FAR_LAG, 0, L - 1))
        # pk偏晚(概率饱和): 手腕峰≈触球稍后, 回退CONTACT_LATE_LAG
        return int(np.clip(burst + CONTACT_LATE_LAG, 0, L - 1))
    # 6. 中档: burst噪声较大, 信任 LSTM峰值+lag
    return pk_contact


def elbow_phases(elbow, wrist_seg=None, center=None):
    """以真正击球点对齐分3阶段, 返回各阶段特征肘角。
    击球肘角 = 击球瞬间的肘角(应较伸展 ~140°+)
    引拍伸展 = 击球前的最大伸展角
    随挥弯曲 = 击球后的最小角(随挥收臂最弯)
    contact_ratio = 击球点相对位置(0~1); center=LSTM峰值局部索引(约束击球点搜索)。
    返回 (引拍伸展, 击球肘角, 随挥弯曲, contact_ratio)
    若无 wrist_seg 则退回肘角最低点(兼容)。
    """
    L = len(elbow)
    if wrist_seg is not None and len(wrist_seg) == L:
        c = find_contact_frame(wrist_seg, elbow, center=center)
    else:
        c = int(np.argmin(elbow))
    contact = float(elbow[c])
    back_ext = float(elbow[:c+1].max()) if c > 0 else contact
    follow_bend = float(elbow[c:].min()) if c < L - 1 else contact
    contact_ratio = c / (L - 1)
    return back_ext, contact, follow_bend, contact_ratio


# 分阶段阈值：优先加载从训练视频学习到的值(学习阈值.py生成)，否则用兜底值
def _load_phase_thresholds():
    path = os.path.join(ROOT, 'forehand_phase_thresholds.npz')
    if os.path.exists(path):
        d = np.load(path)
        return float(d['t_back']), float(d['t_contact'])
    return 130.0, 130.0  # 兜底(引拍伸展下限, 击球肘角下限)

PHASE_T_BACK, PHASE_T_CONTACT = _load_phase_thresholds()


def elbow_phase_tips(elbow, wrist_seg=None, center=None, t_back=None, t_contact=None):
    """基于分阶段肘角生成诊断。以挥速峰值定位真正击球点。
    击球瞬间手臂应较伸展(阈值=训练视频击球肘角的25分位)。
    阈值从训练视频学习得到:
      引拍伸展下限 = 引拍伸展角的25分位
      击球肘角下限 = 击球肘角的25分位
    返回 (tip列表, (引拍伸展,击球肘角,随挥弯曲), is_valid)
    is_valid=False 表示段落不完整(击球点在边缘)。
    """
    if t_back is None:
        t_back = PHASE_T_BACK
    if t_contact is None:
        t_contact = PHASE_T_CONTACT
    back_ext, contact, follow_bend, cr = elbow_phases(elbow, wrist_seg, center=center)
    # 击球点须在序列中段(15%~90%), 否则是不完整挥拍
    is_valid = 0.15 <= cr <= 0.90
    tips = []
    if contact < t_contact:
        tips.append(f"击球时手臂过弯,未充分伸展发力({contact:.0f}°→宜≥{t_contact:.0f}°)")
    if back_ext < t_back:
        tips.append(f"引拍伸展不足({back_ext:.0f}°→宜≥{t_back:.0f}°)")
    return tips, (back_ext, contact, follow_bend), is_valid


def calc_swing_speed(raw_kp, start, end, fps=FPS):
    """计算挥拍速度：手腕(kp10)帧间像素位移 → 峰值速度和平均速度(px/s)
    用中位数绝对偏差(MAD)过滤关键点跳变引起的异常值"""
    kp_seg = raw_kp[start:end+1]  # (frames, 17, 2)
    wrist = kp_seg[:, 10]          # (frames, 2) 右手腕
    valid = (wrist[:, 0] > 0) & (wrist[:, 1] > 0)
    if valid.sum() < 3:
        return 0.0, 0.0, np.zeros(len(kp_seg))
    # 帧间位移
    disp = np.zeros(len(wrist))
    for i in range(1, len(wrist)):
        if valid[i] and valid[i-1]:
            disp[i] = np.linalg.norm(wrist[i] - wrist[i-1]) * fps
    # MAD 过滤异常跳变（>3倍中位数偏差视为噪声）
    nonzero = disp[disp > 0]
    if len(nonzero) > 3:
        med = np.median(nonzero)
        mad = np.median(np.abs(nonzero - med)) + 1e-6
        threshold = med + 3 * mad * 1.4826
        disp = np.where(disp > threshold, med, disp)
    # 平滑
    if len(disp) > 5:
        disp = np.convolve(disp, np.ones(3)/3, mode='same')
    peak_speed = float(disp.max())
    avg_speed = float(disp[disp > 0].mean()) if (disp > 0).any() else 0.0
    return peak_speed, avg_speed, disp


def get_fps(video, default=FPS):
    """读取视频真实帧率(读不到则回退 default)。避免用写死的 FPS 导致时间/挥速偏差。"""
    try:
        cap = cv2.VideoCapture(video)
        f = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return f if f and f > 1 else default
    except Exception:
        return default


SWING_MIN_TRAVEL = 2.0  # 手腕轨迹包围盒/躯干长 的下限; 低于此判定该段无挥拍(假阳性)


def swing_travel(raw_kp, start, end):
    """挥拍存在度(尺度无关): 段内右手腕(kp10)轨迹包围盒对角线 / 躯干长(肩中—髋中)。
    真实正手 ≥2.4, 无挥拍的假阳性段 ≤1.6(经00005/00014标定)。返回值越大挥动越明显。"""
    kp = raw_kp[start:end+1]
    wrist = kp[:, 10].astype(float)
    ok = (wrist[:, 0] > 0) & (wrist[:, 1] > 0)
    if ok.sum() < 3:
        return 0.0
    sh = (kp[:, 5] + kp[:, 6]) / 2.0
    hip = (kp[:, 11] + kp[:, 12]) / 2.0
    torso = np.linalg.norm(sh - hip, axis=1)
    torso = torso[torso > 1]
    torso = float(np.median(torso)) if len(torso) else 0.0
    if torso <= 0:
        return 0.0
    bb = np.hypot(wrist[ok, 0].max() - wrist[ok, 0].min(),
                  wrist[ok, 1].max() - wrist[ok, 1].min())
    return float(bb / torso)


def dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
    d = np.full((n+1, m+1), np.inf); d[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.linalg.norm(s1[i-1] - s2[j-1])
            d[i, j] = cost + min(d[i-1, j], d[i, j-1], d[i-1, j-1])
    return d[n, m]


def detect_segments(probs, elbow_angles=None):
    """检测正手段落。支持两种模式:
    1. 概率有峰谷 → 传统峰值检测
    2. 概率持续高（连续正手）→ 用肘关节角度周期切分
    """
    n = len(probs)
    # ── 传统峰值检测 ──
    cand = [i for i in range(1, n-1)
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
        while e < n-1 and probs[e] > EXPAND_LOW and e - pk < MAX_HALF:
            e += 1
        raw.append((s, pk, e))
    raw.sort(key=lambda x: x[0])
    segs = []
    for s, pk, e in raw:
        if segs and s <= segs[-1][2]:
            ps, ppk, pe = segs[-1]
            best = pk if probs[pk] > probs[ppk] else ppk
            segs[-1] = (min(ps, s), best, max(pe, e))
        else:
            segs.append((s, pk, e))

    # ── 角度周期分割：拆分过长的段 ──
    # 如果一个段超过 2*MAX_HALF 帧且有肘角数据，用肘角极大值（击球=手臂最伸展）再切
    if elbow_angles is not None:
        refined = []
        for s, pk, e in segs:
            seg_len = e - s
            if seg_len <= 2 * MAX_HALF:
                refined.append((s, pk, e))
                continue
            # 在此段内找肘角极大值（击球瞬间=手臂最伸展）
            elb = elbow_angles[s:e+1].copy()
            elb_smooth = np.convolve(elb, np.ones(7)/7, mode='same')
            sub_peaks, _ = find_peaks(elb_smooth, distance=60, prominence=15)
            if len(sub_peaks) < 2:
                refined.append((s, pk, e))
                continue
            sub_peaks = sub_peaks + s  # 转回全局索引
            # 用相邻极大值的中点作为分界
            for j in range(len(sub_peaks)):
                seg_s = (sub_peaks[j-1] + sub_peaks[j]) // 2 if j > 0 else s
                seg_e = (sub_peaks[j] + sub_peaks[j+1]) // 2 if j < len(sub_peaks)-1 else e
                refined.append((seg_s, int(sub_peaks[j]), seg_e))
        segs = refined

    return segs


def resample(seq, n):
    """把 (L,2) 重采样到 (n,2)，消除长度差异（DTW 前置）"""
    x_old = np.linspace(0, 1, len(seq))
    x_new = np.linspace(0, 1, n)
    return np.column_stack([np.interp(x_new, x_old, seq[:, c]) for c in range(2)])


def generate_summary(results, out_path, video_name):
    """生成汇总报告图"""
    n = len(results)
    if n == 0:
        return
    scores = [r['score'] for r in results]
    has_speed = 'peak_speed' in results[0]
    fig = plt.figure(figsize=(max(12, n * 0.4), 12 if has_speed else 9))
    gs = fig.add_gridspec(4 if has_speed else 3, 1,
                          height_ratios=[3, 1.2, 1.2, 0.8] if has_speed else [3, 1.2, 0.8],
                          hspace=0.35)

    # 上: 评分柱状图
    ax1 = fig.add_subplot(gs[0])
    colors = ['#2ecc71' if s >= 80 else '#f39c12' if s >= 60 else '#e74c3c' for s in scores]
    ax1.bar(range(1, n+1), scores, color=colors, edgecolor='white', lw=0.5)
    ax1.axhline(80, color='green', ls='--', alpha=0.4, label='优秀(80)')
    ax1.axhline(60, color='orange', ls='--', alpha=0.4, label='及格(60)')
    ax1.set_xlabel('第N次正手'); ax1.set_ylabel('评分')
    ax1.set_ylim(0, 105); ax1.legend(loc='lower right'); ax1.grid(axis='y', alpha=0.3)
    ax1.set_title(f'{video_name}\n{n}次正手  平均{np.mean(scores):.0f}分  '
                  f'最高{max(scores)}  最低{min(scores)}', fontsize=14)

    # 中1: 肘/肩偏差散点图
    ax2 = fig.add_subplot(gs[1])
    e_diffs = [r['e_diff'] for r in results]
    s_diffs = [r['s_diff'] for r in results]
    ax2.scatter(range(1, n+1), e_diffs, c='blue', s=30, alpha=0.7, label='肘偏差(°)')
    ax2.scatter(range(1, n+1), s_diffs, c='red', s=30, alpha=0.7, label='肩偏差(°)')
    ax2.axhline(0, color='gray', ls='-', alpha=0.3)
    ax2.axhline(12, color='gray', ls=':', alpha=0.3); ax2.axhline(-12, color='gray', ls=':', alpha=0.3)
    ax2.set_xlabel('第N次正手'); ax2.set_ylabel('角度偏差(°)'); ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # 中2: 挥拍速度图
    if has_speed:
        ax_spd = fig.add_subplot(gs[2])
        peak_spds = [r['peak_speed'] for r in results]
        avg_spds = [r['avg_speed'] for r in results]
        x = range(1, n+1)
        ax_spd.bar(x, peak_spds, color='#3498db', alpha=0.7, label='峰值速度')
        ax_spd.plot(x, avg_spds, 'ro-', ms=4, lw=1.5, label='平均速度')
        ax_spd.set_xlabel('第N次正手'); ax_spd.set_ylabel('速度 (px/s)')
        ax_spd.legend(fontsize=8); ax_spd.grid(axis='y', alpha=0.3)

    # 下: 问题统计
    ax3 = fig.add_subplot(gs[-1])
    ax3.axis('off')
    cats = {'动作标准': 0, '肘弯曲过多': 0, '肘偏直': 0, '肩偏高': 0, '肩偏低': 0}
    for r in results:
        t = r['tips']
        if '接近标准' in t: cats['动作标准'] += 1
        if '弯曲过多' in t: cats['肘弯曲过多'] += 1
        if '偏直' in t: cats['肘偏直'] += 1
        if '肩部偏高' in t or '偏高' in t: cats['肩偏高'] += 1
        if '肩部偏低' in t or '偏低' in t: cats['肩偏低'] += 1
    parts = [f'{k}: {v}次' for k, v in cats.items() if v > 0]
    ax3.text(0.5, 0.5, '  |  '.join(parts), ha='center', va='center', fontsize=13,
             transform=ax3.transAxes,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray'))

    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()


def extract_cached(video, mean, std):
    """YOLO提关键点(带缓存)→标准化特征、插值后角度、原始角度有效掩码、原始关键点"""
    cache = os.path.join(ROOT, '学习', '_cache_' +
                         os.path.splitext(os.path.basename(video))[0] + '.npz')
    need_extract = True
    if os.path.exists(cache):
        d = np.load(cache)
        if 'raw_kp' in d.files:
            feats_arr, angles, valid, raw_kp = d['feats'], d['angles'], d['valid'], d['raw_kp']
            need_extract = False
        d.close()
        if need_extract:
            os.remove(cache)
    if need_extract:
        from ultralytics import YOLO
        yolo = YOLO(os.path.join(ROOT, 'models', 'yolov8n-pose.pt'))
        cap = cv2.VideoCapture(video)
        F, A, H, V, K = [], [], [], [], []
        print("提取关键点中...")
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            r = yolo.predict(frame, conf=0.5, imgsz=640, classes=[0], verbose=False)
            feat = np.zeros(34); ang = [np.nan, np.nan]; h = [np.nan, np.nan]; okk = False
            kp_raw = np.zeros((17, 2))
            if r[0].keypoints is not None and len(r[0].keypoints.xy) > 0:
                kp = r[0].keypoints.xy[0].cpu().numpy()
                if len(kp) == 17 and kp[11][0] > 0 and kp[12][0] > 0:
                    kp_raw = kp.copy()
                    hc = (kp[11] + kp[12]) / 2
                    feat, okk = normalize_kp(kp); h = hc.tolist()
                    rs, re, rw, rh = kp[6], kp[8], kp[10], kp[12]
                    if np.linalg.norm(rs-re) > 15 and np.linalg.norm(re-rw) > 15:
                        e = angle_3pts(rs, re, rw); s = angle_3pts(rh, rs, re)
                        if 30 < e < 175 and 10 < s < 160:
                            ang = [e, s]
            F.append(feat); A.append(ang); H.append(h); V.append(okk); K.append(kp_raw)
        cap.release()
        feats_arr = np.array(F, dtype=np.float32); angles = np.array(A, dtype=float)
        raw_kp = np.array(K, dtype=np.float32)
        np.savez(cache, feats=feats_arr, angles=angles,
                 hip=np.array(H, dtype=float), valid=np.array(V, dtype=bool), raw_kp=raw_kp)
        valid = np.array(V, dtype=bool)
    n = len(feats_arr)
    idx = np.arange(n)
    good = np.where(valid)[0]
    for c in range(34):
        feats_arr[:, c] = np.interp(idx, good, feats_arr[good, c])
    ang_valid = ~np.isnan(angles[:, 0])
    ga = np.where(ang_valid)[0]
    angles_filled = angles.copy()
    for c in range(2):
        angles_filled[:, c] = np.interp(idx, ga, angles[ga, c])
    return (feats_arr - mean) / std, angles_filled, ang_valid, n, raw_kp


def export_stroke_clips(video_path, raw_kp, angles, results, out_dir,
                        fps=FPS, pad_sec=0.4, slow_factor=3):
    """把每段正手裁剪为带YOLO骨架+肘角标注的慢动作短视频。
    slow_factor: 慢放倍数(输出帧率=原帧率/slow_factor), 击球帧高亮黄色并停顿。
    返回 [(路径, 标签), ...]
    """
    import subprocess, imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(out_dir, exist_ok=True)
    out_fps = max(1.0, vid_fps / slow_factor)  # 慢动作: 降低输出帧率
    clips = []
    for r in results:
        pad = int(pad_sec * vid_fps)
        f_start = max(0, r['start'] - pad)
        f_end = min(total_frames - 1, r['end'] + pad)
        contact_f = r.get('contact_frame', -1)
        out_path = os.path.join(out_dir, f"clip_{r['rank']:02d}.mp4")
        cmd = [
            ffmpeg_exe, '-y',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{w}x{h}', '-r', f'{out_fps:.3f}',
            '-i', '-',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-pix_fmt', 'yuv420p',
            out_path
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
        for i in range(f_start, f_end + 1):
            ok, frame = cap.read()
            if not ok:
                break
            is_contact = (i == contact_f)
            frame = _draw_skeleton(frame, raw_kp[i], highlight=is_contact)
            # 顶部信息条
            frame = _put_chinese(frame, f"第{r['rank']}次  {r['score']}分",
                                 (10, 8), (0, 255, 128), font_size=26)
            if is_contact:
                frame = _put_chinese(frame, "★ 击球瞬间", (10, 42), (0, 255, 255), font_size=24)
            # 击球帧多写几帧形成停顿, 强调关键姿态
            reps = 6 if is_contact else 1
            for _ in range(reps):
                proc.stdin.write(frame.tobytes())
        proc.stdin.close(); proc.wait()
        label = f"第{r['rank']}次  {r['score']}分  {r['tips']}"
        clips.append((out_path, label))
    cap.release()
    return clips


_font_cache = {}
def _get_font(size=24):
    """加载中文字体（带缓存）"""
    if size not in _font_cache:
        from PIL import ImageFont
        _font_cache[size] = None
        for p in [r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf',
                  r'C:\Windows\Fonts\simsun.ttc']:
            if os.path.exists(p):
                _font_cache[size] = ImageFont.truetype(p, size)
                break
        if _font_cache[size] is None:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]

def _put_chinese(frame, text, pos, color_rgb, font_size=24):
    """用 PIL 在 OpenCV 帧上绘制中文"""
    from PIL import Image, ImageDraw
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(pos, text, font=_get_font(font_size), fill=color_rgb)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _draw_skeleton(frame, kp, highlight=False):
    """画YOLO骨架+关节点, 并在右肘标注角度。highlight=True(击球帧)用醒目黄色。"""
    if kp.sum() <= 0:
        return frame
    line_col = (0, 255, 255) if highlight else (0, 255, 128)
    dot_col = (0, 128, 255) if highlight else (0, 200, 255)
    thick = 3 if highlight else 2
    for (a, b) in SKELETON:
        if kp[a][0] > 0 and kp[b][0] > 0:
            cv2.line(frame, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)), line_col, thick)
    for j in range(17):
        if kp[j][0] > 0:
            cv2.circle(frame, tuple(kp[j].astype(int)), 5 if highlight else 4, dot_col, -1)
    # 右肘角度标注
    rs, re, rw = kp[6], kp[8], kp[10]
    if np.linalg.norm(rs - re) > 10 and np.linalg.norm(re - rw) > 10:
        elb = angle_3pts(rs, re, rw)
        ex, ey = kp[8].astype(int)
        col = (0, 255, 255) if highlight else (255, 200, 0)
        frame = _put_chinese(frame, f"肘 {elb:.0f}°", (ex + 10, ey - 15), col, font_size=22)
    return frame


def generate_annotated_video(video_path, raw_kp, results, out_path):
    """生成带骨架+角度+评分标注的视频 (H.264, 支持中文)"""
    import subprocess, imageio_ffmpeg
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{w}x{h}', '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-pix_fmt', 'yuv420p',
        out_path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_seg = {}
    for r in results:
        for f in range(r['start'], r['end'] + 1):
            frame_seg[f] = r

    total = len(raw_kp)
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        kp = raw_kp[i]

        has_kp = kp.sum() > 0
        if has_kp:
            for (a, b) in SKELETON:
                if kp[a][0] > 0 and kp[b][0] > 0:
                    pa = tuple(kp[a].astype(int))
                    pb = tuple(kp[b].astype(int))
                    cv2.line(frame, pa, pb, (0, 255, 128), 2)
            for j in range(17):
                if kp[j][0] > 0:
                    cv2.circle(frame, tuple(kp[j].astype(int)), 4, (0, 200, 255), -1)

        info = frame_seg.get(i)
        if info:
            sc = info['score']
            color_bgr = (0, 200, 0) if sc >= 80 else (0, 200, 200) if sc >= 60 else (0, 0, 220)
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            frame = _put_chinese(frame, f"第{info['rank']}次正手  评分: {sc}",
                                 (10, 8), color_rgb, font_size=28)
            frame = _put_chinese(frame, info['tips'],
                                 (10, 45), (255, 255, 255), font_size=20)
            if info.get('peak_speed', 0) > 0:
                frame = _put_chinese(frame, f"挥速: {info['peak_speed']:.0f} px/s",
                                     (10, 70), (100, 220, 255), font_size=18)
            if has_kp:
                rs, re, rw = kp[6], kp[8], kp[10]
                if np.linalg.norm(rs - re) > 10 and np.linalg.norm(re - rw) > 10:
                    elb = angle_3pts(rs, re, rw)
                    ex, ey = kp[8].astype(int)
                    frame = _put_chinese(frame, f"肘:{elb:.0f}°",
                                         (ex + 10, ey - 20), (255, 200, 0), font_size=18)
            cv2.rectangle(frame, (0, 0), (w-1, h-1), color_bgr, 4)

        proc.stdin.write(frame.tobytes())

    cap.release()
    proc.stdin.close(); proc.wait()
    print(f"  标注视频: {out_path}")


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'video1', 'tennis video2.mp4')
    if not os.path.exists(video):
        print("用法: python 学习/正手教练.py \"后视角正手视频.mp4\"")
        return
    print(f"分析视频: {video}")

    # 1. 模型
    model = ForehandLSTM()
    model.load_state_dict(torch.load(os.path.join(ROOT, 'models', 'forehand_lstm.pth'), map_location='cpu'))
    model.eval()
    norm = np.load(os.path.join(ROOT, 'models', 'forehand_lstm_norm.npz'))
    mean, std = norm['mean'], norm['std']

    # 2. 提取关键点(缓存)+插值
    feats, angles, ang_valid, n, raw_kp = extract_cached(video, mean, std)
    fps = get_fps(video)  # 用视频真实帧率(而非写死FPS), 保证时间/挥速正确

    # 3. LSTM 逐帧概率 → 检测段
    probs = np.zeros(n)
    wins = np.array([feats[i:i+SEQ] for i in range(0, n - SEQ + 1)], dtype=np.float32)
    with torch.no_grad():
        out = []
        for b in range(0, len(wins), 512):
            out.append(torch.softmax(model(torch.tensor(wins[b:b+512])), dim=1)[:, 1])
        p = torch.cat(out).numpy()
    for i, pv in enumerate(p):
        probs[i + SEQ // 2] = pv
    probs = moving_average(probs, SMOOTH_K)
    segments = detect_segments(probs, elbow_angles=angles[:, 0])

    # 4. 单视角DBA模板 + 标准差带
    template = resample(np.load(os.path.join(ROOT, 'models', 'forehand_template_clean.npy')), WIN)
    tstd = resample(np.load(os.path.join(ROOT, 'models', 'forehand_template_clean_std.npy')), WIN)

    # 5. 每段做 DTW 分析 + 出图
    out_dir = os.path.join(ROOT, '学习', '教练报告')
    os.makedirs(out_dir, exist_ok=True)
    print(f"检测到 {len(segments)} 段正手，逐段分析...\n")

    results = []
    for rank, (s, pk, e) in enumerate(segments, 1):
        if ang_valid[s:e+1].mean() < 0.5:
            print(f"  第{rank:2d}次({pk/fps:.1f}s): 角度有效率过低，跳过")
            continue
        travel = swing_travel(raw_kp, s, e)
        if travel < SWING_MIN_TRAVEL:
            print(f"  第{rank:2d}次({pk/fps:.1f}s): 无明显挥拍动作(行程{travel:.1f}<{SWING_MIN_TRAVEL})，跳过")
            continue
        user = smooth2d(resample(angles[s:e+1], WIN), k=5)
        dist = dtw_distance(template, user)
        score = dtw_to_score(dist)
        e_diff = user[:, 0].mean() - template[:, 0].mean()
        s_diff = user[:, 1].mean() - template[:, 1].mean()

        # 相位诊断：用原始帧的肘角+手腕(挥速定位真正击球点)
        elbow_orig = smooth2d(angles[s:e+1], k=5)[:, 0]
        wrist_orig = raw_kp[s:e+1, 10]
        center = pk - s  # LSTM峰值(接近击球点)作为搜索中心
        etips, (back_ext, contact, follow_bend), valid = elbow_phase_tips(elbow_orig, wrist_orig, center=center)
        contact_frame = s + find_contact_frame(wrist_orig, elbow_orig, center=center)  # 真正击球帧(全局)
        tips = list(etips) if valid else []  # 击球点在边缘时不给分阶段建议
        if abs(s_diff) > 12:
            tips.append(f"肩部{'偏低' if s_diff < 0 else '偏高'}({s_diff:+.0f}°)")
        tip_str = '; '.join(tips) if tips else '动作接近标准，很好'

        peak_spd, avg_spd, _ = calc_swing_speed(raw_kp, s, e, fps=fps)

        results.append({
            'rank': rank, 'start': s, 'peak': pk, 'end': e,
            'score': score, 'dtw': dist,
            'e_diff': e_diff, 's_diff': s_diff, 'tips': tip_str,
            'contact_frame': contact_frame,
            'peak_speed': peak_spd, 'avg_speed': avg_spd
        })

        xt = np.linspace(0, 1, WIN)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle(f'第{rank}次正手  击球≈{pk/fps:.1f}秒  评分{score}  DTW={dist:.0f}\n{tip_str}',
                     fontsize=12)
        for ax, c, name, col in [(ax1, 0, '肘关节', 'b'), (ax2, 1, '肩关节', 'r')]:
            ax.fill_between(xt, template[:, c]-tstd[:, c], template[:, c]+tstd[:, c],
                            color='green', alpha=0.13, label='标准范围(±1σ)')
            ax.plot(xt, template[:, c], 'g--', lw=2, label=f'标准 {template[:,c].mean():.0f}°')
            ax.plot(xt, user[:, c], col+'-', lw=2, label=f'你 {user[:,c].mean():.0f}°')
            ax.set_title(name+'角度'); ax.set_ylabel('角度(°)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax2.set_xlabel('动作进程')
        plt.tight_layout()
        path = os.path.join(out_dir, f'stroke_{rank:02d}_{pk/fps:.1f}s.png')
        plt.savefig(path, dpi=120); plt.close()
        print(f"  第{rank:2d}次({pk/fps:4.1f}s): {score:3d}分  DTW={dist:5.0f}  肘{e_diff:+5.0f}°  肩{s_diff:+5.0f}°  速度{peak_spd:4.0f}px/s  → {tip_str}")

    # 6. 汇总报告
    if results:
        vname = os.path.splitext(os.path.basename(video))[0]
        summary_path = os.path.join(out_dir, f'汇总_{vname}.png')
        generate_summary(results, summary_path, vname)
        scores = [r['score'] for r in results]
        print(f"\n{'='*60}")
        print(f"  汇总: {len(results)}次正手  平均{np.mean(scores):.0f}分  最高{max(scores)}  最低{min(scores)}")
        print(f"  汇总图: {summary_path}")
    # 7. 标注视频
    vname = os.path.splitext(os.path.basename(video))[0] if not results else vname
    vid_out = os.path.join(out_dir, f'标注_{vname}.mp4')
    print("\n生成标注视频...")
    generate_annotated_video(video, raw_kp, results, vid_out)
    print(f"  诊断图: {out_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
