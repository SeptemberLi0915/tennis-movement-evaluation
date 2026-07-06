"""
共享姿态归一化（尺度不变）
==========================
关键点特征统一做两步归一化，保证不同球员、不同机位距离下数值一致：
  1. 平移：减去髋部中心 (kp11,kp12 的中点)
  2. 尺度：除以躯干长度 (肩中心 kp5,kp6 → 髋中心 的距离)

只有这样，离镜头近(画面大)和离镜头远(画面小)的同一个动作，特征才相同。
训练(正手检测器_LSTM)、建模板(从视频建模板)、教练(正手教练) 必须用同一套归一化。
"""
import numpy as np


def normalize_kp(kp):
    """单帧 kp:(17,2) → (feat:(34,), ok:bool)。尺度无效时返回零向量+False。"""
    hip = (kp[11] + kp[12]) / 2.0
    sh = (kp[5] + kp[6]) / 2.0
    torso = float(np.linalg.norm(sh - hip))
    if not np.isfinite(torso) or torso < 1e-3:
        return np.zeros(34, dtype=np.float32), False
    return ((kp - hip) / torso).reshape(-1).astype(np.float32), True


def normalize_pose_df(df, kp_cols):
    """训练用：对整张 DataFrame 向量化做尺度归一化（无效躯干长用中位数兜底）。"""
    df = df.copy()
    arr = df[kp_cols].values.reshape(len(df), 17, 2).astype(np.float64)
    hip = (arr[:, 11] + arr[:, 12]) / 2.0
    sh = (arr[:, 5] + arr[:, 6]) / 2.0
    torso = np.linalg.norm(sh - hip, axis=1)
    good = torso > 1e-3
    med = np.median(torso[good]) if good.any() else 1.0
    torso = np.where(good, torso, med)
    arr = (arr - hip[:, None, :]) / torso[:, None, None]
    df[kp_cols] = arr.reshape(len(df), 34)
    return df
