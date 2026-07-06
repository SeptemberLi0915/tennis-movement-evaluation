"""
采集 idle（站立/等待，没在击球）样本
=====================================
思路：整段视频里，离任何一次击球都"足够远"的帧 = idle。
做法：用已有的击球时间标注，把每次击球前后 MARGIN 帧标为"动作期"排除，
      剩下的帧就是干净的 idle，存成 keypoints_idle_all.csv（格式同 forehand_all）。

为什么需要 idle 类：
  现在分类器只有正手/反手两类，球员站着不动也会被硬分成其中一类。
  有了 idle 类，才能在整段视频里判断"现在没在击球"，
  这是后面"滑动窗口检测击球边界"的前提。
"""

import pandas as pd
import numpy as np
import os

FPS = 29.15
MARGIN = 30   # 击球中心前后 30 帧算"动作期"，排除掉（一次完整挥拍约这么长）

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 每个视频：(关键点csv路径, 击球时间秒列表)
# 击球时间来自 archive/tennis_classifier.py 里你的人工标注（fh/bh 这里不区分，只要"有击球"）
VIDEOS = [
    (
        os.path.join(ROOT, 'keypoints', 'keypoints.csv'),
        [1, 5, 8, 11, 15, 18, 21, 24, 29, 33, 40, 44, 47, 50, 52, 55],
    ),
    (
        os.path.join(ROOT, 'keypoints', 'keypoints_test.csv'),
        [0, 2, 4.5, 7, 10, 12, 14, 17, 19, 21, 27, 39, 45, 48, 50, 57, 59, 61, 64],
    ),
]

KP_COLS = [f'kp{i}_{c}' for i in range(17) for c in ['x', 'y']]


def collect_idle_from_video(csv_path, hit_seconds):
    """返回这个视频里所有 idle 帧（DataFrame，含 frame + 关键点列）"""
    df = pd.read_csv(csv_path)
    n_frames = df['frame'].max() + 1

    # 1. 标记"动作期"：每次击球中心前后 MARGIN 帧
    is_action = np.zeros(int(n_frames) + 1, dtype=bool)
    for sec in hit_seconds:
        center = int(sec * FPS)
        lo = max(0, center - MARGIN)
        hi = min(len(is_action) - 1, center + MARGIN)
        is_action[lo:hi + 1] = True

    # 2. idle = 不在动作期 的帧
    idle_mask = df['frame'].apply(lambda f: not is_action[int(f)] if int(f) < len(is_action) else True)
    idle_df = df[idle_mask].copy()

    print(f"  {os.path.basename(csv_path)}: 总帧 {len(df)} → 动作期排除后 idle 帧 {len(idle_df)}")
    return idle_df


def main():
    all_idle = []
    print("采集 idle 样本...")
    for csv_path, hits in VIDEOS:
        if not os.path.exists(csv_path):
            print(f"  ⚠️ 找不到 {csv_path}，跳过")
            continue
        all_idle.append(collect_idle_from_video(csv_path, hits))

    idle = pd.concat(all_idle, ignore_index=True)
    idle['label'] = 'idle'

    # 整理成和 keypoints_forehand_all.csv 一样的列顺序：frame, label, kp...
    out = idle[['frame', 'label'] + KP_COLS]
    out_path = os.path.join(ROOT, 'data', 'keypoints_idle_all.csv')
    out.to_csv(out_path, index=False)

    print(f"\n共采集 idle 样本 {len(out)} 帧")
    print(f"已保存到 {out_path}")

    # 对比一下三类的数据量
    print("\n===== 三类数据量对比 =====")
    for name, path in [('forehand', os.path.join('data', 'keypoints_forehand_all.csv')),
                       ('backhand', os.path.join('keypoints', 'keypoints_backhand.csv')),
                       ('idle', os.path.join('data', 'keypoints_idle_all.csv'))]:
        p = os.path.join(ROOT, path)
        if os.path.exists(p):
            n = sum(1 for _ in open(p)) - 1
            print(f"  {name:10s}: {n} 帧")


if __name__ == '__main__':
    main()
