"""
A3：IoU —— 框预测得准不准，用一个数字衡量
==========================================
IoU(交并比) = 两个框的"交集面积 / 并集面积"。
  完全重合 IoU=1(完美)，完全不沾 IoU=0(最差)。
YOLO 的框损失就建立在 IoU 上：loss 越小 = 框越准。

画三种情况：预测框(蓝) vs 真实框(绿) 不同重合度，看 IoU 怎么变。
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def iou(a, b):
    """a,b = [x1,y1,x2,y2]"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union, inter, union


def main():
    gt = [2, 2, 6, 6]   # 真实框(绿)，固定
    preds = [[2, 2, 6, 6], [3, 3, 7, 7], [5, 5, 9, 9]]  # 三个预测框，越来越偏
    titles = ['预测=真实', '部分重合', '几乎不沾']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, pred, t in zip(axes, preds, titles):
        v, inter, union = iou(gt, pred)
        ax.add_patch(plt.Rectangle((gt[0], gt[1]), gt[2]-gt[0], gt[3]-gt[1],
                                   fill=False, edgecolor='green', lw=3, label='真实框'))
        ax.add_patch(plt.Rectangle((pred[0], pred[1]), pred[2]-pred[0], pred[3]-pred[1],
                                   fill=False, edgecolor='blue', lw=3, ls='--', label='预测框'))
        # 交集涂色
        ix1, iy1 = max(gt[0], pred[0]), max(gt[1], pred[1])
        ix2, iy2 = min(gt[2], pred[2]), min(gt[3], pred[3])
        if ix2 > ix1 and iy2 > iy1:
            ax.add_patch(plt.Rectangle((ix1, iy1), ix2-ix1, iy2-iy1,
                                       color='red', alpha=0.3))
        ax.set_xlim(0, 11); ax.set_ylim(0, 11); ax.set_aspect('equal')
        ax.invert_yaxis(); ax.grid(alpha=0.3); ax.legend(loc='upper right')
        ax.set_title(f'{t}\nIoU = 交{inter:.0f}/并{union:.0f} = {v:.2f}   框损失=1-IoU={1-v:.2f}',
                     fontsize=11)
    fig.suptitle('IoU 交并比：框预测得多准（红=交集）', fontsize=14)
    plt.tight_layout()
    out = os.path.join(ROOT, '学习', '输出图', 'IoU损失.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"已保存 {out}")


if __name__ == '__main__':
    main()
