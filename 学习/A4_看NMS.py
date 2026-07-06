"""
A4：NMS 非极大值抑制 —— 同一物体多个重复框，怎么只留一个
==========================================================
YOLO 每个格子都可能为同一个人冒出一个框 → 一个人好几个框。
NMS 用 IoU 清理：留置信度最高的，删掉和它重叠太多的。

左图：NMS 前(5个框)  右图：NMS 后(只剩2个真物体)
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0


def nms(boxes, scores, iou_thresh=0.5):
    """经典 NMS。boxes:[[x1,y1,x2,y2]], scores:[..]; 返回保留的下标"""
    idx = list(np.argsort(scores)[::-1])   # 按置信度从高到低
    keep = []
    while idx:
        cur = idx.pop(0)                   # 取当前最高分
        keep.append(cur)
        # 删掉和 cur 重叠 > 阈值 的
        idx = [i for i in idx if iou(boxes[cur], boxes[i]) <= iou_thresh]
    return keep


def draw(ax, boxes, scores, keep_set, title):
    colors = plt.cm.tab10(np.linspace(0, 1, len(boxes)))
    for i, (b, s) in enumerate(zip(boxes, scores)):
        alive = i in keep_set
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1],
                     fill=False, lw=3 if alive else 1.2,
                     edgecolor=colors[i],
                     ls='-' if alive else ':',
                     alpha=1.0 if alive else 0.35))
        ax.text(b[0], b[1]-0.15, f'{s:.2f}', color=colors[i],
                fontsize=10, fontweight='bold' if alive else 'normal',
                alpha=1.0 if alive else 0.4)
    ax.set_xlim(0, 14); ax.set_ylim(0, 11); ax.set_aspect('equal')
    ax.invert_yaxis(); ax.grid(alpha=0.3); ax.set_title(title, fontsize=12)


def main():
    # 物体A: 3个重叠框；物体B: 2个重叠框
    boxes = [
        [2, 2, 6, 8], [2.5, 2.3, 6.4, 8.2], [1.7, 1.8, 5.6, 7.5],   # A
        [8, 3, 12, 9], [8.4, 3.4, 12.3, 9.2],                        # B
    ]
    scores = [0.92, 0.85, 0.78, 0.88, 0.70]
    keep = nms(boxes, scores, iou_thresh=0.5)
    print("保留的框下标:", keep, " 置信度:", [scores[i] for i in keep])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    draw(ax1, boxes, scores, set(range(len(boxes))), 'NMS 前：5 个框（同物体多框）')
    draw(ax2, boxes, scores, set(keep), f'NMS 后：留 {len(keep)} 个（实=保留 虚=删除）')
    fig.suptitle('NMS：按置信度排序，留最高分，删与它 IoU>0.5 的重复框', fontsize=14)
    plt.tight_layout()
    out = os.path.join(ROOT, '学习', '输出图', 'NMS效果.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"已保存 {out}")


if __name__ == '__main__':
    main()
