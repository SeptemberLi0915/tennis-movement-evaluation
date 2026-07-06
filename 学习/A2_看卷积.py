"""
A2：卷积到底在干嘛 —— 在网球帧上跑几个经典卷积核
====================================================
卷积 = 一个小窗口(卷积核)在图上滑动，每个位置做"加权求和"。
不同的核，提取不同的特征：边缘、模糊、锐化……
YOLO 的 Backbone 就是一层层卷积，核的数值是"学"出来的。

用法: python 学习/A2_看卷积.py "video1/tennis video2.mp4" 200
"""
import os
import sys
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 几个经典 3x3 卷积核
KERNELS = {
    '原图(灰度)': None,
    '竖直边缘(Sobel-X)': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    '水平边缘(Sobel-Y)': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
    '模糊(均值)': np.ones((3, 3), np.float32) / 9,
    '锐化': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    '轮廓(拉普拉斯)': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32),
}


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'video1', 'tennis video2.mp4')
    frame_no = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("读不到这一帧")
        return
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (name, k) in zip(axes.flat, KERNELS.items()):
        if k is None:
            out = gray
        else:
            # filter2D = 做卷积(严格说是相关)，每个像素 = 周围3x3和核加权求和
            out = cv2.filter2D(gray, ddepth=cv2.CV_32F, kernel=k)
            out = np.clip(np.abs(out), 0, 255).astype(np.uint8)
        ax.imshow(out, cmap='gray')
        ax.set_title(name, fontsize=12)
        ax.axis('off')
    fig.suptitle(f'同一张图 × 不同卷积核 = 不同特征  (帧{frame_no})', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(ROOT, '学习', '输出图', '卷积效果.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"已保存 {out_path}")


if __name__ == '__main__':
    main()
