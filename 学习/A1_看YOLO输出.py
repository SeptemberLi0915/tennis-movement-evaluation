"""
A1：看清 YOLO-Pose 到底输出了什么
=================================
在一张网球帧上画出 YOLO 的输出：
  1. 人体检测框 (boxes) + 置信度
  2. 17 个 COCO 关键点 (keypoints) + 编号
  3. 骨架连线

用法: python 学习/A1_看YOLO输出.py "video1/tennis video2.mp4" 150
      （第二个参数是帧号，默认 150）
"""
import os
import sys
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ultralytics import YOLO

matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# COCO 17 关键点名称
NAMES = ['鼻', '左眼', '右眼', '左耳', '右耳', '左肩', '右肩', '左肘', '右肘',
         '左腕', '右腕', '左髋', '右髋', '左膝', '右膝', '左踝', '右踝']
# 骨架连线（哪两个点连起来）
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (0, 5), (0, 6), (3, 5), (4, 6)]


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'video1', 'tennis video2.mp4')
    frame_no = int(sys.argv[2]) if len(sys.argv) > 2 else 150

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("读不到这一帧，换个帧号试试")
        return
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    yolo = YOLO(os.path.join(ROOT, 'models', 'yolov8n-pose.pt'))
    r = yolo.predict(frame, conf=0.5, imgsz=640, classes=[0], verbose=False)[0]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(rgb)

    n_person = 0 if r.boxes is None else len(r.boxes)
    print(f"检测到 {n_person} 个人")

    for pi in range(n_person):
        # 1. 框 + 置信度
        x1, y1, x2, y2 = r.boxes.xyxy[pi].cpu().numpy()
        conf = float(r.boxes.conf[pi])
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor='lime', lw=2))
        ax.text(x1, y1 - 6, f'人 {conf:.2f}', color='black',
                fontsize=11, bbox=dict(facecolor='lime', alpha=0.8, pad=1))

        # 2. 关键点 + 编号
        kp = r.keypoints.xy[pi].cpu().numpy()       # (17,2)
        kconf = r.keypoints.conf[pi].cpu().numpy()  # (17,)
        # 3. 骨架连线
        for a, b in SKELETON:
            if kconf[a] > 0.3 and kconf[b] > 0.3:
                ax.plot([kp[a, 0], kp[b, 0]], [kp[a, 1], kp[b, 1]],
                        color='cyan', lw=2, alpha=0.7)
        for i, (x, y) in enumerate(kp):
            if kconf[i] > 0.3:
                ax.scatter(x, y, c='red', s=40, zorder=3)
                ax.text(x + 4, y, f'{i}.{NAMES[i]}', color='yellow', fontsize=8,
                        bbox=dict(facecolor='black', alpha=0.5, pad=0.5))

    ax.set_title(f'YOLO-Pose 输出  帧{frame_no}  绿框=人体检测  红点=17关键点  青线=骨架', fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    out = os.path.join(ROOT, '学习', '输出图', 'YOLO输出可视化.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"已保存 {out}")


if __name__ == '__main__':
    main()
