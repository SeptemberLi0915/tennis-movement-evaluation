"""
查看视频中的肘角度
用法: python 学习/查看肘角.py video2/forehand_side_view4.mp4
输出: 带骨架+肘角标注的视频
"""
import sys, os, cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from 正手教练 import angle_3pts, _put_chinese

# COCO skeleton (用于画骨架)
SKELETON = [
    (5, 7), (7, 9),    # 左臂
    (6, 8), (8, 10),   # 右臂
    (5, 6),            # 双肩
    (5, 11), (6, 12),  # 躯干
    (11, 12),          # 髋部
    (11, 13), (13, 15), # 左腿
    (12, 14), (14, 16), # 右腿
]

KP_NAMES = {
    0: '鼻', 5: '左肩', 6: '右肩', 7: '左肘', 8: '右肘',
    9: '左腕', 10: '右腕', 11: '左髋', 12: '右髋',
    13: '左膝', 14: '右膝', 15: '左踝', 16: '右踝',
}


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else 'video2/forehand side view4.mp4'
    video = os.path.join(ROOT, video) if not os.path.isabs(video) else video

    from ultralytics import YOLO
    yolo = YOLO(os.path.join(ROOT, 'models', 'yolov8n-pose.pt'))

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    vname = os.path.splitext(os.path.basename(video))[0]
    out_path = os.path.join(ROOT, '学习', '教练报告', f'肘角_{vname}.mp4')

    import subprocess, imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{w}x{h}', '-r', str(fps),
        '-i', 'pipe:0',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
        '-pix_fmt', 'yuv420p', out_path
    ]
    writer = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    elbow_angles = []
    fi = 0
    print(f"处理 {total} 帧...")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        r = yolo.predict(frame, conf=0.5, imgsz=640, classes=[0], verbose=False)

        e_angle = None
        s_angle = None

        if r[0].keypoints is not None and len(r[0].keypoints.xy) > 0:
            kp = r[0].keypoints.xy[0].cpu().numpy()
            if len(kp) == 17 and kp[11][0] > 0 and kp[12][0] > 0:
                # 画骨架
                for (i, j) in SKELETON:
                    p1 = tuple(kp[i].astype(int))
                    p2 = tuple(kp[j].astype(int))
                    if p1[0] > 0 and p2[0] > 0:
                        # 右臂用亮色
                        color = (0, 255, 255) if (i, j) in [(6, 8), (8, 10)] else (0, 255, 128)
                        thickness = 4 if (i, j) in [(6, 8), (8, 10)] else 2
                        cv2.line(frame, p1, p2, color, thickness)

                # 画关键点
                for i, pt in enumerate(kp):
                    if pt[0] > 0 and pt[1] > 0:
                        color = (0, 0, 255) if i in [6, 8, 10] else (255, 200, 0)
                        cv2.circle(frame, tuple(pt.astype(int)), 5, color, -1)

                # 计算肘角 (肩-肘-腕)
                rs, re, rw, rh = kp[6], kp[8], kp[10], kp[12]
                if np.linalg.norm(rs - re) > 15 and np.linalg.norm(re - rw) > 15:
                    e_angle = angle_3pts(rs, re, rw)
                    s_angle = angle_3pts(rh, rs, re)

                    # 画肘角弧线
                    mid_e = re.astype(int)
                    a1 = np.degrees(np.arctan2(rs[1] - re[1], rs[0] - re[0]))
                    a2 = np.degrees(np.arctan2(rw[1] - re[1], rw[0] - re[0]))
                    cv2.ellipse(frame, tuple(mid_e), (35, 35), 0, int(min(a1, a2)), int(max(a1, a2)), (0, 255, 255), 2)

                    # 标注肘角
                    frame = _put_chinese(frame, f'肘角: {e_angle:.0f}°',
                                         (mid_e[0] + 15, mid_e[1] - 10), (0, 255, 255), font_size=22)

                    # 标注肩角
                    mid_s = rs.astype(int)
                    frame = _put_chinese(frame, f'肩角: {s_angle:.0f}°',
                                         (mid_s[0] + 15, mid_s[1] - 10), (255, 100, 255), font_size=18)

        # 顶部信息栏
        t = fi / fps
        ang_str = f'肘角={e_angle:.0f}°' if e_angle else '无数据'
        frame = _put_chinese(frame, f'帧{fi}  {t:.1f}s  {ang_str}',
                             (10, 10), (255, 255, 255), font_size=20)

        if e_angle:
            elbow_angles.append(e_angle)

        writer.stdin.write(frame.tobytes())
        fi += 1
        if fi % 100 == 0:
            print(f"  {fi}/{total} ({fi/total*100:.0f}%)")

    cap.release()
    writer.stdin.close()
    writer.wait()

    arr = np.array(elbow_angles)
    print(f"\n完成! 共 {len(arr)} 帧有效")
    print(f"肘角: 均值={arr.mean():.1f}°  最小={arr.min():.1f}°  最大={arr.max():.1f}°")
    print(f"输出: {out_path}")


if __name__ == '__main__':
    main()
