import cv2
import csv
import os
from ultralytics import YOLO
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(ROOT, 'models', 'yolov8n-pose.pt'))

# 你的视频所在文件夹路径
video_folder = r'C:\Users\18518\Desktop\vscode\tennis movement\video'
print("文件夹存在:", os.path.exists(video_folder))
print("文件列表:")
for f in os.listdir(video_folder):
    print(f)
all_rows = []
header = ['frame', 'label'] + [f'kp{i}_{c}' for i in range(17) for c in ['x', 'y']]

for video_file in sorted(os.listdir(video_folder)):
    if not video_file.endswith('.mp4'):
        continue
    
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    extracted = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        results = model.predict(frame, conf=0.5, imgsz=640, classes=[0], verbose=False)
        
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            kps = results[0].keypoints.xy[0].tolist()
            if len(kps) == 17:
                row = [frame_idx, 'forehand']
                for kp in kps:
                    row += [kp[0], kp[1]]
                all_rows.append(row)
                extracted += 1
        
        frame_idx += 1
    
    cap.release()
    print(f"{video_file}: {extracted} 帧")

# 存成一个CSV
with open(os.path.join(ROOT, 'data', 'keypoints_forehand_all.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(all_rows)

print(f"\n完成！总共 {len(all_rows)} 帧，存到 keypoints_forehand_all.csv")