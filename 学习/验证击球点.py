"""
验证击球点检测：对每次正手，导出击球点前后几帧的对比图。
红框=我判定的击球帧。你看红框那帧是不是真正触球瞬间。
输出到 学习/击球点验证/
"""
import numpy as np, os, sys, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from 正手教练 import (ForehandLSTM, extract_cached, detect_segments,
    smooth2d, moving_average, find_contact_frame, angle_3pts,
    _draw_skeleton, _put_chinese, SKELETON, ROOT, SEQ, SMOOTH_K)

VIDEO = sys.argv[1] if len(sys.argv) > 1 else 'video2/forehand_00014.mp4'
OFFSETS = [-6, -4, -2, 0, 2, 4, 6]  # 相对击球点的帧偏移

path = os.path.join(ROOT, VIDEO)
model = ForehandLSTM()
model.load_state_dict(torch.load(os.path.join(ROOT, 'models', 'forehand_lstm.pth'), map_location='cpu'))
model.eval()
norm = np.load(os.path.join(ROOT, 'models', 'forehand_lstm_norm.npz'))
mean, std = norm['mean'], norm['std']

feats, angles, ang_valid, n, raw_kp = extract_cached(path, mean, std)
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

out_dir = os.path.join(ROOT, '学习', '击球点验证')
# 每次运行先清空旧图, 只保留最新一版, 避免版本混淆
if os.path.isdir(out_dir):
    for f in os.listdir(out_dir):
        if f.lower().endswith('.jpg'):
            try:
                os.remove(os.path.join(out_dir, f))
            except OSError:
                pass
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(path)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def read_frame(idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    return fr if ok else None


count = 0
for rank, (s, pk, e) in enumerate(segments, 1):
    if ang_valid[s:e+1].mean() < 0.5:
        continue
    wrist_orig = raw_kp[s:e+1, 10]
    elbow_orig = smooth2d(angles[s:e+1], k=5)[:, 0]
    cf = s + find_contact_frame(wrist_orig, elbow_orig, center=pk-s)  # 全局击球帧(pk窗口约束)
    tiles = []
    for off in OFFSETS:
        fidx = cf + off
        if fidx < 0 or fidx >= total:
            continue
        fr = read_frame(fidx)
        if fr is None:
            continue
        is_contact = (off == 0)
        fr = _draw_skeleton(fr, raw_kp[fidx], highlight=is_contact)
        # 缩小到高度360便于拼接
        h, w = fr.shape[:2]
        scale = 360.0 / h
        fr = cv2.resize(fr, (int(w*scale), 360))
        # 标注帧号和偏移
        el = angles[fidx, 0]
        label = f"帧{fidx} ({off:+d})  肘{el:.0f}"
        col = (0, 255, 255) if is_contact else (255, 255, 255)
        fr = _put_chinese(fr, label, (5, 5), col, font_size=20)
        if is_contact:
            cv2.rectangle(fr, (0, 0), (fr.shape[1]-1, fr.shape[0]-1), (0, 0, 255), 6)
            fr = _put_chinese(fr, "★击球", (5, 32), (0, 0, 255), font_size=22)
        tiles.append(fr)
    if not tiles:
        continue
    strip = np.hstack(tiles)
    out_path = os.path.join(out_dir, f"stroke_{rank:02d}_contact{cf}.jpg")
    cv2.imencode('.jpg', strip)[1].tofile(out_path)  # 支持中文路径
    count += 1
    print(f"第{rank}次: 击球帧={cf} (段{s}-{e}) → {os.path.basename(out_path)}")

cap.release()
print(f"\n共导出 {count} 张对比图到: {out_dir}")
print("红框那帧是我判定的击球点，请看是否为真正触球瞬间。")
