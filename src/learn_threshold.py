"""
从训练视频统计学习分阶段肘角阈值
输出: forehand_phase_thresholds.npz (引拍/击球阈值)
"""
import numpy as np, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from 正手教练 import (ForehandLSTM, extract_cached, detect_segments,
    smooth2d, resample, moving_average, elbow_phases,
    ROOT, SEQ, WIN, SMOOTH_K)

VIDEOS = [
    'video2/dj.mp4', 'video2/forehand_00005.mp4', 'video2/forehand_00010.mp4',
    'video2/forehand_00013.mp4', 'video2/forehand_00014.mp4', 'video2/forehand_00015.mp4',
    'video2/forehand side view1.mp4', 'video2/forehand side view2.mp4',
    'video2/forehand side view3.mp4', 'video2/forehand side view4.mp4',
]

model = ForehandLSTM()
model.load_state_dict(torch.load(os.path.join(ROOT, 'models', 'forehand_lstm.pth'), map_location='cpu'))
model.eval()
norm = np.load(os.path.join(ROOT, 'models', 'forehand_lstm_norm.npz'))
mean, std = norm['mean'], norm['std']

backs, contacts, follows = [], [], []
for video in VIDEOS:
    path = os.path.join(ROOT, video)
    if not os.path.exists(path):
        print(f'跳过(不存在): {video}'); continue
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
    vb, vc, vf = [], [], []
    for s, pk, e in segments:
        if ang_valid[s:e+1].mean() < 0.5: continue
        elbow_orig = smooth2d(angles[s:e+1], k=5)[:, 0]
        wrist_orig = raw_kp[s:e+1, 10]
        be, ct, fb, cr = elbow_phases(elbow_orig, wrist_orig, center=pk-s)  # pk定位击球点
        if not (0.15 <= cr <= 0.90): continue  # 只用完整挥拍学习
        vb.append(be); vc.append(ct); vf.append(fb)
    if vb:
        backs += vb; contacts += vc; follows += vf
        print(f'{os.path.basename(video):25s}: {len(vb):3d}段  引拍{np.mean(vb):.0f} 击球{np.mean(vc):.0f} 随挥{np.mean(vf):.0f}')

backs = np.array(backs); contacts = np.array(contacts); follows = np.array(follows)
print(f'\n=== 汇总 {len(backs)} 段完整挥拍 ===')
print(f'引拍伸展: 均值{backs.mean():.0f} 中位{np.median(backs):.0f} 10分位{np.percentile(backs,10):.0f} 25分位{np.percentile(backs,25):.0f}')
print(f'击球肘角: 均值{contacts.mean():.0f} 中位{np.median(contacts):.0f} 10分位{np.percentile(contacts,10):.0f} 25分位{np.percentile(contacts,25):.0f}')
print(f'随挥弯曲: 均值{follows.mean():.0f} 中位{np.median(follows):.0f}')

# 阈值: 引拍/击球都取25分位(低于此为伸展不足, 因击球时手臂应伸展)
t_back = float(np.percentile(backs, 25))
t_contact = float(np.percentile(contacts, 25))
np.savez(os.path.join(ROOT, 'forehand_phase_thresholds.npz'),
         t_back=t_back, t_contact=t_contact,
         back_mean=backs.mean(), contact_mean=contacts.mean(), follow_mean=follows.mean())
print(f'\n学习到的阈值: 引拍伸展下限={t_back:.0f}°  击球肘角下限={t_contact:.0f}°')
print(f'已保存到 forehand_phase_thresholds.npz')
