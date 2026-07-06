"""
正手 AI 教练 — Gradio Web 界面
==============================
上传侧面正手视频 → 自动分析 → 返回:
  1. 评分汇总图
  2. 标注视频（骨架+评分+诊断）
  3. 逐次正手的文字报告

启动: python 学习/app_coach.py
"""

import os
import sys
import tempfile
import gradio as gr

# 确保能导入正手教练模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from 正手教练 import (
    ForehandLSTM, extract_cached, detect_segments,
    smooth2d, resample, dtw_distance, dtw_to_score,
    generate_summary, generate_annotated_video, moving_average,
    calc_swing_speed, export_stroke_clips, elbow_phase_tips, find_contact_frame,
    get_fps, swing_travel, SWING_MIN_TRAVEL,
    ROOT, FPS, SEQ, WIN, SMOOTH_K
)


def load_model():
    model = ForehandLSTM()
    model.load_state_dict(torch.load(os.path.join(ROOT, 'models', 'forehand_lstm.pth'), map_location='cpu'))
    model.eval()
    norm = np.load(os.path.join(ROOT, 'models', 'forehand_lstm_norm.npz'))
    return model, norm['mean'], norm['std']


MODEL, MEAN, STD = load_model()
TEMPLATE = resample(np.load(os.path.join(ROOT, 'models', 'forehand_template_clean.npy')), WIN)
TSTD = resample(np.load(os.path.join(ROOT, 'models', 'forehand_template_clean_std.npy')), WIN)


# 示例视频（选一个较短的）
EXAMPLE_DIR = os.path.join(ROOT, 'video2')
EXAMPLE_VIDEOS = [
    os.path.join(EXAMPLE_DIR, 'forehand side view4.mp4'),
    os.path.join(EXAMPLE_DIR, 'forehand_00015.mp4'),
]
EXAMPLE_VIDEOS = [v for v in EXAMPLE_VIDEOS if os.path.exists(v)]


# 全局状态：保存当前分析的片段列表
_current_clips = []


def analyze(video_path):
    """核心分析流程，用 generator yield 实时更新报告文本框"""
    global _current_clips
    _current_clips = []
    if video_path is None:
        yield None, None, "请上传视频", gr.update(choices=[], value=None), None
        return

    import uuid, gc
    gc.collect()                     # 释放上次分析残留的文件句柄
    uid = uuid.uuid4().hex[:8]
    # 用二进制读写复制，避免 shutil.copy2 在 Windows 上复制 ACL 导致锁文件
    ext = os.path.splitext(video_path)[1]
    tmp_dir = os.path.join(tempfile.gettempdir(), f'coach_{uid}')
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_video = os.path.join(tmp_dir, f'vid_{uid}{ext}')
    with open(video_path, 'rb') as src:
        data = src.read()
    with open(tmp_video, 'wb') as dst:
        dst.write(data)
    del data
    video_path = tmp_video

    yield None, None, "⏳ ① 提取关键点中（首次较慢）...", gr.update(), None
    feats, angles, ang_valid, n, raw_kp = extract_cached(video_path, MEAN, STD)
    fps = get_fps(video_path)  # 用视频真实帧率, 保证时间/挥速正确

    yield None, None, f"⏳ ② LSTM 检测正手段落... ({n} 帧)", gr.update(), None
    probs = np.zeros(n)
    wins = np.array([feats[i:i+SEQ] for i in range(0, n - SEQ + 1)], dtype=np.float32)
    with torch.no_grad():
        out = []
        for b in range(0, len(wins), 512):
            out.append(torch.softmax(MODEL(torch.tensor(wins[b:b+512])), dim=1)[:, 1])
        p = torch.cat(out).numpy()
    for i, pv in enumerate(p):
        probs[i + SEQ // 2] = pv
    probs = moving_average(probs, SMOOTH_K)
    segments = detect_segments(probs, elbow_angles=angles[:, 0])

    yield None, None, f"⏳ ③ 分析 {len(segments)} 段正手...", gr.update(), None
    results = []
    lines = []
    for rank, (s, pk, e) in enumerate(segments, 1):
        if ang_valid[s:e+1].mean() < 0.5:
            lines.append(f"第{rank}次({pk/fps:.1f}s): 角度有效率过低，跳过")
            continue
        if swing_travel(raw_kp, s, e) < SWING_MIN_TRAVEL:
            lines.append(f"第{rank}次({pk/fps:.1f}s): 无明显挥拍动作，跳过")
            continue
        user = smooth2d(resample(angles[s:e+1], WIN), k=5)
        dist = dtw_distance(TEMPLATE, user)
        score = dtw_to_score(dist)
        e_diff = user[:, 0].mean() - TEMPLATE[:, 0].mean()
        s_diff = user[:, 1].mean() - TEMPLATE[:, 1].mean()

        # 相位诊断：用原始帧的肘角+手腕(挥速定位真正击球点)
        elbow_orig = smooth2d(angles[s:e+1], k=5)[:, 0]
        wrist_orig = raw_kp[s:e+1, 10]
        center = pk - s  # LSTM峰值(接近击球点)作为搜索中心
        etips, (back_ext, contact, follow_bend), valid = elbow_phase_tips(elbow_orig, wrist_orig, center=center)
        contact_frame = s + find_contact_frame(wrist_orig, elbow_orig, center=center)  # 真正击球帧(全局)
        tips = list(etips) if valid else []  # 击球点在边缘时不给分阶段建议
        if abs(s_diff) > 12:
            tips.append(f"肩部{'偏低' if s_diff < 0 else '偏高'}({s_diff:+.0f}°)")
        phase_str = f"引拍{back_ext:.0f}°/击球{contact:.0f}°/随挥{follow_bend:.0f}°"
        tip_str = '; '.join(tips) if tips else '动作标准'

        peak_spd, avg_spd, _ = calc_swing_speed(raw_kp, s, e, fps=fps)

        results.append({
            'rank': rank, 'start': s, 'peak': pk, 'end': e,
            'score': score, 'dtw': dist,
            'e_diff': e_diff, 's_diff': s_diff, 'tips': tip_str,
            'phase': phase_str, 'contact_frame': contact_frame,
            'peak_speed': peak_spd, 'avg_speed': avg_spd
        })
        spd_str = f"  挥速{peak_spd:.0f}px/s" if peak_spd > 0 else ""
        lines.append(f"第{rank}次({pk/fps:.1f}s): {score}分{spd_str}  肘角[{phase_str}] — {tip_str}")

    if not results:
        yield None, None, "未检测到有效正手动作。请确保视频为侧面视角、分辨率足够。", gr.update(choices=[], value=None), None
        return

    scores = [r['score'] for r in results]
    header = f"共 {len(results)} 次正手 | 平均 {np.mean(scores):.0f} 分 | 最高 {max(scores)} | 最低 {min(scores)}\n"
    header += "=" * 50 + "\n"
    report = header + "\n".join(lines)

    yield None, None, "⏳ ④ 生成汇总图...", gr.update(), None
    summary_path = os.path.join(tmp_dir, f'summary.png')
    generate_summary(results, summary_path, f'分析_{uid}')

    yield summary_path, None, "⏳ ⑤ 生成标注视频（H.264）...", gr.update(), None
    vid_out = os.path.join(tmp_dir, f'annotated.mp4')
    generate_annotated_video(video_path, raw_kp, results, vid_out)

    yield summary_path, vid_out, "⏳ ⑥ 裁剪正手片段...", gr.update(), None
    clips_dir = os.path.join(tmp_dir, 'clips')
    clips = export_stroke_clips(video_path, raw_kp, angles, results, clips_dir)
    _current_clips = clips
    choices = [c[1] for c in clips]
    first_clip = clips[0][0] if clips else None

    yield summary_path, vid_out, report, gr.update(choices=choices, value=choices[0] if choices else None), first_clip


def on_clip_select(choice):
    """用户选择某次正手时，返回对应片段视频"""
    for path, label in _current_clips:
        if label == choice:
            return path
    return None


# ── Gradio 界面 ──
with gr.Blocks(title="正手 AI 教练") as demo:
    gr.Markdown("# 🎾 正手 AI 教练")
    gr.Markdown("上传一段**侧面**正手视频，AI 自动检测每次正手并评分。\n"
                "支持 mp4/avi/mov，建议侧面机位、单人、分辨率 ≥ 480p。")

    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(label="上传视频")
            btn = gr.Button("🚀 开始分析", variant="primary", size="lg")

        with gr.Column(scale=2):
            summary_img = gr.Image(label="评分汇总", type="filepath")
            video_out = gr.Video(label="标注视频（骨架+评分）")
            report = gr.Textbox(label="分析报告", lines=12, max_lines=25)

    gr.Markdown("### 🎬 逐次正手回放")
    with gr.Row():
        clip_dropdown = gr.Dropdown(label="选择正手", choices=[], interactive=True)
        clip_video = gr.Video(label="片段回放")

    clip_dropdown.change(fn=on_clip_select, inputs=[clip_dropdown], outputs=[clip_video])

    if EXAMPLE_VIDEOS:
        gr.Markdown("### 📂 示例视频（点击直接加载）")
        gr.Examples(
            examples=[[v] for v in EXAMPLE_VIDEOS],
            inputs=[video_in],
            label="",
        )

    btn.click(fn=analyze, inputs=[video_in],
              outputs=[summary_img, video_out, report, clip_dropdown, clip_video])

demo.launch(server_name="127.0.0.1", server_port=7870, share=False)
