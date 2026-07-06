import subprocess
import os

# 创建video文件夹如果不存在
os.makedirs("./video", exist_ok=True)

# 视频URL列表
urls = [
    "https://youtu.be/KLzaaln6Rf0",
    "https://youtu.be/SFPjhs0wPGA",
    "https://youtu.be/Oy0JOe3dLOQ",
    "https://youtu.be/Nw_2I2ksX3U"
]

# 下载每个视频
for url in urls:
    print(f"正在下载: {url}")
    subprocess.run([
        "yt-dlp",
        "-f", "best",
        "-o", "./video/%(title)s.%(ext)s",
        url
    ])
    print(f"下载完成: {url}\n")

print("所有视频下载完成！")
