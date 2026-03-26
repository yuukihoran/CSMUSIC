import os
import torch

# 项目基础信息
PROJECT_NAME = "川时的音乐工坊"
VERSION = "1.0.0"

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
PROJECT_DIR = os.path.join(DATA_DIR, "projects")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# 自动创建文件夹
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# 设备配置：优先GPU，自动检测CUDA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"【{PROJECT_NAME}】使用设备: {DEVICE.upper()}")
if DEVICE == "cuda":
    print(f"【{PROJECT_NAME}】GPU型号: {torch.cuda.get_device_name(0)}")

# 音频参数
SAMPLE_RATE = 48000
DEMUCS_SR = 44100
MAX_AUDIO_LENGTH = 600  # 最长支持10分钟音频