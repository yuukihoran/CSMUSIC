import os
import numpy as np
import soundfile as sf
import librosa
import torch
from demucs import pretrained
from demucs.apply import apply_model
from config import DEVICE, DEMUCS_SR, MAX_AUDIO_LENGTH, UPLOAD_DIR, PROJECT_DIR

# 预加载拆分模型
print("【歌曲拆分模块】正在加载Demucs模型...")
splitter_model = pretrained.get_model("htdemucs").to(DEVICE)
print("【歌曲拆分模块】模型加载完成！")

def split_song(song_file_path, project_id):
    """
    拆分歌曲为伴奏和原唱人声
    :param song_file_path: 原曲文件路径
    :param project_id: 工程ID
    :return: 拆分后的伴奏文件名、人声文件名
    """
    try:
        print(f"【歌曲拆分】工程 {project_id} 开始处理")
        # 加载音频
        song_y, sr = librosa.load(song_file_path, sr=DEMUCS_SR, mono=False)
        if song_y.ndim == 1:
            song_y = np.stack([song_y, song_y])
        
        # 限制长度
        max_length = min(int(DEMUCS_SR * MAX_AUDIO_LENGTH), len(song_y[0]))
        song_y = song_y[:, :max_length]
        
        # 转Tensor送入GPU
        song_tensor = torch.from_numpy(song_y).float().to(DEVICE).unsqueeze(0)
        
        # AI分离
        with torch.no_grad():
            tracks = apply_model(splitter_model, song_tensor, shifts=1, split=True)[0]
        
        # 提取轨道：0-鼓 1-贝斯 2-其他 3-人声
        original_vocal = tracks[3].cpu().numpy()
        instrumental_acc = (tracks[0] + tracks[1] + tracks[2]).cpu().numpy()
        
        # 保存文件
        acc_file_name = f"{project_id}_纯伴奏.wav"
        vocal_file_name = f"{project_id}_原唱人声.wav"
        acc_save_path = os.path.join(PROJECT_DIR, acc_file_name)
        vocal_save_path = os.path.join(PROJECT_DIR, vocal_file_name)
        
        sf.write(acc_save_path, instrumental_acc.T, DEMUCS_SR)
        sf.write(vocal_save_path, original_vocal.T, DEMUCS_SR)
        
        print(f"【歌曲拆分】工程 {project_id} 处理完成")
        return {
            "code": 0,
            "acc_file": acc_file_name,
            "vocal_file": vocal_file_name,
            "msg": "拆分成功"
        }
    except Exception as e:
        print(f"【歌曲拆分】失败：{str(e)}")
        return {
            "code": -1,
            "msg": f"拆分失败：{str(e)}"
        }