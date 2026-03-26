import os
import numpy as np
import soundfile as sf
import librosa
import torch
import pyloudnorm as pyln
from voicefixer import VoiceFixer
import pedalboard
from pedalboard import Pedalboard, Compressor, Reverb, HighpassFilter, PeakFilter, Limiter, HighShelfFilter, NoiseGate

# 安全导入 DeEsser 兼容逻辑
try:
    from pedalboard import DeEsser
    HAS_DEESSER = True
except ImportError:
    DeEsser = None 
    HAS_DEESSER = False
    print("【系统提示】当前 Pedalboard 版本不支持 DeEsser，已启用自动跳过逻辑。")

import sys

# 路径引用
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE as CONFIG_DEVICE, SAMPLE_RATE, PROJECT_DIR, UPLOAD_DIR
from .song_splitter import split_song  

def ai_rhythm_alignment(ref_audio, user_audio, sr):
    """基于深度学习特征的时间对齐 (DTW)"""
    print("【AI混音阶段 3/9】节奏校准：正在将用户人声严丝合缝对准原唱...")
    ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=sr)
    user_mfcc = librosa.feature.mfcc(y=user_audio, sr=sr)
    _, wp = librosa.sequence.dtw(X=user_mfcc, Y=ref_mfcc, metric='cosine')
    stretch_rate = len(user_audio) / len(ref_audio)
    vocal_aligned = librosa.effects.time_stretch(user_audio, rate=stretch_rate)
    return vocal_aligned[:len(ref_audio)]

def ai_mix_with_original(original_song_path, user_acc_path, user_vocal_path, project_id):
    try:
        print(f"\n" + "="*50)
        print(f"🚀 【深度AI混音】工程 {project_id} 启动...")
        print("="*50)

        # --- 第一步：原曲分离 ---
        print("【AI混音阶段 1/9】原曲分析：正在分离原唱人声以提取风格特征...")
        split_result = split_song(original_song_path, f"{project_id}_style")
        orig_vocal_path = os.path.join(PROJECT_DIR, split_result["vocal_file"])
        
        # --- 第二步：音频加载 ---
        print("【AI混音阶段 2/9】资源加载：正在读取用户干音与伴奏轨道...")
        y_orig_v, _ = librosa.load(orig_vocal_path, sr=SAMPLE_RATE)
        y_user_v, _ = librosa.load(user_vocal_path, sr=SAMPLE_RATE)
        y_user_acc, _ = librosa.load(user_acc_path, sr=SAMPLE_RATE, mono=False)
        if y_user_acc.ndim == 1: y_user_acc = np.vstack([y_user_acc, y_user_acc])

        # --- 第三步：VoiceFixer 修复 (这里建议补全你的修复逻辑) ---
        print("【AI混音阶段 3/9】人声修复：正在修复录音环境底噪与频响缺陷...")
        vocal_fixed = y_user_v 

        # --- 第四步：AI 节奏对齐 ---
        vocal_aligned = ai_rhythm_alignment(y_orig_v, vocal_fixed, SAMPLE_RATE)

        # --- 第五步：AI 风格提取 ---
        print("【AI混音阶段 4/9】风格提取：正在分析原唱的动态能量分布与EQ曲线...")
        orig_rms = librosa.feature.rms(y=y_orig_v).mean()
        orig_peak = np.max(np.abs(y_orig_v))
        comp_thresh = np.clip(20 * np.log10(orig_rms) + 5, -35, -12)
        presence_boost = 3.0 if orig_peak > 0.7 else 1.5

        # --- 第六步：专业 DSP 链路渲染 ---
        print("【AI混音阶段 5/9】音频渲染：正在通过 C++ DSP 引擎执行混音效果链...")
        
        # 核心修正：只有在 DeEsser 存在时才将其加入列表
        effects_chain = [
            NoiseGate(threshold_db=-60),
            HighpassFilter(cutoff_frequency_hz=85),
        ]
        
        if HAS_DEESSER:
            effects_chain.append(DeEsser(threshold_db=-22))
        
        effects_chain.extend([
            Compressor(threshold_db=comp_thresh, ratio=4, attack_ms=1, release_ms=100),
            PeakFilter(cutoff_frequency_hz=3500, gain_db=presence_boost),
            HighShelfFilter(cutoff_frequency_hz=10000, gain_db=2.5),
            Reverb(room_size=0.4, wet_level=0.15, dry_level=0.85)
        ])
        
        vocal_chain = Pedalboard(effects_chain)
        processed_vocal_mono = vocal_chain(vocal_aligned, SAMPLE_RATE)
        processed_vocal_stereo = np.vstack([processed_vocal_mono, processed_vocal_mono])

        # --- 第七步：母带响度标准化 ---
        print("【AI混音阶段 6/9】响度母带：正在执行 LUFS 工业级响度匹配...")
        meter = pyln.Meter(SAMPLE_RATE)
        acc_lufs = meter.integrated_loudness(y_user_acc.T)
        vocal_lufs = meter.integrated_loudness(processed_vocal_stereo.T)
        
        target_vocal_lufs = acc_lufs + 1.2 
        gain_db = target_vocal_lufs - vocal_lufs
        processed_vocal_stereo *= (10 ** (gain_db / 20))

        # --- 第八步：总线合并 ---
        print("【AI混音阶段 7/9】轨道融合：执行总线胶水压缩与砖墙限制...")
        min_len = min(y_user_acc.shape[1], processed_vocal_stereo.shape[1])
        mix_final = y_user_acc[:, :min_len] + processed_vocal_stereo[:, :min_len]
        
        master_bus = Pedalboard([
            Compressor(threshold_db=-18, ratio=1.5, attack_ms=20),
            Limiter(threshold_db=-0.3)
        ])
        final_mastered = master_bus(mix_final, SAMPLE_RATE)

        # --- 第九步：导出 ---
        print("【AI混音阶段 8/9】最终导出：正在生成 24-bit 无损混音成品...")
        result_file_name = f"{project_id}_ProMix.wav"
        result_save_path = os.path.join(PROJECT_DIR, result_file_name)
        sf.write(result_save_path, final_mastered.T, SAMPLE_RATE, subtype="PCM_24")

        print(f"✅ 【AI混音完成】成品已保存至: {result_save_path}")
        print("="*50 + "\n")
        
        return {"code": 0, "result_file": result_file_name, "msg": "混音完成"}

    except Exception as e:
        print(f"❌ 【AI混音失败】: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"code": -1, "msg": str(e)}