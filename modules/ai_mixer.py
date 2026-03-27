"""
AI 专业贴唱混音模块 V3.0 (修复版)
=============================
完整的混音引擎，整合所有处理环节
混音流程（专业标准）：
1. 轻微修音 - 自动音高校正（可选）
2. 深度降噪 - 多方案降噪处理
3. 去齿音 - MultibandCompressor 等效实现
4. 低切 - 去除低频噪声
5. 第一次压缩 - 控制动态
6. 第一次EQ - 基础频率调整
7. 第二次压缩 - 精细控制
8. 第二次EQ - 问题频段修正
9. 动态共振控制 - 类似 soothe2 效果
10. 混响 - 空间感
11. 伴奏处理 - 侧链压缩、Mid/Side EQ
12. 轨道混合 - 响度平衡
13. 风格匹配 - Matchering（可选）
14. 母带处理 - 限制器输出
依赖：
- pyworld: 音高检测与校正
- librosa: 音频分析
- pedalboard: 专业音频效果
- pyloudnorm: 响度标准化
- matchering: 风格匹配
- noisereduce: 深度降噪
"""
import os
import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
import warnings
from typing import Dict, Optional, Tuple, List
warnings.filterwarnings('ignore')

# ============================================================================
#                           依赖检测
# ============================================================================
# pyworld 用于音高校正
try:
    import pyworld as pw
    HAS_PYWORLD = True
except ImportError:
    HAS_PYWORLD = False
    print("【提示】pyworld 未安装，轻微修音功能将受限")

# noisereduce 用于深度降噪
try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False
    print("【提示】noisereduce 未安装，将使用备选降噪方案")

# matchering 用于风格匹配
try:
    import matchering as mg
    HAS_MATCHERING = True
except ImportError:
    HAS_MATCHERING = False
    print("【提示】Matchering 未安装，风格匹配功能将受限")

# pedalboard 用于专业音频效果
try:
    from pedalboard import (
        Pedalboard, Compressor, Reverb, HighpassFilter, LowpassFilter,
        PeakFilter, LowShelfFilter, HighShelfFilter, Limiter, NoiseGate,
        Delay, Gain, Clipping, MultibandCompressor
    )
    HAS_PEDALBOARD = True
    
    # DeEsser 兼容性检测 - Windows 版 pedalboard 0.9.22 存在打包问题
    try:
        from pedalboard import DeEsser
        HAS_DEESSER = True
    except ImportError:
        DeEsser = None
        HAS_DEESSER = False
        print("【提示】DeEsser 不可用，将使用 MultibandCompressor 等效实现")
except ImportError:
    HAS_PEDALBOARD = False
    HAS_DEESSER = False
    print("【警告】Pedalboard 未安装，音频效果处理将受限")

# scipy 用于信号处理
try:
    from scipy import signal
    from scipy.ndimage import uniform_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# 配置导入（请确保 config.py 存在）
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import SAMPLE_RATE, PROJECT_DIR
except ImportError:
    # 兜底配置
    SAMPLE_RATE = 48000
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
#                           混音配置
# ============================================================================
class MixConfig:
    """专业混音配置参数"""
    
    # 采样率
    SAMPLE_RATE = 48000
    
    # ==================== 响度标准 ====================
    TARGET_LOUDNESS_LUFS = -14.0      # 流媒体平台标准
    VOCAL_TO_ACC_OFFSET = 1.5         # 人声比伴奏响 1.5dB
    
    # ==================== 前期处理 ====================
    LOW_CUT_HZ = 80.0                 # 人声低切频率
    
    # 去齿音参数
    DEESS_THRESHOLD = -25.0
    DEESS_FREQUENCY = 6000.0
    DEESS_RATIO = 4.0
    
    # ==================== 第一次压缩 ====================
    COMP1_THRESHOLD = -18.0
    COMP1_RATIO = 3.0
    COMP1_ATTACK = 10.0
    COMP1_RELEASE = 100.0
    
    # ==================== 第一次 EQ ====================
    EQ1_LOW_SHELF_HZ = 250.0
    EQ1_LOW_SHELF_GAIN = -1.5
    EQ1_PRESENCE_HZ = 3000.0
    EQ1_PRESENCE_GAIN = 2.0
    EQ1_AIR_HZ = 10000.0
    EQ1_AIR_GAIN = 2.0
    
    # ==================== 第二次压缩 ====================
    COMP2_THRESHOLD = -24.0
    COMP2_RATIO = 2.0
    COMP2_ATTACK = 5.0
    COMP2_RELEASE = 150.0
    
    # ==================== 第二次 EQ（问题频段）====================
    EQ2_PROBLEM_FREQS = [
        (200, -1.0, 2.0, "去除闷音"),
        (1000, -0.5, 1.5, "减少鼻音"),
        (4000, -1.0, 2.0, "减少刺耳"),
    ]
    
    # ==================== 动态共振控制 ====================
    RESONANCE_THRESHOLD = -30.0
    RESONANCE_SMOOTH = 0.7
    
    # ==================== 混响 ====================
    REVERB_ROOM_SIZE = 0.3
    REVERB_DAMPING = 0.5
    REVERB_WET_LEVEL = 0.15
    
    # ==================== 限制器 ====================
    LIMITER_THRESHOLD = -0.5

# ============================================================================
#                           工具函数
# ============================================================================
def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """确保音频是单声道"""
    if audio.ndim > 1:
        return np.mean(audio, axis=0)
    return audio

def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    """确保音频是立体声"""
    if audio.ndim == 1:
        return np.vstack([audio, audio])
    elif audio.shape[0] == 1:
        return np.vstack([audio[0], audio[0]])
    return audio

def pedalboard_process_mono(board, audio: np.ndarray, sr: int) -> np.ndarray:
    """使用 Pedalboard 处理单声道音频"""
    if audio.ndim > 1:
        return board(audio, sr)
    
    result = board(audio, sr)
    
    if result.ndim > 1:
        result = np.mean(result, axis=0)
    
    return result

def detect_key(audio: np.ndarray, sr: int) -> str:
    """检测音频调性"""
    try:
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_idx = np.argmax(chroma_mean)
        return key_names[key_idx]
    except:
        return 'C'

# ============================================================================
#                           音高校正模块
# ============================================================================
class PitchCorrector:
    """轻微修音模块 - 不影响原本声音质感"""
    
    def __init__(self, sr: int = MixConfig.SAMPLE_RATE):
        self.sr = sr
        self.key_frequencies = self._build_scale_frequencies()
    
    def _build_scale_frequencies(self) -> Dict[str, List[float]]:
        """构建各调式的标准频率"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        frequencies = {}
        
        for note in notes:
            frequencies[note] = []
            for octave in range(-1, 9):
                note_idx = notes.index(note)
                midi_note = (octave + 1) * 12 + note_idx
                freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
                frequencies[note].append(freq)
        
        return frequencies
    
    def _get_key_notes(self, key: str) -> List[str]:
        """获取调内音符（大调）"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        major_intervals = [2, 2, 1, 2, 2, 2, 1]
        
        key_idx = notes.index(key) if key in notes else 0
        key_notes = [notes[key_idx]]
        
        current_idx = key_idx
        for interval in major_intervals[:-1]:
            current_idx = (current_idx + interval) % 12
            key_notes.append(notes[current_idx])
        
        return key_notes
    
    def get_nearest_note_frequency(self, freq: float, key: str = 'C') -> float:
        """获取最近的调内音符频率"""
        if freq <= 0:
            return freq
        
        key_notes = self._get_key_notes(key)
        min_diff = float('inf')
        nearest_freq = freq
        
        for note in key_notes:
            for note_freq in self.key_frequencies.get(note, []):
                diff = abs(freq - note_freq)
                if diff < min_diff:
                    min_diff = diff
                    nearest_freq = note_freq
        
        return nearest_freq
    
    def correct_pitch(self, audio: np.ndarray, 
                      key: str = 'C',
                      correction_strength: float = 0.3) -> np.ndarray:
        """
        轻微修音
        
        参数:
            audio: 输入音频
            key: 调性
            correction_strength: 校正强度 (0.0-1.0)，越小越自然
            
        返回:
            校正后的音频
        """
        if not HAS_PYWORLD:
            print("  ⚠️ pyworld 未安装，跳过修音")
            return audio
        
        print(f"【轻微修音】调性: {key}大调, 强度: {correction_strength}")
        
        try:
            audio = audio.astype(np.float64)
            
            # 使用 pyworld 提取音高
            _f0, t = pw.dio(audio, self.sr)
            f0 = pw.stonemask(audio, _f0, t, self.sr)
            
            # 提取频谱包络和非周期性
            sp = pw.cheaptrick(audio, f0, t, self.sr)
            ap = pw.d4c(audio, f0, t, self.sr)
            
            # 校正音高
            corrected_f0 = f0.copy()
            
            for i, freq in enumerate(f0):
                if freq > 0:
                    target_freq = self.get_nearest_note_frequency(freq, key)
                    corrected_f0[i] = freq + (target_freq - freq) * correction_strength
            
            # 使用校正后的参数重新合成
            corrected = pw.synthesize(corrected_f0, sp, ap, self.sr)
            
            # 长度对齐
            if len(corrected) > len(audio):
                corrected = corrected[:len(audio)]
            elif len(corrected) < len(audio):
                corrected = np.pad(corrected, (0, len(audio) - len(corrected)))
            
            print("  ✅ 轻微修音完成")
            return corrected.astype(np.float32)
            
        except Exception as e:
            print(f"  ⚠️ 修音失败: {str(e)}")
            return audio

# ============================================================================
#                           动态共振控制模块
# ============================================================================
class ResonanceController:
    """动态共振控制 - 类似 soothe2 效果"""
    
    def __init__(self, sr: int = MixConfig.SAMPLE_RATE):
        self.sr = sr
    
    def process(self, audio: np.ndarray, 
                threshold_db: float = -30.0,
                smoothing: float = 0.7) -> np.ndarray:
        """自动检测并衰减刺耳的共振频率"""
        if not HAS_SCIPY:
            print("  ⚠️ scipy 未安装，跳过动态共振控制")
            return audio
        
        print("【动态共振控制】处理中...")
        
        try:
            n_fft = 2048
            hop_length = n_fft // 4
            
            D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(D)
            
            # 计算每个频段的平均能量和标准差
            freq_energy = np.mean(magnitude, axis=1, keepdims=True)
            freq_std = np.std(magnitude, axis=1, keepdims=True)
            freq_mean = np.mean(magnitude, axis=1, keepdims=True)
            
            # 创建动态增益曲线
            gain = np.ones_like(magnitude)
            
            for i in range(magnitude.shape[0]):
                dynamic_threshold = freq_mean[i] + 2 * freq_std[i]
                
                mask = magnitude[i:i+1] > dynamic_threshold
                excess = magnitude[i:i+1] - dynamic_threshold
                gain[i:i+1] = np.where(mask, 
                                       1 - (excess / magnitude[i:i+1]) * smoothing,
                                       1)
            
            gain = np.clip(gain, 0.1, 1.0)
            
            D_processed = D * gain
            processed = librosa.istft(D_processed, hop_length=hop_length, length=len(audio))
            
            print("  ✅ 动态共振控制完成")
            return processed
            
        except Exception as e:
            print(f"  ⚠️ 动态共振控制失败: {str(e)}")
            return audio

# ============================================================================
#                           完整人声处理链
# ============================================================================
class VocalProcessor:
    """完整人声处理链"""
    
    def __init__(self, sr: int = MixConfig.SAMPLE_RATE):
        self.sr = sr
        self.pitch_corrector = PitchCorrector(sr)
        self.resonance_controller = ResonanceController(sr)
    
    def deep_denoise(self, audio: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """深度降噪 - 多方案实现"""
        print(f"【深度降噪】处理中 (强度: {strength})...")
        
        # 方案1: noisereduce
        if HAS_NOISEREDUCE:
            try:
                cleaned = nr.reduce_noise(
                    y=audio, sr=self.sr,
                    prop_decrease=strength,
                    stationary=False
                )
                if len(cleaned) > len(audio):
                    cleaned = cleaned[:len(audio)]
                elif len(cleaned) < len(audio):
                    cleaned = np.pad(cleaned, (0, len(audio) - len(cleaned)))
                print("  ✅ 深度降噪完成 (noisereduce)")
                return cleaned
            except Exception as e:
                print(f"  ⚠️ noisereduce 失败: {str(e)}，尝试备选方案")
        
        # 方案2: scipy 频谱门限
        if HAS_SCIPY:
            try:
                f, t, Zxx = signal.stft(audio, fs=self.sr, nperseg=1024, noverlap=768)
                magnitude = np.abs(Zxx)
                noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
                threshold = noise_floor * (1 + strength * 3)
                
                mask = magnitude > threshold
                gain = np.where(mask, 1.0, 0.1 + 0.9 * (magnitude / threshold) ** 2)
                
                Zxx_clean = Zxx * gain
                _, cleaned = signal.istft(Zxx_clean, fs=self.sr, nperseg=1024, noverlap=768)
                
                if len(cleaned) > len(audio):
                    cleaned = cleaned[:len(audio)]
                elif len(cleaned) < len(audio):
                    cleaned = np.pad(cleaned, (0, len(audio) - len(cleaned)))
                
                print("  ✅ 深度降噪完成 (scipy 频谱门限)")
                return cleaned
            except Exception as e:
                print(f"  ⚠️ scipy 频谱门限失败: {str(e)}")
        
        # 方案3: NoiseGate
        if HAS_PEDALBOARD:
            try:
                board = Pedalboard([
                    NoiseGate(
                        threshold_db=-40 + strength * 20,
                        ratio=10,
                        attack_ms=0.1,
                        release_ms=50
                    )
                ])
                cleaned = pedalboard_process_mono(board, audio, self.sr)
                print("  ✅ 深度降噪完成 (NoiseGate)")
                return cleaned
            except Exception as e:
                print(f"  ⚠️ NoiseGate 失败: {str(e)}")
        
        print("  ⚠️ 所有降噪方案均失败，返回原始音频")
        return audio
    
    def apply_deessing(self, audio: np.ndarray,
                       threshold_db: float = MixConfig.DEESS_THRESHOLD,
                       frequency: float = MixConfig.DEESS_FREQUENCY,
                       ratio: float = MixConfig.DEESS_RATIO) -> np.ndarray:
        """去齿音处理"""
        if not HAS_PEDALBOARD:
            print("  ⚠️ Pedalboard 不可用，跳过去齿音")
            return audio
        
        print("【去齿音】处理中...")
        
        try:
            # 方案1: 原生 DeEsser
            if HAS_DEESSER:
                board = Pedalboard([DeEsser(threshold_db=threshold_db, frequency=frequency)])
                processed = pedalboard_process_mono(board, audio, self.sr)
                print("  ✅ 去齿音完成 (DeEsser)")
                return processed
            
            # 方案2: MultibandCompressor
            board = Pedalboard([
                MultibandCompressor(
                    low_band_threshold_db=-12.0,
                    low_band_ratio=2.0,
                    low_band_attack_ms=15.0,
                    low_band_release_ms=150.0,
                    
                    mid_low_band_threshold_db=-15.0,
                    mid_low_band_ratio=2.5,
                    mid_low_band_attack_ms=10.0,
                    mid_low_band_release_ms=100.0,
                    
                    mid_high_band_threshold_db=threshold_db,
                    mid_high_band_ratio=ratio,
                    mid_high_band_attack_ms=1.0,
                    mid_high_band_release_ms=50.0,
                    
                    high_band_threshold_db=threshold_db + 3,
                    high_band_ratio=3.0,
                    high_band_attack_ms=2.0,
                    high_band_release_ms=80.0,
                )
            ])
            processed = pedalboard_process_mono(board, audio, self.sr)
            print("  ✅ 去齿音完成 (MultibandCompressor)")
            return processed
            
        except Exception as e:
            print(f"  ⚠️ 去齿音失败: {str(e)}")
            return audio
    
    def apply_low_cut(self, audio: np.ndarray) -> np.ndarray:
        """低切滤波"""
        if not HAS_PEDALBOARD:
            return audio
        
        print(f"【低切】频率: {MixConfig.LOW_CUT_HZ} Hz")
        
        try:
            board = Pedalboard([HighpassFilter(cutoff_frequency_hz=MixConfig.LOW_CUT_HZ)])
            return pedalboard_process_mono(board, audio, self.sr)
        except Exception as e:
            print(f"  ⚠️ 低切失败: {str(e)}")
            return audio
    
    def apply_first_compression(self, audio: np.ndarray) -> np.ndarray:
        """第一次压缩 - 控制动态"""
        if not HAS_PEDALBOARD:
            return audio
        
        print(f"【第一次压缩】阈值: {MixConfig.COMP1_THRESHOLD} dB, 比例: {MixConfig.COMP1_RATIO}:1")
        
        try:
            board = Pedalboard([
                Compressor(
                    threshold_db=MixConfig.COMP1_THRESHOLD,
                    ratio=MixConfig.COMP1_RATIO,
                    attack_ms=MixConfig.COMP1_ATTACK,
                    release_ms=MixConfig.COMP1_RELEASE
                )
            ])
            return pedalboard_process_mono(board, audio, self.sr)
        except Exception as e:
            print(f"  ⚠️ 第一次压缩失败: {str(e)}")
            return audio
    
    def apply_first_eq(self, audio: np.ndarray) -> np.ndarray:
        """第一次 EQ - 基础频率调整"""
        if not HAS_PEDALBOARD:
            return audio
        
        print("【第一次EQ】基础频率调整")
        
        try:
            board = Pedalboard([
                LowShelfFilter(
                    cutoff_frequency_hz=MixConfig.EQ1_LOW_SHELF_HZ,
                    gain_db=MixConfig.EQ1_LOW_SHELF_GAIN
                ),
                PeakFilter(
                    cutoff_frequency_hz=MixConfig.EQ1_PRESENCE_HZ,
                    gain_db=MixConfig.EQ1_PRESENCE_GAIN,
                    q=1.0
                ),
                HighShelfFilter(
                    cutoff_frequency_hz=MixConfig.EQ1_AIR_HZ,
                    gain_db=MixConfig.EQ1_AIR_GAIN
                ),
            ])
            return pedalboard_process_mono(board, audio, self.sr)
        except Exception as e:
            print(f"  ⚠️ 第一次EQ失败: {str(e)}")
            return audio
    
    def apply_second_compression(self, audio: np.ndarray) -> np.ndarray:
        """第二次压缩 - 精细控制"""
        if not HAS_PEDALBOARD:
            return audio
        
        print(f"【第二次压缩】阈值: {MixConfig.COMP2_THRESHOLD} dB, 比例: {MixConfig.COMP2_RATIO}:1")
        
        try:
            board = Pedalboard([
                Compressor(
                    threshold_db=MixConfig.COMP2_THRESHOLD,
                    ratio=MixConfig.COMP2_RATIO,
                    attack_ms=MixConfig.COMP2_ATTACK,
                    release_ms=MixConfig.COMP2_RELEASE
                )
            ])
            return pedalboard_process_mono(board, audio, self.sr)
        except Exception as e:
            print(f"  ⚠️ 第二次压缩失败: {str(e)}")
            return audio
    
    def apply_second_eq(self, audio: np.ndarray) -> np.ndarray:
        """第二次 EQ - 精细频率调整"""
        if not HAS_PEDALBOARD:
            return audio
        
        print("【第二次EQ】精细频率调整")
        
        try:
            effects = []
            for freq, gain, q, desc in MixConfig.EQ2_PROBLEM_FREQS:
                print(f"  - {desc}: {freq} Hz, {gain} dB")
                effects.append(PeakFilter(cutoff_frequency_hz=freq, gain_db=gain, q=q))
            
            board = Pedalboard(effects)
            return pedalboard_process_mono(board, audio, self.sr)
        except Exception as e:
            print(f"  ⚠️ 第二次EQ失败: {str(e)}")
            return audio
    
    def apply_reverb(self, audio: np.ndarray) -> np.ndarray:
        """应用混响"""
        if not HAS_PEDALBOARD:
            return audio
        
        print(f"【混响】房间大小: {MixConfig.REVERB_ROOM_SIZE}, 湿声: {MixConfig.REVERB_WET_LEVEL*100}%")
        
        try:
            board = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=200),
                LowpassFilter(cutoff_frequency_hz=8000),
                Reverb(
                    room_size=MixConfig.REVERB_ROOM_SIZE,
                    damping=MixConfig.REVERB_DAMPING,
                    wet_level=MixConfig.REVERB_WET_LEVEL,
                    dry_level=1 - MixConfig.REVERB_WET_LEVEL
                )
            ])
            return pedalboard_process_mono(board, audio, self.sr)
        except Exception as e:
            print(f"  ⚠️ 混响失败: {str(e)}")
            return audio
    
    def process_vocal(self, audio: np.ndarray,
                      key: str = 'C',
                      enable_pitch_correction: bool = True,
                      pitch_correction_strength: float = 0.3,
                      enable_denoise: bool = True) -> np.ndarray:
        """
        完整人声处理链
        
        流程:
        1. 轻微修音（可选）
        2. 深度降噪
        3. 去齿音
        4. 低切
        5. 第一次压缩
        6. 第一次EQ
        7. 第二次压缩
        8. 第二次EQ
        9. 动态共振控制
        10. 混响
        """
        print("\n" + "="*50)
        print("【人声处理链】开始")
        print("="*50)
        
        result = audio.copy()
        
        # 1. 轻微修音（可选）
        if enable_pitch_correction:
            result = self.pitch_corrector.correct_pitch(result, key, pitch_correction_strength)
        
        # 2. 深度降噪
        if enable_denoise:
            result = self.deep_denoise(result, strength=0.5)
        
        # 3. 去齿音
        result = self.apply_deessing(result)
        
        # 4. 低切
        result = self.apply_low_cut(result)
        
        # 5. 第一次压缩
        result = self.apply_first_compression(result)
        
        # 6. 第一次EQ
        result = self.apply_first_eq(result)
        
        # 7. 第二次压缩
        result = self.apply_second_compression(result)
        
        # 8. 第二次EQ
        result = self.apply_second_eq(result)
        
        # 9. 动态共振控制
        result = self.resonance_controller.process(result)
        
        # 10. 混响
        result = self.apply_reverb(result)
        
        print("【人声处理链】完成\n")
        return result

# ============================================================================
#                           伴奏处理模块
# ============================================================================
class AccompanimentProcessor:
    """伴奏处理器"""
    
    def __init__(self, sr: int = MixConfig.SAMPLE_RATE):
        self.sr = sr
    
    def apply_mid_side_eq(self, audio: np.ndarray) -> np.ndarray:
        """Mid/Side EQ - 为人声让出空间"""
        print("【伴奏 Mid/Side EQ】为人声让出空间")
        
        if audio.ndim == 1:
            audio = np.vstack([audio, audio])
        
        try:
            # 转换为 Mid/Side
            mid = (audio[0] + audio[1]) / 2
            side = (audio[0] - audio[1]) / 2
            
            # 在 Mid 通道挖一个坑给人声
            if HAS_PEDALBOARD:
                mid_board = Pedalboard([
                    PeakFilter(cutoff_frequency_hz=2000, gain_db=-2.0, q=0.7)
                ])
                mid = mid_board(mid, self.sr)
                if mid.ndim > 1:
                    mid = np.mean(mid, axis=0)
                
                # Side 通道稍微提升增加宽度
                side_board = Pedalboard([
                    HighShelfFilter(cutoff_frequency_hz=8000, gain_db=1.0)
                ])
                side = side_board(side, self.sr)
                if side.ndim > 1:
                    side = np.mean(side, axis=0)
            
            # 转回立体声
            left = mid + side
            right = mid - side
            
            print("  ✅ Mid/Side EQ 完成")
            return np.vstack([left, right])
            
        except Exception as e:
            print(f"  ⚠️ Mid/Side EQ 失败: {str(e)}")
            return audio
    
    def apply_sidechain(self, accompaniment: np.ndarray, 
                        vocal: np.ndarray,
                        reduction_db: float = 3.0) -> np.ndarray:
        """侧链压缩 - 人声响时降低伴奏"""
        print(f"【侧链压缩】最大衰减: {reduction_db} dB")
        
        try:
            vocal_mono = ensure_mono(vocal)
            
            frame_size = int(self.sr * 0.01)
            hop_size = frame_size // 2
            
            vocal_envelope = librosa.feature.rms(
                y=vocal_mono,
                frame_length=frame_size,
                hop_length=hop_size
            )[0]
            
            if HAS_SCIPY:
                vocal_envelope = uniform_filter1d(vocal_envelope, size=10)
            
            vocal_envelope = vocal_envelope / (np.max(vocal_envelope) + 1e-10)
            
            gain_curve = 1 - vocal_envelope * (1 - 10 ** (-reduction_db / 20))
            
            acc_len = accompaniment.shape[1] if accompaniment.ndim > 1 else len(accompaniment)
            
            gain_interpolated = np.interp(
                np.arange(acc_len),
                np.linspace(0, acc_len, len(gain_curve)),
                gain_curve
            )
            
            if accompaniment.ndim > 1:
                processed = accompaniment * gain_interpolated[np.newaxis, :]
            else:
                processed = accompaniment * gain_interpolated
            
            print("  ✅ 侧链压缩完成")
            return processed
            
        except Exception as e:
            print(f"  ⚠️ 侧链压缩失败: {str(e)}")
            return accompaniment
    
    def process_accompaniment(self, accompaniment: np.ndarray,
                              vocal: np.ndarray,
                              enable_sidechain: bool = True) -> np.ndarray:
        """完整伴奏处理"""
        print("\n" + "="*50)
        print("【伴奏处理链】开始")
        print("="*50)
        
        result = accompaniment.copy()
        
        if result.ndim == 1:
            result = np.vstack([result, result])
        
        # Mid/Side EQ
        result = self.apply_mid_side_eq(result)
        
        # 侧链压缩
        if enable_sidechain and vocal is not None:
            result = self.apply_sidechain(result, vocal)
        
        print("【伴奏处理链】完成\n")
        return result

# ============================================================================
#                           风格匹配模块
# ============================================================================
class StyleMatcher:
    """使用 Matchering 进行风格匹配和母带处理"""
    
    def __init__(self, sr: int = MixConfig.SAMPLE_RATE):
        self.sr = sr
    
    def match_style(self, target_path: str, reference_path: str, 
                    output_path: str) -> bool:
        """使用 Matchering 进行风格匹配"""
        if not HAS_MATCHERING:
            print("  ⚠️ Matchering 不可用，跳过风格匹配")
            return False
        
        print("【风格匹配】使用 Matchering 进行深度风格迁移...")
        print(f"  目标: {os.path.basename(target_path)}")
        print(f"  参考: {os.path.basename(reference_path)}")
        
        try:
            mg.log(print)
            mg.process(
                target=target_path,
                reference=reference_path,
                results=[mg.pcm24(output_path)],
            )
            print("  ✅ 风格匹配完成")
            return True
        except Exception as e:
            print(f"  ⚠️ 风格匹配失败: {str(e)}")
            return False

# ============================================================================
#                           混音引擎
# ============================================================================
class AIMixEngine:
    """
    AI 混音引擎 V3.0
    ================
    整合所有处理环节的完整混音引擎
    """
    
    def __init__(self, sr: int = MixConfig.SAMPLE_RATE):
        self.sr = sr
        self.vocal_processor = VocalProcessor(sr)
        self.acc_processor = AccompanimentProcessor(sr)
        self.style_matcher = StyleMatcher(sr)
    
    def balance_volumes(self, vocal: np.ndarray, 
                        accompaniment: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """音量平衡"""
        print("【音量平衡】")
        
        try:
            meter = pyln.Meter(self.sr)
            
            acc_stereo = ensure_stereo(accompaniment)
            acc_lufs = meter.integrated_loudness(acc_stereo.T)
            if np.isinf(acc_lufs) or np.isnan(acc_lufs):
                acc_lufs = -18.0
            
            target_vocal_lufs = acc_lufs + MixConfig.VOCAL_TO_ACC_OFFSET
            
            vocal_stereo_temp = ensure_stereo(vocal)
            vocal_lufs = meter.integrated_loudness(vocal_stereo_temp.T)
            
            if not np.isinf(vocal_lufs) and not np.isnan(vocal_lufs):
                gain_db = target_vocal_lufs - vocal_lufs
                vocal = vocal * (10 ** (gain_db / 20))
                print(f"  人声响度调整: {gain_db:.1f} dB")
            
            return vocal, accompaniment
            
        except Exception as e:
            print(f"  ⚠️ 音量平衡失败: {str(e)}")
            return vocal, accompaniment
    
    def mix_tracks(self, vocal: np.ndarray, accompaniment: np.ndarray) -> np.ndarray:
        """混合轨道"""
        print("【轨道混合】")
        
        vocal_mono = ensure_mono(vocal)
        acc_stereo = ensure_stereo(accompaniment)
        vocal_stereo = np.vstack([vocal_mono, vocal_mono])
        
        min_len = min(acc_stereo.shape[1], vocal_stereo.shape[1])
        
        print(f"  人声长度: {vocal_stereo.shape[1]} 样本 ({vocal_stereo.shape[1]/self.sr:.1f}秒)")
        print(f"  伴奏长度: {acc_stereo.shape[1]} 样本 ({acc_stereo.shape[1]/self.sr:.1f}秒)")
        print(f"  混合长度: {min_len} 样本 ({min_len/self.sr:.1f}秒)")
        
        mix = acc_stereo[:, :min_len] + vocal_stereo[:, :min_len]
        
        print("  ✅ 轨道混合完成")
        return mix
    
    def apply_bus_processing(self, mix: np.ndarray) -> np.ndarray:
        """总线处理"""
        print("【总线处理】")
        
        if not HAS_PEDALBOARD:
            return mix
        
        try:
            board = Pedalboard([
                Compressor(threshold_db=-18, ratio=1.5, attack_ms=20, release_ms=150),
            ])
            processed = board(mix, self.sr)
            print("  ✅ 总线处理完成")
            return processed
        except Exception as e:
            print(f"  ⚠️ 总线处理失败: {str(e)}")
            return mix
    
    def apply_mastering(self, mix: np.ndarray) -> np.ndarray:
        """母带处理"""
        print("【母带处理】")
        
        try:
            meter = pyln.Meter(self.sr)
            current_lufs = meter.integrated_loudness(mix.T)
            
            if not np.isinf(current_lufs) and not np.isnan(current_lufs):
                gain_db = MixConfig.TARGET_LOUDNESS_LUFS - current_lufs
                mix = mix * (10 ** (gain_db / 20))
                print(f"  响度调整: {gain_db:.1f} dB")
            
            if HAS_PEDALBOARD:
                board = Pedalboard([
                    Limiter(threshold_db=MixConfig.LIMITER_THRESHOLD)
                ])
                mix = board(mix, self.sr)
            
            print("  ✅ 母带处理完成")
            return mix
            
        except Exception as e:
            print(f"  ⚠️ 母带处理失败: {str(e)}")
            return mix

# ============================================================================
#                           对外接口（修复版）
# ============================================================================
def ai_mix_with_original(original_song_path: str,
                         user_acc_path: str,
                         user_vocal_path: str,
                         project_id: str,
                         options: dict = None) -> dict:
    """
    AI 混音主接口 V3.0 (修复版)
    
    完整混音流程：
    1. 加载音频
    2. 检测调性
    3. 处理人声（修音、降噪、去齿音、压缩、EQ、共振控制、混响）
    4. 处理伴奏（Mid/Side EQ、侧链压缩）
    5. 混合轨道
    6. 总线处理
    7. 风格匹配（可选）
    8. 母带处理
    
    参数:
        original_song_path: 原曲文件路径（用于风格参考，可选）
        user_acc_path: 用户伴奏文件路径
        user_vocal_path: 用户干声文件路径（需用户自行对齐）
        project_id: 工程ID
        options: 混音选项
            - enable_pitch_correction: 是否启用修音（默认True）
            - pitch_correction_strength: 修音强度（默认0.3）
            - enable_denoise: 是否启用降噪（默认True）
            - enable_sidechain: 是否启用侧链压缩（默认True）
    
    返回:
        {"code": 0/-1, "result_file": "输出文件名", "msg": "消息", "stats": {...}}
    """
    # 默认选项
    default_options = {
        "enable_pitch_correction": True,
        "pitch_correction_strength": 0.3,
        "enable_denoise": True,
        "enable_sidechain": True,
    }
    if options:
        default_options.update(options)
    options = default_options
    
    try:
        print(f"\n{'='*60}")
        print(f"🚀 【AI混音 V3.0】工程 {project_id} 启动...")
        print(f"{'='*60}")
        
        # ===== 第一步：加载音频 =====
        print("\n【步骤 1/7】资源加载...")
        
        user_vocal, _ = librosa.load(user_vocal_path, sr=MixConfig.SAMPLE_RATE)
        print(f"  干声加载完成: {len(user_vocal)/MixConfig.SAMPLE_RATE:.1f}秒")
        
        user_acc, _ = librosa.load(user_acc_path, sr=MixConfig.SAMPLE_RATE, mono=False)
        if user_acc.ndim == 1:
            user_acc = np.vstack([user_acc, user_acc])
        print(f"  伴奏加载完成: {user_acc.shape[1]/MixConfig.SAMPLE_RATE:.1f}秒")
        
        # ===== 第二步：检测调性 =====
        print("\n【步骤 2/7】检测调性...")
        key = detect_key(ensure_mono(user_acc), MixConfig.SAMPLE_RATE)
        print(f"  检测到调性: {key}大调")
        
        # ===== 第三步：处理人声 =====
        print("\n【步骤 3/7】处理人声...")
        engine = AIMixEngine(sr=MixConfig.SAMPLE_RATE)
        processed_vocal = engine.vocal_processor.process_vocal(
            user_vocal,
            key=key,
            enable_pitch_correction=options["enable_pitch_correction"],
            pitch_correction_strength=options["pitch_correction_strength"],
            enable_denoise=options["enable_denoise"]
        )
        
        # ===== 第四步：处理伴奏 =====
        print("\n【步骤 4/7】处理伴奏...")
        processed_acc = engine.acc_processor.process_accompaniment(
            user_acc,
            processed_vocal,
            enable_sidechain=options["enable_sidechain"]
        )
        
        # ===== 第五步：混合轨道 =====
        print("\n【步骤 5/7】混合轨道...")
        engine.balance_volumes(processed_vocal, processed_acc)
        mix = engine.mix_tracks(processed_vocal, processed_acc)
        
        # ===== 第六步：总线处理 =====
        print("\n【步骤 6/7】总线处理...")
        mix = engine.apply_bus_processing(mix)
        
        # ===== 第七步：导出与母带处理 =====
        print("\n【步骤 7/7】导出与母带处理...")
        
        temp_mix_path = os.path.join(PROJECT_DIR, f"{project_id}_mix_temp.wav")
        result_file_name = f"{project_id}_ProMix.wav"
        result_save_path = os.path.join(PROJECT_DIR, result_file_name)
        
        sf.write(temp_mix_path, mix.T, MixConfig.SAMPLE_RATE, subtype="PCM_24")
        
        # 尝试风格匹配
        style_matched = False
        if HAS_MATCHERING and original_song_path and os.path.exists(original_song_path):
            style_matched = engine.style_matcher.match_style(temp_mix_path, original_song_path, result_save_path)
        
        # 如果风格匹配失败，回退到基础母带处理
        if not style_matched:
            mix = engine.apply_mastering(mix)
            sf.write(result_save_path, mix.T, MixConfig.SAMPLE_RATE, subtype="PCM_24")
        
        # 清理临时文件
        if os.path.exists(temp_mix_path):
            os.remove(temp_mix_path)
        
        # 测量最终结果
        final_audio, _ = librosa.load(result_save_path, sr=MixConfig.SAMPLE_RATE, mono=False)
        if final_audio.ndim == 1:
            final_audio = np.vstack([final_audio, final_audio])
        
        meter = pyln.Meter(MixConfig.SAMPLE_RATE)
        final_lufs = meter.integrated_loudness(final_audio.T)
        final_peak = 20 * np.log10(np.max(np.abs(final_audio)) + 1e-10)
        
        print(f"\n{'='*60}")
        print(f"✅ 【AI混音完成】")
        print(f"{'='*60}")
        print(f"  📁 输出文件: {result_file_name}")
        print(f"  📊 最终响度: {final_lufs:.1f} LUFS")
        print(f"  📊 峰值电平: {final_peak:.1f} dB")
        print(f"  📊 采样率: {MixConfig.SAMPLE_RATE} Hz")
        print(f"{'='*60}\n")
        
        # 🔥 关键修复：将 numpy 类型转为 Python 原生类型
        return {
            "code": 0,
            "result_file": result_file_name,
            "msg": "混音完成",
            "stats": {
                "lufs": float(final_lufs),      # 转为原生 float
                "peak_db": float(final_peak),   # 转为原生 float
                "sample_rate": int(MixConfig.SAMPLE_RATE), # 转为原生 int
                "key": key
            }
        }
        
    except Exception as e:
        print(f"\n❌ 【AI混音失败】: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "code": -1,
            "result_file": "",
            "msg": str(e),
            "stats": {
                "lufs": 0.0,
                "peak_db": 0.0,
                "sample_rate": 0,
                "key": "C"
            }
        }

# ============================================================================
#                           导出
# ============================================================================
__all__ = [
    'AIMixEngine',
    'MixConfig',
    'VocalProcessor',
    'AccompanimentProcessor',
    'PitchCorrector',
    'ResonanceController',
    'StyleMatcher',
    'ai_mix_with_original'
]