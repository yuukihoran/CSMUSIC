"""
专业贴唱混音模块 V3.0
=====================
基于专业混音师工作流程的完整混音引擎

混音流程（按顺序）：
1. 对轨修音 - 自动音高校正（可选）
2. 前期处理 - 低切、去齿音
3. 第一次压缩 - 控制动态
4. 第一次EQ - 基础频率调整
5. 第二次压缩 - 精细控制
6. 第二次EQ - 精细频率调整
7. 动态共振控制 - 类似 soothe2 效果
8. 混响延迟 - 空间感
9. 限制器 - 最终输出

依赖：
- pyworld: 音高检测与校正
- librosa: 音频分析
- pedalboard: 专业音频效果
- pyloudnorm: 响度标准化
- scipy: 信号处理
"""

import os
import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
import warnings
from typing import Dict, Optional, Tuple, List
warnings.filterwarnings('ignore')

# 尝试导入 pyworld 用于音高校正
try:
    import pyworld as pw
    HAS_PYWORLD = True
except ImportError:
    HAS_PYWORLD = False
    print("【提示】pyworld 未安装，轻微修音功能将受限")

# 尝试导入 pedalboard
try:
    from pedalboard import (
        Pedalboard, Compressor, Reverb, HighpassFilter, LowpassFilter,
        PeakFilter, LowShelfFilter, HighShelfFilter, Limiter, NoiseGate,
        Delay, Gain, Clipping, MultibandCompressor
    )
    HAS_PEDALBOARD = True
    
    # DeEsser 兼容性检测
    try:
        from pedalboard import DeEsser
        HAS_DEESSER = True
    except ImportError:
        DeEsser = None
        HAS_DEESSER = False
except ImportError:
    HAS_PEDALBOARD = False
    HAS_DEESSER = False
    print("【警告】Pedalboard 未安装，音频效果处理将受限")

# 尝试导入 scipy
try:
    from scipy import signal
    from scipy.ndimage import uniform_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE, PROJECT_DIR


# ============================================================================
#                           混音配置
# ============================================================================

class ProMixConfig:
    """专业混音配置参数"""
    
    # 采样率
    SAMPLE_RATE = 48000
    
    # ==================== 响度标准 ====================
    TARGET_LOUDNESS_LUFS = -14.0  # 流媒体平台标准
    VOCAL_TO_ACC_OFFSET = 1.5     # 人声比伴奏响 1.5dB
    
    # ==================== 前期处理 ====================
    # 低切
    LOW_CUT_HZ = 80.0             # 人声低切频率
    
    # 去齿音
    DEESS_THRESHOLD = -25.0       # 去齿音阈值
    DEESS_FREQUENCY = 6000.0      # 齿音中心频率
    DEESS_RATIO = 4.0             # 压缩比
    
    # ==================== 第一次压缩 ====================
    # 控制：力度活跃 或 温暖稳定（场景选择）
    COMP1_THRESHOLD = -18.0       # 阈值
    COMP1_RATIO = 3.0             # 压缩比
    COMP1_ATTACK = 10.0           # 起始时间 (ms)
    COMP1_RELEASE = 100.0         # 释放时间 (ms)
    
    # ==================== 第一次 EQ ====================
    # 基础频率调整
    EQ1_LOW_SHELF_HZ = 250.0      # 低频架中心
    EQ1_LOW_SHELF_GAIN = -1.5     # 低频衰减 (去闷)
    EQ1_PRESENCE_HZ = 3000.0      # 穿透力频段
    EQ1_PRESENCE_GAIN = 2.0       # 穿透力增益
    EQ1_AIR_HZ = 10000.0          # 空气感频段
    EQ1_AIR_GAIN = 2.0            # 空气感增益
    
    # ==================== 第二次压缩 ====================
    # 精细控制
    COMP2_THRESHOLD = -24.0       # 阈值（更低）
    COMP2_RATIO = 2.0             # 压缩比（更温和）
    COMP2_ATTACK = 5.0            # 起始时间
    COMP2_RELEASE = 150.0         # 释放时间
    
    # ==================== 第二次 EQ ====================
    # 精细频率调整（问题频段）
    EQ2_PROBLEM_FREQS = [
        # (频率, 增益, Q值, 描述)
        (200, -1.0, 2.0, "去除闷音"),
        (1000, -0.5, 1.5, "减少鼻音"),
        (4000, -1.0, 2.0, "减少刺耳"),
    ]
    
    # ==================== 动态共振控制 ====================
    # 类似 soothe2 效果
    RESONANCE_THRESHOLD = -30.0   # 共振检测阈值
    RESONANCE_SMOOTH = 0.7        # 平滑系数
    
    # ==================== 混响 ====================
    REVERB_ROOM_SIZE = 0.3        # 房间大小
    REVERB_DAMPING = 0.5          # 阻尼
    REVERB_WET_LEVEL = 0.15       # 湿声比例
    REVERB_DRY_LEVEL = 0.85       # 干声比例
    REVERB_PREDELAY_MS = 20.0     # 预延迟
    
    # ==================== 延迟 ====================
    DELAY_DELAY_SECONDS = 0.375   # 延迟时间（八分音符）
    DELAY_FEEDBACK = 0.3          # 反馈
    DELAY_MIX = 0.2               # 混合比例
    
    # ==================== 限制器 ====================
    LIMITER_THRESHOLD = -0.5      # 限制器阈值
    
    # ==================== 多段压缩（伴奏处理）====================
    MULTIBAND_CROSSOVERS = [200, 2000, 8000]  # 频段分割点


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


def detect_bpm(audio: np.ndarray, sr: int) -> float:
    """检测音频 BPM"""
    try:
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        return float(tempo)
    except:
        return 120.0  # 默认 BPM


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
    
    def __init__(self, sr: int = ProMixConfig.SAMPLE_RATE):
        self.sr = sr
        self.key_frequencies = self._build_scale_frequencies()
    
    def _build_scale_frequencies(self) -> Dict[str, List[float]]:
        """构建各调式的标准频率"""
        # A4 = 440Hz
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        frequencies = {}
        
        for note in notes:
            frequencies[note] = []
            for octave in range(-1, 9):
                # 计算每个音符的频率
                note_idx = notes.index(note)
                midi_note = (octave + 1) * 12 + note_idx
                freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
                frequencies[note].append(freq)
        
        return frequencies
    
    def get_nearest_note_frequency(self, freq: float, key: str = 'C') -> float:
        """获取最近的调内音符频率"""
        if freq <= 0:
            return freq
        
        # 获取调内音符
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
    
    def _get_key_notes(self, key: str) -> List[str]:
        """获取调内音符（大调）"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # 大调音程：全全半全全全半
        major_intervals = [2, 2, 1, 2, 2, 2, 1]
        
        key_idx = notes.index(key) if key in notes else 0
        key_notes = [notes[key_idx]]
        
        current_idx = key_idx
        for interval in major_intervals[:-1]:
            current_idx = (current_idx + interval) % 12
            key_notes.append(notes[current_idx])
        
        return key_notes
    
    def correct_pitch(self, audio: np.ndarray, 
                      key: str = 'C',
                      correction_strength: float = 0.3,
                      retune_speed: float = 50.0) -> np.ndarray:
        """
        轻微修音
        
        参数:
            audio: 输入音频
            key: 调性
            correction_strength: 校正强度 (0.0-1.0)，越小越自然
            retune_speed: 校正速度 (ms)，越大越自然
            
        返回:
            校正后的音频
        """
        if not HAS_PYWORLD:
            print("  ⚠️ pyworld 未安装，跳过修音")
            return audio
        
        print(f"【轻微修音】调性: {key}大调, 强度: {correction_strength}")
        
        try:
            # 确保是 float64 类型
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
                if freq > 0:  # 只处理有声段
                    # 获取目标频率
                    target_freq = self.get_nearest_note_frequency(freq, key)
                    
                    # 根据校正强度调整
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
    """
    动态共振控制 - 类似 soothe2 效果
    自动检测并衰减刺耳的共振频率
    """
    
    def __init__(self, sr: int = ProMixConfig.SAMPLE_RATE):
        self.sr = sr
    
    def process(self, audio: np.ndarray, 
                threshold_db: float = -30.0,
                smoothing: float = 0.7) -> np.ndarray:
        """
        动态共振控制
        
        参数:
            audio: 输入音频
            threshold_db: 检测阈值 (dB)
            smoothing: 平滑系数 (0-1)
            
        返回:
            处理后的音频
        """
        if not HAS_SCIPY:
            print("  ⚠️ scipy 未安装，跳过动态共振控制")
            return audio
        
        print("【动态共振控制】处理中...")
        
        try:
            # 短时傅里叶变换
            n_fft = 2048
            hop_length = n_fft // 4
            
            D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(D)
            phase = np.angle(D)
            
            # 计算阈值（线性）
            threshold = 10 ** (threshold_db / 20) * np.max(magnitude)
            
            # 计算每个频段的平均能量
            freq_energy = np.mean(magnitude, axis=1, keepdims=True)
            
            # 检测共振峰值
            # 使用标准差来识别异常频率
            freq_std = np.std(magnitude, axis=1, keepdims=True)
            freq_mean = np.mean(magnitude, axis=1, keepdims=True)
            
            # 创建动态增益曲线
            # 当某频段能量异常高时，衰减该频段
            gain = np.ones_like(magnitude)
            
            for i in range(magnitude.shape[0]):
                # 计算该频段的动态阈值
                dynamic_threshold = freq_mean[i] + 2 * freq_std[i]
                
                # 应用增益衰减
                mask = magnitude[i:i+1] > dynamic_threshold
                excess = magnitude[i:i+1] - dynamic_threshold
                gain[i:i+1] = np.where(mask, 
                                       1 - (excess / magnitude[i:i+1]) * smoothing,
                                       1)
            
            # 确保增益不为负
            gain = np.clip(gain, 0.1, 1.0)
            
            # 应用增益
            D_processed = D * gain
            
            # 逆变换
            processed = librosa.istft(D_processed, hop_length=hop_length, length=len(audio))
            
            print("  ✅ 动态共振控制完成")
            return processed
            
        except Exception as e:
            print(f"  ⚠️ 动态共振控制失败: {str(e)}")
            return audio


# ============================================================================
#                           专业人声处理链
# ============================================================================

class VocalChain:
    """
    专业人声处理链
    
    流程：
    1. 低切 - 去除低频噪声
    2. 去齿音 - 衰减刺耳齿音
    3. 第一次压缩 - 控制动态
    4. 第一次EQ - 基础频率调整
    5. 第二次压缩 - 精细控制
    6. 第二次EQ - 精细频率调整
    7. 动态共振控制 - 去除刺耳频率
    8. 混响 - 空间感
    """
    
    def __init__(self, sr: int = ProMixConfig.SAMPLE_RATE):
        self.sr = sr
        self.pitch_corrector = PitchCorrector(sr)
        self.resonance_controller = ResonanceController(sr)
    
    def apply_low_cut(self, audio: np.ndarray, 
                      cutoff_hz: float = ProMixConfig.LOW_CUT_HZ) -> np.ndarray:
        """低切滤波"""
        if not HAS_PEDALBOARD:
            return audio
        
        print(f"【低切】频率: {cutoff_hz} Hz")
        
        try:
            board = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=cutoff_hz)
            ])
            return pedalboard_process_mono(board, audio, self.sr)
        except Exception as e:
            print(f"  ⚠️ 低切失败: {str(e)}")
            return audio
    
    def apply_deessing(self, audio: np.ndarray,
                       threshold_db: float = ProMixConfig.DEESS_THRESHOLD,
                       frequency: float = ProMixConfig.DEESS_FREQUENCY,
                       ratio: float = ProMixConfig.DEESS_RATIO) -> np.ndarray:
        """去齿音处理"""
        if not HAS_PEDALBOARD:
            return audio
        
        print(f"【去齿音】阈值: {threshold_db} dB, 频率: {frequency} Hz")
        
        try:
            # 优先使用原生 DeEsser
            if HAS_DEESSER:
                board = Pedalboard([
                    DeEsser(threshold_db=threshold_db, frequency=frequency)
                ])
                processed = pedalboard_process_mono(board, audio, self.sr)
                print("  ✅ 去齿音完成 (DeEsser)")
                return processed
            
            # 使用 MultibandCompressor 实现等效效果
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
    
    def apply_first_compression(self, audio: np.ndarray,
                                 threshold: float = ProMixConfig.COMP1_THRESHOLD,
                                 ratio: float = ProMixConfig.COMP1_RATIO,
                                 attack: float = ProMixConfig.COMP1_ATTACK,
                                 release: float = ProMixConfig.COMP1_RELEASE) -> np.ndarray:
        """第一次压缩 - 控制动态"""
        if not HAS_PEDALBOARD:
            return audio
        
        print(f"【第一次压缩】阈值: {threshold} dB, 比例: {ratio}:1")
        
        try:
            board = Pedalboard([
                Compressor(
                    threshold_db=threshold,
                    ratio=ratio,
                    attack_ms=attack,
                    release_ms=release
                )
            ])
            return pedalboard_process_mono(board, audio, self.sr)
        except Exception as e:
            print(f"  ⚠️ 压缩失败: {str(e)}")
            return audio
    
    def apply_first_eq(self, audio: np.ndarray) -> np.ndarray:
        """第一次 EQ - 基础频率调整"""
        if not HAS_PEDALBOARD:
            return audio
        
        print("【第一次EQ】基础频率调整")
        
        try:
            board = Pedalboard([
                # 去除闷音
                LowShelfFilter(
                    cutoff_frequency_hz=ProMixConfig.EQ1_LOW_SHELF_HZ,
                    gain_db=ProMixConfig.EQ1_LOW_SHELF_GAIN
                ),
                # 穿透力提升
                PeakFilter(
                    cutoff_frequency_hz=ProMixConfig.EQ1_PRESENCE_HZ,
                    gain_db=ProMixConfig.EQ1_PRESENCE_GAIN,
                    q=1.0
                ),
                # 空气感
                HighShelfFilter(
                    cutoff_frequency_hz=ProMixConfig.EQ1_AIR_HZ,
                    gain_db=ProMixConfig.EQ1_AIR_GAIN
                ),
            ])
            return pedalboard_process_mono(board, audio, self.sr)
        except Exception as e:
            print(f"  ⚠️ EQ失败: {str(e)}")
            return audio
    
    def apply_second_compression(self, audio: np.ndarray,
                                  threshold: float = ProMixConfig.COMP2_THRESHOLD,
                                  ratio: float = ProMixConfig.COMP2_RATIO,
                                  attack: float = ProMixConfig.COMP2_ATTACK,
                                  release: float = ProMixConfig.COMP2_RELEASE) -> np.ndarray:
        """第二次压缩 - 精细控制"""
        if not HAS_PEDALBOARD:
            return audio
        
        print(f"【第二次压缩】阈值: {threshold} dB, 比例: {ratio}:1")
        
        try:
            board = Pedalboard([
                Compressor(
                    threshold_db=threshold,
                    ratio=ratio,
                    attack_ms=attack,
                    release_ms=release
                )
            ])
            return pedalboard_process_mono(board, audio, self.sr)
        except Exception as e:
            print(f"  ⚠️ 压缩失败: {str(e)}")
            return audio
    
    def apply_second_eq(self, audio: np.ndarray,
                        problem_freqs: List[Tuple] = None) -> np.ndarray:
        """第二次 EQ - 精细频率调整"""
        if not HAS_PEDALBOARD:
            return audio
        
        print("【第二次EQ】精细频率调整")
        
        if problem_freqs is None:
            problem_freqs = ProMixConfig.EQ2_PROBLEM_FREQS
        
        try:
            effects = []
            for freq, gain, q, desc in problem_freqs:
                print(f"  - {desc}: {freq} Hz, {gain} dB")
                effects.append(
                    PeakFilter(
                        cutoff_frequency_hz=freq,
                        gain_db=gain,
                        q=q
                    )
                )
            
            board = Pedalboard(effects)
            return pedalboard_process_mono(board, audio, self.sr)
        except Exception as e:
            print(f"  ⚠️ EQ失败: {str(e)}")
            return audio
    
    def apply_reverb(self, audio: np.ndarray,
                     room_size: float = ProMixConfig.REVERB_ROOM_SIZE,
                     wet_level: float = ProMixConfig.REVERB_WET_LEVEL) -> np.ndarray:
        """应用混响"""
        if not HAS_PEDALBOARD:
            return audio
        
        print(f"【混响】房间大小: {room_size}, 湿声: {wet_level*100}%")
        
        try:
            board = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=200),
                LowpassFilter(cutoff_frequency_hz=8000),
                Reverb(
                    room_size=room_size,
                    damping=ProMixConfig.REVERB_DAMPING,
                    wet_level=wet_level,
                    dry_level=1 - wet_level
                )
            ])
            return pedalboard_process_mono(board, audio, self.sr)
        except Exception as e:
            print(f"  ⚠️ 混响失败: {str(e)}")
            return audio
    
    def process_vocal(self, audio: np.ndarray,
                      key: str = 'C',
                      enable_pitch_correction: bool = True,
                      pitch_correction_strength: float = 0.3) -> np.ndarray:
        """
        完整人声处理链
        
        参数:
            audio: 输入人声
            key: 调性
            enable_pitch_correction: 是否启用修音
            pitch_correction_strength: 修音强度 (0-1)
            
        返回:
            处理后的人声
        """
        print("\n" + "="*50)
        print("【人声处理链】开始")
        print("="*50)
        
        result = audio.copy()
        
        # 0. 轻微修音（可选）
        if enable_pitch_correction:
            result = self.pitch_corrector.correct_pitch(
                result, key, pitch_correction_strength
            )
        
        # 1. 低切
        result = self.apply_low_cut(result)
        
        # 2. 去齿音
        result = self.apply_deessing(result)
        
        # 3. 第一次压缩
        result = self.apply_first_compression(result)
        
        # 4. 第一次 EQ
        result = self.apply_first_eq(result)
        
        # 5. 第二次压缩
        result = self.apply_second_compression(result)
        
        # 6. 第二次 EQ
        result = self.apply_second_eq(result)
        
        # 7. 动态共振控制
        result = self.resonance_controller.process(result)
        
        # 8. 混响
        result = self.apply_reverb(result)
        
        print("【人声处理链】完成\n")
        return result


# ============================================================================
#                           伴奏处理模块
# ============================================================================

class AccompanimentChain:
    """
    伴奏处理模块
    
    功能：
    1. Mid/Side EQ - 为人声让出空间
    2. 侧链压缩 - 人声响时降低伴奏
    3. 多段压缩 - 控制动态
    """
    
    def __init__(self, sr: int = ProMixConfig.SAMPLE_RATE):
        self.sr = sr
    
    def apply_mid_side_eq(self, audio: np.ndarray,
                          vocal_duck_freq: float = 2000.0,
                          duck_db: float = -2.0) -> np.ndarray:
        """
        Mid/Side EQ 处理
        在伴奏的 Mid 通道为人声让出空间
        """
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
                    PeakFilter(
                        cutoff_frequency_hz=vocal_duck_freq,
                        gain_db=duck_db,
                        q=0.7
                    )
                ])
                mid = mid_board(mid, self.sr)
                if mid.ndim > 1:
                    mid = mid[0] if mid.shape[0] == 1 else np.mean(mid, axis=0)
            
            # Side 通道稍微提升增加宽度
            if HAS_PEDALBOARD:
                side_board = Pedalboard([
                    HighShelfFilter(
                        cutoff_frequency_hz=8000,
                        gain_db=1.0
                    )
                ])
                side = side_board(side, self.sr)
                if side.ndim > 1:
                    side = side[0] if side.shape[0] == 1 else np.mean(side, axis=0)
            
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
        """
        侧链压缩
        人声响时降低伴奏
        """
        print(f"【侧链压缩】最大衰减: {reduction_db} dB")
        
        try:
            # 确保人声是单声道
            vocal_mono = ensure_mono(vocal)
            
            # 计算人声包络
            frame_size = int(self.sr * 0.01)  # 10ms
            hop_size = frame_size // 2
            
            vocal_envelope = librosa.feature.rms(
                y=vocal_mono,
                frame_length=frame_size,
                hop_length=hop_size
            )[0]
            
            # 平滑包络
            if HAS_SCIPY:
                vocal_envelope = uniform_filter1d(vocal_envelope, size=10)
            
            # 归一化
            vocal_envelope = vocal_envelope / (np.max(vocal_envelope) + 1e-10)
            
            # 创建增益曲线
            gain_curve = 1 - vocal_envelope * (1 - 10 ** (-reduction_db / 20))
            
            # 获取伴奏长度
            acc_len = accompaniment.shape[1] if accompaniment.ndim > 1 else len(accompaniment)
            
            # 插值到样本级别
            gain_interpolated = np.interp(
                np.arange(acc_len),
                np.linspace(0, acc_len, len(gain_curve)),
                gain_curve
            )
            
            # 应用增益
            if accompaniment.ndim > 1:
                processed = accompaniment * gain_interpolated[np.newaxis, :]
            else:
                processed = accompaniment * gain_interpolated
            
            print("  ✅ 侧链压缩完成")
            return processed
            
        except Exception as e:
            print(f"  ⚠️ 侧链压缩失败: {str(e)}")
            return accompaniment
    
    def apply_multiband_compression(self, audio: np.ndarray) -> np.ndarray:
        """多段压缩"""
        if not HAS_PEDALBOARD:
            return audio
        
        print("【伴奏多段压缩】")
        
        try:
            board = Pedalboard([
                MultibandCompressor(
                    low_band_threshold_db=-15.0,
                    low_band_ratio=2.0,
                    low_band_attack_ms=20.0,
                    low_band_release_ms=200.0,
                    
                    mid_low_band_threshold_db=-12.0,
                    mid_low_band_ratio=2.5,
                    mid_low_band_attack_ms=15.0,
                    mid_low_band_release_ms=150.0,
                    
                    mid_high_band_threshold_db=-10.0,
                    mid_high_band_ratio=2.0,
                    mid_high_band_attack_ms=10.0,
                    mid_high_band_release_ms=100.0,
                    
                    high_band_threshold_db=-8.0,
                    high_band_ratio=1.5,
                    high_band_attack_ms=5.0,
                    high_band_release_ms=80.0,
                )
            ])
            processed = board(audio, self.sr)
            print("  ✅ 多段压缩完成")
            return processed
            
        except Exception as e:
            print(f"  ⚠️ 多段压缩失败: {str(e)}")
            return audio
    
    def process_accompaniment(self, accompaniment: np.ndarray,
                              vocal: np.ndarray,
                              enable_sidechain: bool = True,
                              enable_mid_side_eq: bool = True) -> np.ndarray:
        """
        完整伴奏处理
        
        参数:
            accompaniment: 伴奏音频（立体声）
            vocal: 人声音频（单声道）
            enable_sidechain: 是否启用侧链压缩
            enable_mid_side_eq: 是否启用 Mid/Side EQ
            
        返回:
            处理后的伴奏
        """
        print("\n" + "="*50)
        print("【伴奏处理链】开始")
        print("="*50)
        
        result = accompaniment.copy()
        
        # 确保是立体声
        if result.ndim == 1:
            result = np.vstack([result, result])
        
        # 1. Mid/Side EQ
        if enable_mid_side_eq:
            result = self.apply_mid_side_eq(result)
        
        # 2. 侧链压缩
        if enable_sidechain and vocal is not None:
            result = self.apply_sidechain(result, vocal)
        
        # 3. 多段压缩
        result = self.apply_multiband_compression(result)
        
        print("【伴奏处理链】完成\n")
        return result


# ============================================================================
#                           总线处理模块
# ============================================================================

class BusProcessor:
    """
    总线处理模块
    
    功能：
    1. 胶水压缩
    2. 响度标准化
    3. 限制器
    """
    
    def __init__(self, sr: int = ProMixConfig.SAMPLE_RATE):
        self.sr = sr
    
    def apply_glue_compression(self, audio: np.ndarray) -> np.ndarray:
        """胶水压缩"""
        if not HAS_PEDALBOARD:
            return audio
        
        print("【总线胶水压缩】")
        
        try:
            board = Pedalboard([
                Compressor(
                    threshold_db=-18,
                    ratio=1.5,
                    attack_ms=20,
                    release_ms=150
                )
            ])
            processed = board(audio, self.sr)
            print("  ✅ 胶水压缩完成")
            return processed
        except Exception as e:
            print(f"  ⚠️ 胶水压缩失败: {str(e)}")
            return audio
    
    def normalize_loudness(self, audio: np.ndarray,
                           target_lufs: float = ProMixConfig.TARGET_LOUDNESS_LUFS) -> np.ndarray:
        """响度标准化"""
        print(f"【响度标准化】目标: {target_lufs} LUFS")
        
        try:
            # 确保是立体声
            if audio.ndim == 1:
                audio = np.vstack([audio, audio])
            
            meter = pyln.Meter(self.sr)
            current_lufs = meter.integrated_loudness(audio.T)
            
            if not np.isinf(current_lufs) and not np.isnan(current_lufs):
                gain_db = target_lufs - current_lufs
                gain_linear = 10 ** (gain_db / 20)
                audio = audio * gain_linear
                print(f"  响度调整: {gain_db:.1f} dB")
            
            return audio
        except Exception as e:
            print(f"  ⚠️ 响度标准化失败: {str(e)}")
            return audio
    
    def apply_limiter(self, audio: np.ndarray,
                      threshold_db: float = ProMixConfig.LIMITER_THRESHOLD) -> np.ndarray:
        """限制器"""
        if not HAS_PEDALBOARD:
            return audio
        
        print(f"【限制器】阈值: {threshold_db} dB")
        
        try:
            board = Pedalboard([
                Limiter(threshold_db=threshold_db)
            ])
            processed = board(audio, self.sr)
            print("  ✅ 限制器完成")
            return processed
        except Exception as e:
            print(f"  ⚠️ 限制器失败: {str(e)}")
            return audio
    
    def process_bus(self, audio: np.ndarray) -> np.ndarray:
        """
        完整总线处理
        
        参数:
            audio: 混合后的音频
            
        返回:
            处理后的音频
        """
        print("\n" + "="*50)
        print("【总线处理】开始")
        print("="*50)
        
        # 确保是立体声
        if audio.ndim == 1:
            audio = np.vstack([audio, audio])
        
        result = audio.copy()
        
        # 1. 胶水压缩
        result = self.apply_glue_compression(result)
        
        # 2. 响度标准化
        result = self.normalize_loudness(result)
        
        # 3. 限制器
        result = self.apply_limiter(result)
        
        print("【总线处理】完成\n")
        return result


# ============================================================================
#                           专业混音引擎
# ============================================================================

class ProMixEngine:
    """
    专业混音引擎 V3.0
    
    完整流程：
    1. 检测调性
    2. 处理人声
    3. 处理伴奏
    4. 混合
    5. 总线处理
    """
    
    def __init__(self, sr: int = ProMixConfig.SAMPLE_RATE):
        self.sr = sr
        self.vocal_chain = VocalChain(sr)
        self.acc_chain = AccompanimentChain(sr)
        self.bus_processor = BusProcessor(sr)
    
    def detect_song_key(self, vocal: np.ndarray, 
                        accompaniment: np.ndarray) -> str:
        """检测歌曲调性"""
        # 优先使用伴奏检测调性
        key = detect_key(ensure_mono(accompaniment), self.sr)
        print(f"【调性检测】歌曲调性: {key}大调")
        return key
    
    def balance_volumes(self, vocal: np.ndarray, 
                        accompaniment: np.ndarray,
                        vocal_offset_db: float = ProMixConfig.VOCAL_TO_ACC_OFFSET) -> Tuple[np.ndarray, np.ndarray]:
        """
        音量平衡
        
        参数:
            vocal: 人声
            accompaniment: 伴奏
            vocal_offset_db: 人声相对于伴奏的响度偏移
            
        返回:
            平衡后的人声和伴奏
        """
        print("【音量平衡】")
        
        try:
            meter = pyln.Meter(self.sr)
            
            # 测量伴奏响度
            acc_stereo = ensure_stereo(accompaniment)
            acc_lufs = meter.integrated_loudness(acc_stereo.T)
            if np.isinf(acc_lufs) or np.isnan(acc_lufs):
                acc_lufs = -18.0
            
            # 目标人声响度
            target_vocal_lufs = acc_lufs + vocal_offset_db
            
            # 测量人声响度并调整
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
    
    def mix_tracks(self, vocal: np.ndarray, 
                   accompaniment: np.ndarray) -> np.ndarray:
        """
        混合轨道
        
        参数:
            vocal: 处理后的人声（单声道）
            accompaniment: 处理后的伴奏（立体声）
            
        返回:
            混合后的音频（立体声）
        """
        print("【轨道混合】")
        
        # 确保人声是单声道
        vocal_mono = ensure_mono(vocal)
        
        # 确保伴奏是立体声
        acc_stereo = ensure_stereo(accompaniment)
        
        # 创建立体声人声
        vocal_stereo = np.vstack([vocal_mono, vocal_mono])
        
        # 长度对齐
        min_len = min(acc_stereo.shape[1], vocal_stereo.shape[1])
        
        print(f"  人声长度: {vocal_stereo.shape[1]} 样本 ({vocal_stereo.shape[1]/self.sr:.1f}秒)")
        print(f"  伴奏长度: {acc_stereo.shape[1]} 样本 ({acc_stereo.shape[1]/self.sr:.1f}秒)")
        print(f"  混合长度: {min_len} 样本 ({min_len/self.sr:.1f}秒)")
        
        # 混合
        mix = acc_stereo[:, :min_len] + vocal_stereo[:, :min_len]
        
        print("  ✅ 轨道混合完成")
        return mix
    
    def full_mix(self, vocal: np.ndarray,
                 accompaniment: np.ndarray,
                 enable_pitch_correction: bool = True,
                 pitch_correction_strength: float = 0.3,
                 enable_sidechain: bool = True) -> np.ndarray:
        """
        完整混音流程
        
        参数:
            vocal: 原始人声
            accompaniment: 原始伴奏
            enable_pitch_correction: 是否启用修音
            pitch_correction_strength: 修音强度 (0-1)
            enable_sidechain: 是否启用侧链压缩
            
        返回:
            混音成品（立体声）
        """
        print("\n" + "="*60)
        print("🎙️  专业混音引擎 V3.0 启动")
        print("="*60)
        print("流程: 修音 → 去齿音 → 压缩 → EQ → 共振控制 → 混响 → 限制")
        
        # 检测调性
        key = self.detect_song_key(vocal, accompaniment)
        
        # 音量平衡
        vocal, accompaniment = self.balance_volumes(vocal, accompaniment)
        
        # 处理人声
        processed_vocal = self.vocal_chain.process_vocal(
            vocal,
            key=key,
            enable_pitch_correction=enable_pitch_correction,
            pitch_correction_strength=pitch_correction_strength
        )
        
        # 处理伴奏
        processed_acc = self.acc_chain.process_accompaniment(
            accompaniment,
            processed_vocal,
            enable_sidechain=enable_sidechain
        )
        
        # 混合
        mix = self.mix_tracks(processed_vocal, processed_acc)
        
        # 总线处理
        final_mix = self.bus_processor.process_bus(mix)
        
        print("\n" + "="*60)
        print("✅ 专业混音完成")
        print("="*60 + "\n")
        
        return final_mix


# ============================================================================
#                           对外接口
# ============================================================================

def pro_mix(vocal_path: str,
            accompaniment_path: str,
            output_path: str,
            enable_pitch_correction: bool = True,
            pitch_correction_strength: float = 0.3) -> dict:
    """
    专业混音接口
    
    参数:
        vocal_path: 人声文件路径
        accompaniment_path: 伴奏文件路径
        output_path: 输出文件路径
        enable_pitch_correction: 是否启用修音
        pitch_correction_strength: 修音强度 (0-1)
        
    返回:
        {
            "code": 0/-1,
            "output_file": "输出文件路径",
            "msg": "消息",
            "stats": {...}
        }
    """
    try:
        print(f"\n{'='*60}")
        print(f"🚀 【专业混音】启动...")
        print(f"{'='*60}")
        
        # 加载音频
        print("\n【步骤 1/4】资源加载...")
        
        vocal, _ = librosa.load(vocal_path, sr=ProMixConfig.SAMPLE_RATE)
        print(f"  人声加载完成: {len(vocal)/ProMixConfig.SAMPLE_RATE:.1f}秒")
        
        acc, _ = librosa.load(accompaniment_path, sr=ProMixConfig.SAMPLE_RATE, mono=False)
        if acc.ndim == 1:
            acc = np.vstack([acc, acc])
        print(f"  伴奏加载完成: {acc.shape[1]/ProMixConfig.SAMPLE_RATE:.1f}秒")
        
        # 执行混音
        print("\n【步骤 2/4】执行混音引擎...")
        
        engine = ProMixEngine(sr=ProMixConfig.SAMPLE_RATE)
        mix = engine.full_mix(
            vocal=vocal,
            accompaniment=acc,
            enable_pitch_correction=enable_pitch_correction,
            pitch_correction_strength=pitch_correction_strength
        )
        
        # 导出
        print("\n【步骤 3/4】导出混音...")
        sf.write(output_path, mix.T, ProMixConfig.SAMPLE_RATE, subtype="PCM_24")
        
        # 统计
        print("\n【步骤 4/4】生成统计...")
        
        meter = pyln.Meter(ProMixConfig.SAMPLE_RATE)
        final_lufs = meter.integrated_loudness(mix.T)
        final_peak = 20 * np.log10(np.max(np.abs(mix)) + 1e-10)
        
        print(f"\n{'='*60}")
        print(f"✅ 【专业混音完成】")
        print(f"{'='*60}")
        print(f"  📁 输出文件: {output_path}")
        print(f"  📊 最终响度: {final_lufs:.1f} LUFS")
        print(f"  📊 峰值电平: {final_peak:.1f} dB")
        print(f"  📊 采样率: {ProMixConfig.SAMPLE_RATE} Hz")
        print(f"{'='*60}\n")
        
        return {
            "code": 0,
            "output_file": output_path,
            "msg": "混音完成",
            "stats": {
                "lufs": round(final_lufs, 1),
                "peak_db": round(final_peak, 1),
                "sample_rate": ProMixConfig.SAMPLE_RATE
            }
        }
        
    except Exception as e:
        print(f"\n❌ 【专业混音失败】: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "code": -1,
            "msg": str(e)
        }


# ============================================================================
#                           导出
# ============================================================================

__all__ = [
    'ProMixEngine',
    'ProMixConfig',
    'VocalChain',
    'AccompanimentChain',
    'BusProcessor',
    'PitchCorrector',
    'ResonanceController',
    'pro_mix'
]