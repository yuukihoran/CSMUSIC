"""
专业音频效果器封装模块 V2.0
===========================
封装所有音频处理效果器，确保可用性和稳定性
包含多方案备份，当某个组件不可用时有替代方案

功能包括：
- 多段压缩器 (Multiband Compressor)
- 去齿音器 (De-Esser)  
- 动态压缩器 (Compressor)
- 参数均衡器 (Parametric EQ)
- 混响器 (Reverb)
- 限制器 (Limiter)
- 噪声门 (Noise Gate)
- 音高校正 (Pitch Correction)
- 动态共振控制 (Resonance Control)
- 侧链压缩 (Sidechain Compression)
- 响度标准化 (Loudness Normalization)

依赖：
- pedalboard: 主要音频效果
- audiocomplib: 专业压缩器和限制器
- scipy: 信号处理
- pyworld: 音高处理
- noisereduce: 降噪
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

# ============================================================================
#                           依赖检测与导入
# ============================================================================

# Pedalboard - 主要音频效果库
HAS_PEDALBOARD = False
PEDALBOARD_COMPONENTS = []
try:
    from pedalboard import (
        Pedalboard, Compressor, Reverb, HighpassFilter, LowpassFilter,
        PeakFilter, LowShelfFilter, HighShelfFilter, Limiter, NoiseGate,
        Delay, Gain, Clipping, Bitcrush, Chorus, Phaser, Distortion,
        Convolution, PitchShift, IIRFilter, LadderFilter
    )
    HAS_PEDALBOARD = True
    PEDALBOARD_COMPONENTS = ['Compressor', 'Reverb', 'HighpassFilter', 'LowpassFilter',
                              'PeakFilter', 'LowShelfFilter', 'HighShelfFilter', 
                              'Limiter', 'NoiseGate', 'Delay', 'Gain']
    print("【音频效果器】Pedalboard 加载成功")
except ImportError as e:
    print(f"【警告】Pedalboard 未安装: {e}")

# audiocomplib - 专业压缩器和限制器
HAS_AUDIOCOMPLIB = False
try:
    from audiocomplib import AudioCompressor, PeakLimiter
    HAS_AUDIOCOMPLIB = True
    print("【音频效果器】AudioComplib 加载成功")
except ImportError:
    print("【提示】AudioComplib 未安装，将使用 Pedalboard 替代")

# noisereduce - 降噪库
HAS_NOISEREDUCE = False
try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
    print("【音频效果器】NoiseReduce 加载成功")
except ImportError:
    print("【提示】NoiseReduce 未安装")

# pyworld - 音高处理
HAS_PYWORLD = False
try:
    import pyworld as pw
    HAS_PYWORLD = True
    print("【音频效果器】PyWorld 加载成功")
except ImportError:
    print("【提示】PyWorld 未安装，修音功能将受限")

# scipy - 信号处理
HAS_SCIPY = False
try:
    from scipy import signal
    from scipy.ndimage import uniform_filter1d
    from scipy.fft import fft, ifft, fftfreq
    HAS_SCIPY = True
    print("【音频效果器】SciPy 加载成功")
except ImportError:
    print("【提示】SciPy 未安装")

# pyloudnorm - 响度标准化
HAS_PYLOUDNORM = False
try:
    import pyloudnorm as pyln
    HAS_PYLOUDNORM = True
    print("【音频效果器】PyLoudnorm 加载成功")
except ImportError:
    print("【提示】PyLoudnorm 未安装")

# librosa - 音频分析
HAS_LIBROSA = False
try:
    import librosa
    HAS_LIBROSA = True
    print("【音频效果器】Librosa 加载成功")
except ImportError:
    print("【提示】Librosa 未安装")


# ============================================================================
#                           效果器配置参数
# ============================================================================

@dataclass
class VocalChainConfig:
    """人声处理链配置"""
    
    # 采样率
    sample_rate: int = 48000
    
    # 低切参数
    low_cut_hz: float = 80.0
    
    # 去齿音参数
    deess_threshold_db: float = -25.0
    deess_frequency: float = 6000.0
    deess_ratio: float = 4.0
    
    # 第一次压缩参数
    comp1_threshold_db: float = -18.0
    comp1_ratio: float = 3.0
    comp1_attack_ms: float = 10.0
    comp1_release_ms: float = 100.0
    
    # 第二次压缩参数
    comp2_threshold_db: float = -24.0
    comp2_ratio: float = 2.0
    comp2_attack_ms: float = 5.0
    comp2_release_ms: float = 150.0
    
    # EQ 参数
    eq_low_shelf_hz: float = 250.0
    eq_low_shelf_gain_db: float = -1.5
    eq_presence_hz: float = 3000.0
    eq_presence_gain_db: float = 2.0
    eq_air_hz: float = 10000.0
    eq_air_gain_db: float = 2.0
    
    # 混响参数
    reverb_room_size: float = 0.3
    reverb_damping: float = 0.5
    reverb_wet_level: float = 0.15
    
    # 限制器参数
    limiter_threshold_db: float = -0.5
    
    # 响度目标
    target_lufs: float = -14.0


@dataclass
class AccompanimentConfig:
    """伴奏处理配置"""
    
    # 侧链压缩参数
    sidechain_reduction_db: float = 3.0
    
    # Mid/Side EQ 参数
    vocal_duck_freq: float = 2000.0
    vocal_duck_db: float = -2.0


# ============================================================================
#                           工具函数
# ============================================================================

def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """确保音频是单声道"""
    if audio.ndim > 1:
        return np.mean(audio, axis=0)
    return audio


def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    """确保音频是立体声 (2, samples)"""
    if audio.ndim == 1:
        return np.vstack([audio, audio])
    elif audio.shape[0] == 1:
        return np.vstack([audio[0], audio[0]])
    elif audio.shape[0] > 2:
        return audio[:2]
    return audio


def pedalboard_process_mono(board, audio: np.ndarray, sr: int) -> np.ndarray:
    """使用 Pedalboard 处理单声道音频"""
    if not HAS_PEDALBOARD:
        return audio
    
    was_mono = audio.ndim == 1
    
    if was_mono:
        result = board(audio, sr)
    else:
        result = board(audio, sr)
    
    # 如果输出是立体声但输入是单声道，转回单声道
    if was_mono and result.ndim > 1:
        result = np.mean(result, axis=0)
    
    return result


# ============================================================================
#                           多段压缩器实现
# ============================================================================

class MultibandCompressor:
    """
    多段压缩器实现
    当 Pedalboard 的 MultibandCompressor 不可用时使用
    基于 4 频段压缩
    """
    
    def __init__(self, sr: int = 48000):
        self.sr = sr
        self.crossover_freqs = [200, 2000, 8000]  # 三分频点
        
        # 各频段压缩参数
        self.band_params = {
            'low': {'threshold': -15.0, 'ratio': 2.0, 'attack': 20.0, 'release': 200.0},
            'mid_low': {'threshold': -12.0, 'ratio': 2.5, 'attack': 15.0, 'release': 150.0},
            'mid_high': {'threshold': -10.0, 'ratio': 3.0, 'attack': 10.0, 'release': 100.0},
            'high': {'threshold': -8.0, 'ratio': 2.0, 'attack': 5.0, 'release': 80.0}
        }
    
    def _design_crossover(self, freq: float, order: int = 4):
        """设计分频滤波器"""
        if not HAS_SCIPY:
            return None, None
        
        nyquist = self.sr / 2
        normalized_freq = freq / nyquist
        
        # 低通滤波器
        b_low, a_low = signal.butter(order, normalized_freq, btype='low')
        # 高通滤波器
        b_high, a_high = signal.butter(order, normalized_freq, btype='high')
        
        return (b_low, a_low), (b_high, a_high)
    
    def _compress_band(self, audio: np.ndarray, params: dict) -> np.ndarray:
        """压缩单个频段"""
        if HAS_PEDALBOARD:
            board = Pedalboard([
                Compressor(
                    threshold_db=params['threshold'],
                    ratio=params['ratio'],
                    attack_ms=params['attack'],
                    release_ms=params['release']
                )
            ])
            return pedalboard_process_mono(board, audio, self.sr)
        elif HAS_AUDIOCOMPLIB:
            comp = AudioCompressor(
                threshold=params['threshold'],
                ratio=params['ratio'],
                attack_time_ms=params['attack'],
                release_time_ms=params['release']
            )
            if audio.ndim == 1:
                audio = audio.reshape(1, -1)
            return comp.process(audio.astype(np.float32), self.sr).flatten()
        
        # 简单软限制作为后备
        threshold_linear = 10 ** (params['threshold'] / 20)
        ratio = params['ratio']
        
        compressed = audio.copy()
        mask = np.abs(audio) > threshold_linear
        compressed[mask] = np.sign(audio[mask]) * (threshold_linear + 
                        (np.abs(audio[mask]) - threshold_linear) / ratio)
        
        return compressed
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """处理音频"""
        if not HAS_SCIPY:
            # 无法分频，使用全频段压缩
            return self._compress_band(audio, self.band_params['mid_low'])
        
        audio_mono = ensure_mono(audio)
        
        # 分频处理
        bands = []
        
        # 低频段 (0 - 200 Hz)
        low_filter, _ = self._design_crossover(self.crossover_freqs[0])
        if low_filter:
            low_band = signal.filtfilt(low_filter[0], low_filter[1], audio_mono)
        else:
            low_band = np.zeros_like(audio_mono)
        bands.append(('low', low_band))
        
        # 中低频段 (200 - 2000 Hz)
        mid_low_low, mid_low_high = self._design_crossover(self.crossover_freqs[1])
        if mid_low_low and mid_low_high:
            mid_low_band = signal.filtfilt(mid_low_high[0], mid_low_high[1], audio_mono)
            mid_low_band = signal.filtfilt(mid_low_low[0], mid_low_low[1], mid_low_band, padlen=0)
        else:
            mid_low_band = audio_mono
        bands.append(('mid_low', mid_low_band))
        
        # 中高频段 (2000 - 8000 Hz)
        mid_high_low, mid_high_high = self._design_crossover(self.crossover_freqs[2])
        if mid_high_low and mid_high_high:
            mid_high_band = signal.filtfilt(mid_high_high[0], mid_high_high[1], audio_mono)
            mid_high_band = signal.filtfilt(mid_high_low[0], mid_high_low[1], mid_high_band, padlen=0)
        else:
            mid_high_band = audio_mono
        bands.append(('mid_high', mid_high_band))
        
        # 高频段 (8000+ Hz)
        if mid_high_high:
            high_band = signal.filtfilt(mid_high_high[0], mid_high_high[1], audio_mono)
        else:
            high_band = np.zeros_like(audio_mono)
        bands.append(('high', high_band))
        
        # 分别压缩各频段
        compressed_bands = []
        for name, band in bands:
            if np.max(np.abs(band)) > 1e-10:  # 避免处理静音频段
                compressed = self._compress_band(band, self.band_params[name])
            else:
                compressed = band
            compressed_bands.append(compressed)
        
        # 重组
        result = np.sum(compressed_bands, axis=0)
        
        # 归一化
        max_val = np.max(np.abs(result))
        if max_val > 1.0:
            result = result / max_val * 0.95
        
        return result.astype(np.float32)


# ============================================================================
#                           去齿音器实现
# ============================================================================

class DeEsser:
    """
    去齿音器实现
    多方案实现，确保可用性
    """
    
    def __init__(self, sr: int = 48000, 
                 threshold_db: float = -25.0,
                 frequency: float = 6000.0,
                 ratio: float = 4.0):
        self.sr = sr
        self.threshold_db = threshold_db
        self.frequency = frequency
        self.ratio = ratio
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """处理音频"""
        audio_mono = ensure_mono(audio)
        
        # 方案1: 使用多段压缩器模拟
        mb_comp = MultibandCompressor(self.sr)
        mb_comp.band_params['mid_high'] = {
            'threshold': self.threshold_db,
            'ratio': self.ratio,
            'attack': 1.0,
            'release': 50.0
        }
        mb_comp.band_params['high'] = {
            'threshold': self.threshold_db + 3,
            'ratio': self.ratio + 1,
            'attack': 2.0,
            'release': 80.0
        }
        
        result = mb_comp.process(audio_mono)
        
        return result.astype(np.float32)


# ============================================================================
#                           动态共振控制器
# ============================================================================

class ResonanceController:
    """
    动态共振控制 - 类似 soothe2 效果
    自动检测并衰减刺耳的共振频率
    """
    
    def __init__(self, sr: int = 48000):
        self.sr = sr
    
    def process(self, audio: np.ndarray, 
                threshold_db: float = -30.0,
                smoothing: float = 0.7) -> np.ndarray:
        """处理音频"""
        if not HAS_SCIPY or not HAS_LIBROSA:
            return audio
        
        audio_mono = ensure_mono(audio)
        
        try:
            n_fft = 2048
            hop_length = n_fft // 4
            
            # STFT
            D = librosa.stft(audio_mono, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(D)
            
            # 计算动态阈值
            freq_mean = np.mean(magnitude, axis=1, keepdims=True)
            freq_std = np.std(magnitude, axis=1, keepdims=True)
            
            # 创建增益曲线
            gain = np.ones_like(magnitude)
            
            for i in range(magnitude.shape[0]):
                dynamic_threshold = freq_mean[i] + 2 * freq_std[i]
                
                mask = magnitude[i:i+1] > dynamic_threshold
                excess = magnitude[i:i+1] - dynamic_threshold
                gain[i:i+1] = np.where(mask, 
                                       1 - (excess / (magnitude[i:i+1] + 1e-10)) * smoothing,
                                       1)
            
            gain = np.clip(gain, 0.1, 1.0)
            
            # 应用增益
            D_processed = D * gain
            processed = librosa.istft(D_processed, hop_length=hop_length, length=len(audio_mono))
            
            return processed.astype(np.float32)
            
        except Exception as e:
            print(f"  ⚠️ 动态共振控制失败: {str(e)}")
            return audio


# ============================================================================
#                           音高校正器
# ============================================================================

class PitchCorrector:
    """
    轻微修音模块
    基于 pyworld 实现
    """
    
    def __init__(self, sr: int = 48000):
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
            correction_strength: 校正强度 (0.0-1.0)
        """
        if not HAS_PYWORLD:
            return audio
        
        try:
            audio_f64 = audio.astype(np.float64)
            
            # 提取音高
            _f0, t = pw.dio(audio_f64, self.sr)
            f0 = pw.stonemask(audio_f64, _f0, t, self.sr)
            
            # 提取频谱包络和非周期性
            sp = pw.cheaptrick(audio_f64, f0, t, self.sr)
            ap = pw.d4c(audio_f64, f0, t, self.sr)
            
            # 校正音高
            corrected_f0 = f0.copy()
            
            for i, freq in enumerate(f0):
                if freq > 0:
                    target_freq = self.get_nearest_note_frequency(freq, key)
                    corrected_f0[i] = freq + (target_freq - freq) * correction_strength
            
            # 重新合成
            corrected = pw.synthesize(corrected_f0, sp, ap, self.sr)
            
            # 长度对齐
            if len(corrected) > len(audio):
                corrected = corrected[:len(audio)]
            elif len(corrected) < len(audio):
                corrected = np.pad(corrected, (0, len(audio) - len(corrected)))
            
            return corrected.astype(np.float32)
            
        except Exception as e:
            print(f"  ⚠️ 修音失败: {str(e)}")
            return audio


# ============================================================================
#                           降噪处理器
# ============================================================================

class NoiseReducer:
    """深度降噪处理器"""
    
    def __init__(self, sr: int = 48000):
        self.sr = sr
    
    def process(self, audio: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """降噪处理"""
        audio_mono = ensure_mono(audio)
        
        # 方案1: noisereduce
        if HAS_NOISEREDUCE:
            try:
                cleaned = nr.reduce_noise(
                    y=audio_mono, sr=self.sr,
                    prop_decrease=strength,
                    stationary=False
                )
                if len(cleaned) > len(audio_mono):
                    cleaned = cleaned[:len(audio_mono)]
                elif len(cleaned) < len(audio_mono):
                    cleaned = np.pad(cleaned, (0, len(audio_mono) - len(cleaned)))
                return cleaned.astype(np.float32)
            except Exception as e:
                print(f"  ⚠️ noisereduce 失败: {str(e)}")
        
        # 方案2: scipy 频谱门限
        if HAS_SCIPY:
            try:
                f, t, Zxx = signal.stft(audio_mono, fs=self.sr, nperseg=1024, noverlap=768)
                magnitude = np.abs(Zxx)
                noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
                threshold = noise_floor * (1 + strength * 3)
                
                mask = magnitude > threshold
                gain = np.where(mask, 1.0, 0.1 + 0.9 * (magnitude / (threshold + 1e-10)) ** 2)
                
                Zxx_clean = Zxx * gain
                _, cleaned = signal.istft(Zxx_clean, fs=self.sr, nperseg=1024, noverlap=768)
                
                if len(cleaned) > len(audio_mono):
                    cleaned = cleaned[:len(audio_mono)]
                elif len(cleaned) < len(audio_mono):
                    cleaned = np.pad(cleaned, (0, len(audio_mono) - len(cleaned)))
                
                return cleaned.astype(np.float32)
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
                return pedalboard_process_mono(board, audio_mono, self.sr)
            except Exception as e:
                print(f"  ⚠️ NoiseGate 失败: {str(e)}")
        
        return audio_mono


# ============================================================================
#                           侧链压缩器
# ============================================================================

class SidechainCompressor:
    """侧链压缩器"""
    
    def __init__(self, sr: int = 48000):
        self.sr = sr
    
    def process(self, accompaniment: np.ndarray, 
                vocal: np.ndarray,
                reduction_db: float = 3.0) -> np.ndarray:
        """侧链压缩"""
        vocal_mono = ensure_mono(vocal)
        
        if HAS_LIBROSA and HAS_SCIPY:
            try:
                frame_size = int(self.sr * 0.01)
                hop_size = frame_size // 2
                
                vocal_envelope = librosa.feature.rms(
                    y=vocal_mono,
                    frame_length=frame_size,
                    hop_length=hop_size
                )[0]
                
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
                
                return processed
                
            except Exception as e:
                print(f"  ⚠️ 侧链压缩失败: {str(e)}")
        
        return accompaniment


# ============================================================================
#                           响度标准化器
# ============================================================================

class LoudnessNormalizer:
    """响度标准化器"""
    
    def __init__(self, sr: int = 48000):
        self.sr = sr
    
    def measure_loudness(self, audio: np.ndarray) -> float:
        """测量响度 (LUFS)"""
        if not HAS_PYLOUDNORM:
            # 简单 RMS 计算
            rms = np.sqrt(np.mean(audio ** 2))
            return 20 * np.log10(rms + 1e-10)
        
        audio_stereo = ensure_stereo(audio)
        meter = pyln.Meter(self.sr)
        return meter.integrated_loudness(audio_stereo.T)
    
    def normalize(self, audio: np.ndarray, target_lufs: float = -14.0) -> np.ndarray:
        """响度标准化"""
        current_lufs = self.measure_loudness(audio)
        
        if np.isinf(current_lufs) or np.isnan(current_lufs):
            return audio
        
        gain_db = target_lufs - current_lufs
        gain_linear = 10 ** (gain_db / 20)
        
        return audio * gain_linear


# ============================================================================
#                           完整人声处理链
# ============================================================================

class ProfessionalVocalChain:
    """
    专业人声处理链
    
    流程：
    1. 降噪 (可选)
    2. 修音 (可选)
    3. 低切
    4. 去齿音
    5. 第一次压缩
    6. 第一次 EQ
    7. 第二次压缩
    8. 第二次 EQ
    9. 动态共振控制
    10. 混响
    """
    
    def __init__(self, config: VocalChainConfig = None):
        self.config = config or VocalChainConfig()
        self.sr = self.config.sample_rate
        
        # 初始化各处理器
        self.deesser = DeEsser(
            self.sr,
            self.config.deess_threshold_db,
            self.config.deess_frequency,
            self.config.deess_ratio
        )
        self.resonance_controller = ResonanceController(self.sr)
        self.pitch_corrector = PitchCorrector(self.sr)
        self.noise_reducer = NoiseReducer(self.sr)
        self.loudness_normalizer = LoudnessNormalizer(self.sr)
    
    def process(self, audio: np.ndarray,
                key: str = 'C',
                enable_pitch_correction: bool = True,
                pitch_correction_strength: float = 0.3,
                enable_denoise: bool = True,
                enable_deess: bool = True,
                enable_reverb: bool = True) -> np.ndarray:
        """
        完整人声处理
        """
        result = ensure_mono(audio)
        
        # 1. 降噪
        if enable_denoise:
            print("【降噪】处理中...")
            result = self.noise_reducer.process(result, strength=0.5)
        
        # 2. 修音
        if enable_pitch_correction:
            print(f"【修音】调性: {key}, 强度: {pitch_correction_strength}")
            result = self.pitch_corrector.correct_pitch(result, key, pitch_correction_strength)
        
        if not HAS_PEDALBOARD:
            print("  ⚠️ Pedalboard 不可用，跳过后续处理")
            return result
        
        # 3. 低切
        print(f"【低切】频率: {self.config.low_cut_hz} Hz")
        board = Pedalboard([HighpassFilter(cutoff_frequency_hz=self.config.low_cut_hz)])
        result = pedalboard_process_mono(board, result, self.sr)
        
        # 4. 去齿音
        if enable_deess:
            print("【去齿音】处理中...")
            result = self.deesser.process(result)
        
        # 5. 第一次压缩
        print(f"【第一次压缩】阈值: {self.config.comp1_threshold_db} dB")
        board = Pedalboard([
            Compressor(
                threshold_db=self.config.comp1_threshold_db,
                ratio=self.config.comp1_ratio,
                attack_ms=self.config.comp1_attack_ms,
                release_ms=self.config.comp1_release_ms
            )
        ])
        result = pedalboard_process_mono(board, result, self.sr)
        
        # 6. 第一次 EQ
        print("【第一次EQ】基础频率调整")
        board = Pedalboard([
            LowShelfFilter(
                cutoff_frequency_hz=self.config.eq_low_shelf_hz,
                gain_db=self.config.eq_low_shelf_gain_db
            ),
            PeakFilter(
                cutoff_frequency_hz=self.config.eq_presence_hz,
                gain_db=self.config.eq_presence_gain_db,
                q=1.0
            ),
            HighShelfFilter(
                cutoff_frequency_hz=self.config.eq_air_hz,
                gain_db=self.config.eq_air_gain_db
            ),
        ])
        result = pedalboard_process_mono(board, result, self.sr)
        
        # 7. 第二次压缩
        print(f"【第二次压缩】阈值: {self.config.comp2_threshold_db} dB")
        board = Pedalboard([
            Compressor(
                threshold_db=self.config.comp2_threshold_db,
                ratio=self.config.comp2_ratio,
                attack_ms=self.config.comp2_attack_ms,
                release_ms=self.config.comp2_release_ms
            )
        ])
        result = pedalboard_process_mono(board, result, self.sr)
        
        # 8. 动态共振控制
        print("【动态共振控制】处理中...")
        result = self.resonance_controller.process(result)
        
        # 9. 混响
        if enable_reverb:
            print(f"【混响】房间大小: {self.config.reverb_room_size}")
            board = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=200),
                LowpassFilter(cutoff_frequency_hz=8000),
                Reverb(
                    room_size=self.config.reverb_room_size,
                    damping=self.config.reverb_damping,
                    wet_level=self.config.reverb_wet_level,
                    dry_level=1 - self.config.reverb_wet_level
                )
            ])
            result = pedalboard_process_mono(board, result, self.sr)
        
        return result.astype(np.float32)


# ============================================================================
#                           导出
# ============================================================================

__all__ = [
    # 配置类
    'VocalChainConfig',
    'AccompanimentConfig',
    # 效果器
    'MultibandCompressor',
    'DeEsser',
    'ResonanceController',
    'PitchCorrector',
    'NoiseReducer',
    'SidechainCompressor',
    'LoudnessNormalizer',
    'ProfessionalVocalChain',
    # 工具函数
    'ensure_mono',
    'ensure_stereo',
    'pedalboard_process_mono',
    # 状态标志
    'HAS_PEDALBOARD',
    'HAS_AUDIOCOMPLIB',
    'HAS_NOISEREDUCE',
    'HAS_PYWORLD',
    'HAS_SCIPY',
    'HAS_PYLOUDNORM',
    'HAS_LIBROSA',
]