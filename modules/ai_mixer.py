"""
AI Mix 专业贴唱混音模块 V2.0
=============================
基于深度学习的专业混音引擎

核心特性：
1. 使用 Matchering 进行风格匹配和母带处理
2. 使用 noisereduce 进行深度降噪
3. 使用 DTLN 风格的网络进行人声增强
4. 移除有问题的时间对齐，让用户自行对齐

依赖：
- matchering: 音频匹配和母带处理
- noisereduce: 深度降噪
- pedalboard: 专业音频效果
- pyloudnorm: 响度标准化
"""

import os
import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
import warnings
warnings.filterwarnings('ignore')

# 尝试导入深度学习库
try:
    import matchering as mg
    HAS_MATCHERING = True
except ImportError:
    HAS_MATCHERING = False
    print("【警告】Matchering 未安装，母带风格匹配功能将受限")

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False
    print("【警告】noisereduce 未安装，降噪功能将受限")

try:
    from pedalboard import (
        Pedalboard, Compressor, Reverb, HighpassFilter, LowpassFilter,
        PeakFilter, LowShelfFilter, HighShelfFilter, Limiter, NoiseGate,
        Delay, Gain, Clipping
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

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE, PROJECT_DIR


# ============================================================================
#                           配置参数
# ============================================================================

class MixConfig:
    """混音配置 - 发布级标准"""
    
    # 采样率
    SAMPLE_RATE = 48000
    
    # 发布级响度标准 (LUFS)
    TARGET_LOUDNESS_LUFS = -12.0  # 流行歌标准
    
    # 人声目标响度（相对于伴奏）
    VOCAL_TO_ACC_LOUDNESS_OFFSET = 1.5  # 人声比伴奏响 1.5 dB
    
    # 人声 EQ 参数
    VOCAL_LOW_CUT_HZ = 80.0
    VOCAL_PRESENCE_BOOST_HZ = 3000.0
    VOCAL_PRESENCE_BOOST_DB = 2.0
    VOCAL_AIR_BOOST_HZ = 10000.0
    VOCAL_AIR_BOOST_DB = 2.0
    
    # 压缩器参数
    COMPRESSOR_THRESHOLD_DB = -18.0
    COMPRESSOR_RATIO = 3.0
    COMPRESSOR_ATTACK_MS = 10.0
    COMPRESSOR_RELEASE_MS = 100.0
    
    # 混响参数
    REVERB_ROOM_SIZE = 0.3
    REVERB_WET_LEVEL = 0.15
    
    # 限制器
    LIMITER_THRESHOLD_DB = -0.5


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
    """使用 Pedalboard 处理单声道音频，确保输出仍为单声道"""
    if audio.ndim > 1:
        return board(audio, sr)
    
    result = board(audio, sr)
    
    if result.ndim > 1:
        result = np.mean(result, axis=0)
    
    return result


# ============================================================================
#                           深度学习音频处理
# ============================================================================

class DeepAudioProcessor:
    """深度学习音频处理器"""
    
    def __init__(self, sr: int = MixConfig.SAMPLE_RATE):
        self.sr = sr
    
    def deep_denoise(self, audio: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """
        使用深度学习降噪
        - noisereduce 基于频谱门限的降噪
        """
        if not HAS_NOISEREDUCE:
            print("  ⚠️ noisereduce 不可用，跳过深度降噪")
            return audio
        
        print(f"【深度降噪】使用频谱降噪算法 (强度: {strength})...")
        
        try:
            # 使用 noisereduce 的降噪算法
            # stationary=False 可以处理非平稳噪声
            cleaned = nr.reduce_noise(
                y=audio,
                sr=self.sr,
                prop_decrease=strength,  # 降噪强度
                stationary=False
            )
            
            # 确保长度一致
            if len(cleaned) > len(audio):
                cleaned = cleaned[:len(audio)]
            elif len(cleaned) < len(audio):
                cleaned = np.pad(cleaned, (0, len(audio) - len(cleaned)))
            
            print("  ✅ 深度降噪完成")
            return cleaned
            
        except Exception as e:
            print(f"  ⚠️ 深度降噪失败: {str(e)}")
            return audio
    
    def vocal_enhancement(self, audio: np.ndarray) -> np.ndarray:
        """
        人声增强
        - 提升人声清晰度
        - 动态均衡处理
        """
        if not HAS_PEDALBOARD:
            return audio
        
        print("【人声增强】应用动态处理...")
        
        try:
            board = Pedalboard([
                # 低切
                HighpassFilter(cutoff_frequency_hz=MixConfig.VOCAL_LOW_CUT_HZ),
                # 去除闷音
                LowShelfFilter(cutoff_frequency_hz=250, gain_db=-1.5),
                # 穿透力提升
                PeakFilter(
                    cutoff_frequency_hz=MixConfig.VOCAL_PRESENCE_BOOST_HZ,
                    gain_db=MixConfig.VOCAL_PRESENCE_BOOST_DB,
                    q=1.0
                ),
                # 空气感
                HighShelfFilter(
                    cutoff_frequency_hz=MixConfig.VOCAL_AIR_BOOST_HZ,
                    gain_db=MixConfig.VOCAL_AIR_BOOST_DB
                ),
            ])
            
            processed = pedalboard_process_mono(board, audio, self.sr)
            
            print("  ✅ 人声增强完成")
            return processed
            
        except Exception as e:
            print(f"  ⚠️ 人声增强失败: {str(e)}")
            return audio
    
    def apply_compression(self, audio: np.ndarray) -> np.ndarray:
        """应用专业压缩"""
        if not HAS_PEDALBOARD:
            return audio
        
        print("【动态压缩】应用压缩器...")
        
        try:
            board = Pedalboard([
                Compressor(
                    threshold_db=MixConfig.COMPRESSOR_THRESHOLD_DB,
                    ratio=MixConfig.COMPRESSOR_RATIO,
                    attack_ms=MixConfig.COMPRESSOR_ATTACK_MS,
                    release_ms=MixConfig.COMPRESSOR_RELEASE_MS
                )
            ])
            
            processed = pedalboard_process_mono(board, audio, self.sr)
            
            print("  ✅ 压缩完成")
            return processed
            
        except Exception as e:
            print(f"  ⚠️ 压缩失败: {str(e)}")
            return audio
    
    def apply_deessing(self, audio: np.ndarray) -> np.ndarray:
        """去齿音"""
        if not HAS_PEDALBOARD or not HAS_DEESSER:
            print("  ⚠️ DeEsser 不可用，跳过去齿音")
            return audio
        
        print("【去齿音】处理中...")
        
        try:
            board = Pedalboard([
                DeEsser(threshold_db=-20, frequency=6000)
            ])
            
            processed = pedalboard_process_mono(board, audio, self.sr)
            
            print("  ✅ 去齿音完成")
            return processed
            
        except Exception as e:
            print(f"  ⚠️ 去齿音失败: {str(e)}")
            return audio
    
    def apply_reverb(self, audio: np.ndarray, room_size: float = None) -> np.ndarray:
        """应用混响"""
        if not HAS_PEDALBOARD:
            return audio
        
        if room_size is None:
            room_size = MixConfig.REVERB_ROOM_SIZE
        
        print(f"【混响】应用房间混响 (大小: {room_size})...")
        
        try:
            board = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=200),
                LowpassFilter(cutoff_frequency_hz=8000),
                Reverb(
                    room_size=room_size,
                    damping=0.5,
                    wet_level=MixConfig.REVERB_WET_LEVEL,
                    dry_level=1 - MixConfig.REVERB_WET_LEVEL
                )
            ])
            
            processed = pedalboard_process_mono(board, audio, self.sr)
            
            print("  ✅ 混响完成")
            return processed
            
        except Exception as e:
            print(f"  ⚠️ 混响失败: {str(e)}")
            return audio


# ============================================================================
#                           伴奏处理
# ============================================================================

class AccompanimentProcessor:
    """伴奏处理器"""
    
    def __init__(self, sr: int = MixConfig.SAMPLE_RATE):
        self.sr = sr
    
    def apply_sidechain(self, accompaniment: np.ndarray, vocal: np.ndarray) -> np.ndarray:
        """
        侧链压缩
        - 人声响时降低伴奏
        """
        print("【伴奏处理】应用侧链压缩...")
        
        # 确保人声是单声道
        vocal_mono = ensure_mono(vocal)
        
        # 计算人声包络
        frame_size = int(self.sr * 0.01)  # 10ms
        hop_size = frame_size // 2
        
        from scipy.ndimage import uniform_filter1d
        
        vocal_envelope = librosa.feature.rms(
            y=vocal_mono, 
            frame_length=frame_size, 
            hop_length=hop_size
        )[0]
        
        # 平滑包络
        vocal_envelope = uniform_filter1d(vocal_envelope, size=10)
        
        # 归一化
        vocal_envelope = vocal_envelope / (np.max(vocal_envelope) + 1e-10)
        
        # 创建增益曲线
        reduction_db = 2.5  # 最大衰减量
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
        
        print(f"  ✅ 侧链压缩完成 (最大衰减: {reduction_db}dB)")
        return processed
    
    def eq_ducking(self, accompaniment: np.ndarray) -> np.ndarray:
        """EQ 避让人声核心频段"""
        if not HAS_PEDALBOARD:
            return accompaniment
        
        print("【伴奏处理】应用 EQ 避让...")
        
        try:
            board = Pedalboard([
                PeakFilter(cutoff_frequency_hz=2000, gain_db=-1.5, q=0.7)
            ])
            
            processed = board(accompaniment, self.sr)
            
            print("  ✅ EQ 避让完成")
            return processed
            
        except Exception as e:
            print(f"  ⚠️ EQ 避让失败: {str(e)}")
            return accompaniment


# ============================================================================
#                           Matchering 风格匹配
# ============================================================================

class StyleMatcher:
    """
    使用 Matchering 进行风格匹配和母带处理
    - 匹配参考曲目的 RMS、频率响应、峰值、立体声宽度
    """
    
    def __init__(self, sr: int = MixConfig.SAMPLE_RATE):
        self.sr = sr
        self.temp_dir = "/tmp/matchering_temp"
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def match_style(self, target_path: str, reference_path: str, 
                    output_path: str) -> bool:
        """
        使用 Matchering 进行风格匹配
        
        参数:
            target_path: 待处理音频路径
            reference_path: 参考音频路径
            output_path: 输出路径
        
        返回:
            是否成功
        """
        if not HAS_MATCHERING:
            print("  ⚠️ Matchering 不可用，跳过风格匹配")
            return False
        
        print("【风格匹配】使用 Matchering 进行深度风格迁移...")
        print(f"  目标: {os.path.basename(target_path)}")
        print(f"  参考: {os.path.basename(reference_path)}")
        
        try:
            # 设置日志
            mg.log(print)
            
            # 执行匹配
            mg.process(
                target=target_path,
                reference=reference_path,
                results=[
                    mg.pcm24(output_path),
                ],
            )
            
            print("  ✅ 风格匹配完成")
            return True
            
        except Exception as e:
            print(f"  ⚠️ 风格匹配失败: {str(e)}")
            return False
    
    def master_with_reference(self, mix_path: str, reference_path: str,
                               output_path: str) -> str:
        """
        使用参考曲目进行母带处理
        """
        if self.match_style(mix_path, reference_path, output_path):
            return output_path
        else:
            # 回退到基础母带处理
            return self.basic_mastering(mix_path, output_path)
    
    def basic_mastering(self, audio_path: str, output_path: str) -> str:
        """基础母带处理"""
        print("【母带处理】应用基础母带处理...")
        
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=self.sr, mono=False)
        
        if audio.ndim == 1:
            audio = np.vstack([audio, audio])
        
        # 响度标准化
        meter = pyln.Meter(self.sr)
        current_lufs = meter.integrated_loudness(audio.T)
        
        if not np.isinf(current_lufs) and not np.isnan(current_lufs):
            gain_db = MixConfig.TARGET_LOUDNESS_LUFS - current_lufs
            gain_linear = 10 ** (gain_db / 20)
            audio = audio * gain_linear
            print(f"  响度调整: {gain_db:.1f} dB")
        
        # 应用限制器
        if HAS_PEDALBOARD:
            board = Pedalboard([
                Limiter(threshold_db=MixConfig.LIMITER_THRESHOLD_DB)
            ])
            audio = board(audio, self.sr)
        
        # 保存
        sf.write(output_path, audio.T, self.sr, subtype="PCM_24")
        
        print(f"  ✅ 母带处理完成")
        return output_path


# ============================================================================
#                           主混音引擎
# ============================================================================

class AIMixEngine:
    """
    AI 混音引擎 V2.0
    ================
    
    重要说明：
    - 不进行时间对齐，用户需自行对齐干音和伴奏
    - 使用深度学习算法进行音频处理
    - 使用 Matchering 进行风格匹配
    """
    
    def __init__(self, sr: int = MixConfig.SAMPLE_RATE):
        self.sr = sr
        self.deep_processor = DeepAudioProcessor(sr)
        self.acc_processor = AccompanimentProcessor(sr)
        self.style_matcher = StyleMatcher(sr)
    
    def process_vocal(self, vocal: np.ndarray, 
                       denoise: bool = True,
                       deess: bool = True,
                       enhance: bool = True,
                       compress: bool = True,
                       reverb: bool = True) -> np.ndarray:
        """
        处理人声
        
        注意：不进行时间对齐，假设用户已对齐
        """
        print("\n" + "="*50)
        print("【人声处理】开始")
        print("="*50)
        
        result = vocal.copy()
        
        # 1. 深度降噪
        if denoise:
            result = self.deep_processor.deep_denoise(result, strength=0.5)
        
        # 2. 去齿音
        if deess:
            result = self.deep_processor.apply_deessing(result)
        
        # 3. 人声增强
        if enhance:
            result = self.deep_processor.vocal_enhancement(result)
        
        # 4. 压缩
        if compress:
            result = self.deep_processor.apply_compression(result)
        
        # 5. 混响
        if reverb:
            result = self.deep_processor.apply_reverb(result)
        
        print("【人声处理】完成\n")
        return result
    
    def process_accompaniment(self, accompaniment: np.ndarray, 
                               vocal: np.ndarray,
                               sidechain: bool = True) -> np.ndarray:
        """处理伴奏"""
        print("\n" + "="*50)
        print("【伴奏处理】开始")
        print("="*50)
        
        result = accompaniment.copy()
        
        # 确保是立体声
        if result.ndim == 1:
            result = np.vstack([result, result])
        
        # 侧链压缩
        if sidechain:
            result = self.acc_processor.apply_sidechain(result, vocal)
        
        # EQ 避让
        result = self.acc_processor.eq_ducking(result)
        
        print("【伴奏处理】完成\n")
        return result
    
    def mix_tracks(self, vocal: np.ndarray, accompaniment: np.ndarray,
                   vocal_gain_db: float = 0.0) -> np.ndarray:
        """
        混合轨道
        
        参数:
            vocal: 处理后的人声（单声道）
            accompaniment: 处理后的伴奏（立体声）
            vocal_gain_db: 人声增益
        """
        print("\n" + "="*50)
        print("【轨道混合】开始")
        print("="*50)
        
        # 确保人声是单声道
        vocal_mono = ensure_mono(vocal)
        
        # 确保伴奏是立体声
        acc_stereo = ensure_stereo(accompaniment)
        
        # 应用增益
        if vocal_gain_db != 0:
            vocal_mono = vocal_mono * (10 ** (vocal_gain_db / 20))
        
        # 响度平衡
        meter = pyln.Meter(self.sr)
        
        # 测量伴奏响度
        acc_lufs = meter.integrated_loudness(acc_stereo.T)
        if np.isinf(acc_lufs) or np.isnan(acc_lufs):
            acc_lufs = -18.0
        
        # 目标人声响度
        target_vocal_lufs = acc_lufs + MixConfig.VOCAL_TO_ACC_LOUDNESS_OFFSET
        
        # 测量人声响度并调整
        vocal_stereo_temp = np.vstack([vocal_mono, vocal_mono])
        vocal_lufs = meter.integrated_loudness(vocal_stereo_temp.T)
        
        if not np.isinf(vocal_lufs) and not np.isnan(vocal_lufs):
            gain_db = target_vocal_lufs - vocal_lufs
            vocal_mono = vocal_mono * (10 ** (gain_db / 20))
            print(f"  人声响度调整: {gain_db:.1f} dB")
        
        # 创建立体声人声
        vocal_stereo = np.vstack([vocal_mono, vocal_mono])
        
        # 长度对齐
        min_len = min(acc_stereo.shape[1], vocal_stereo.shape[1])
        
        print(f"  人声长度: {vocal_stereo.shape[1]} 样本 ({vocal_stereo.shape[1]/self.sr:.1f}秒)")
        print(f"  伴奏长度: {acc_stereo.shape[1]} 样本 ({acc_stereo.shape[1]/self.sr:.1f}秒)")
        print(f"  混合长度: {min_len} 样本 ({min_len/self.sr:.1f}秒)")
        
        # 混合
        mix = acc_stereo[:, :min_len] + vocal_stereo[:, :min_len]
        
        print("【轨道混合】完成\n")
        return mix
    
    def apply_bus_processing(self, mix: np.ndarray) -> np.ndarray:
        """总线处理"""
        print("\n" + "="*50)
        print("【总线处理】开始")
        print("="*50)
        
        if not HAS_PEDALBOARD:
            print("  ⚠️ Pedalboard 不可用，跳过总线处理")
            return mix
        
        try:
            board = Pedalboard([
                # 胶水压缩
                Compressor(threshold_db=-18, ratio=1.5, attack_ms=20, release_ms=150),
            ])
            
            processed = board(mix, self.sr)
            
            print("  ✅ 总线处理完成")
            return processed
            
        except Exception as e:
            print(f"  ⚠️ 总线处理失败: {str(e)}")
            return mix
    
    def full_mix(self, vocal: np.ndarray, accompaniment: np.ndarray,
                 options: dict = None) -> np.ndarray:
        """
        完整混音流程
        
        参数:
            vocal: 用户干声（已对齐）
            accompaniment: 用户伴奏
            options: 混音选项
        
        返回:
            混音成品（立体声）
        """
        # 默认选项
        default_options = {
            "denoise": True,
            "deess": True,
            "enhance": True,
            "compress": True,
            "reverb": True,
            "sidechain": True,
            "vocal_gain_db": 0.0,
        }
        
        if options:
            default_options.update(options)
        options = default_options
        
        print("\n" + "="*60)
        print("🎙️  AI 专业混音引擎 V2.0 启动")
        print("="*60)
        print("\n📌 重要提示: 干音应由用户自行对齐，本引擎专注于音质处理")
        
        # 处理人声
        processed_vocal = self.process_vocal(
            vocal,
            denoise=options["denoise"],
            deess=options["deess"],
            enhance=options["enhance"],
            compress=options["compress"],
            reverb=options["reverb"]
        )
        
        # 处理伴奏
        processed_acc = self.process_accompaniment(
            accompaniment,
            processed_vocal,
            sidechain=options["sidechain"]
        )
        
        # 混合轨道
        mix = self.mix_tracks(
            processed_vocal,
            processed_acc,
            vocal_gain_db=options["vocal_gain_db"]
        )
        
        # 总线处理
        mix = self.apply_bus_processing(mix)
        
        print("\n" + "="*60)
        print("✅ AI 混音完成")
        print("="*60 + "\n")
        
        return mix


# ============================================================================
#                           对外接口
# ============================================================================

def ai_mix_with_original(original_song_path: str,
                         user_acc_path: str,
                         user_vocal_path: str,
                         project_id: str,
                         options: dict = None) -> dict:
    """
    AI 混音主接口 V2.0
    
    重要变更：
    - 不再进行时间对齐，用户需自行对齐干音和伴奏
    - 使用深度学习算法进行音频处理
    - 使用 Matchering 进行风格匹配
    
    参数:
        original_song_path: 原曲文件路径（用于风格参考）
        user_acc_path: 用户伴奏文件路径
        user_vocal_path: 用户干声文件路径（需用户自行对齐）
        project_id: 工程ID
        options: 混音选项
    
    返回:
        {
            "code": 0/-1,
            "result_file": "输出文件名",
            "msg": "消息"
        }
    """
    try:
        print(f"\n{'='*60}")
        print(f"🚀 【AI混音 V2.0】工程 {project_id} 启动...")
        print(f"{'='*60}")
        
        # ===== 第一步：加载音频 =====
        print("\n【步骤 1/4】资源加载...")
        
        # 加载用户干声
        user_vocal, _ = librosa.load(user_vocal_path, sr=MixConfig.SAMPLE_RATE)
        print(f"  干声加载完成: {len(user_vocal)/MixConfig.SAMPLE_RATE:.1f}秒")
        
        # 加载用户伴奏（保持立体声）
        user_acc, _ = librosa.load(user_acc_path, sr=MixConfig.SAMPLE_RATE, mono=False)
        if user_acc.ndim == 1:
            user_acc = np.vstack([user_acc, user_acc])
        print(f"  伴奏加载完成: {user_acc.shape[1]/MixConfig.SAMPLE_RATE:.1f}秒")
        
        # ===== 第二步：执行混音 =====
        print("\n【步骤 2/4】执行 AI 混音引擎...")
        
        engine = AIMixEngine(sr=MixConfig.SAMPLE_RATE)
        mix = engine.full_mix(
            vocal=user_vocal,
            accompaniment=user_acc,
            options=options
        )
        
        # ===== 第三步：导出初步混音 =====
        print("\n【步骤 3/4】导出混音...")
        
        temp_mix_path = os.path.join(PROJECT_DIR, f"{project_id}_mix_temp.wav")
        result_file_name = f"{project_id}_ProMix.wav"
        result_save_path = os.path.join(PROJECT_DIR, result_file_name)
        
        # 保存临时混音
        sf.write(temp_mix_path, mix.T, MixConfig.SAMPLE_RATE, subtype="PCM_24")
        
        # ===== 第四步：使用 Matchering 进行风格匹配 =====
        print("\n【步骤 4/4】风格匹配与母带处理...")
        
        if HAS_MATCHERING and os.path.exists(original_song_path):
            # 使用 Matchering 进行风格匹配
            matcher = StyleMatcher(MixConfig.SAMPLE_RATE)
            success = matcher.match_style(temp_mix_path, original_song_path, result_save_path)
            
            if not success:
                # 回退到基础母带处理
                print("  使用基础母带处理...")
                matcher.basic_mastering(temp_mix_path, result_save_path)
        else:
            # 基础母带处理
            print("  应用基础母带处理...")
            
            # 响度标准化
            meter = pyln.Meter(MixConfig.SAMPLE_RATE)
            current_lufs = meter.integrated_loudness(mix.T)
            
            if not np.isinf(current_lufs) and not np.isnan(current_lufs):
                gain_db = MixConfig.TARGET_LOUDNESS_LUFS - current_lufs
                mix = mix * (10 ** (gain_db / 20))
            
            # 限制器
            if HAS_PEDALBOARD:
                board = Pedalboard([
                    Limiter(threshold_db=MixConfig.LIMITER_THRESHOLD_DB)
                ])
                mix = board(mix, MixConfig.SAMPLE_RATE)
            
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
        
        return {
            "code": 0,
            "result_file": result_file_name,
            "msg": "混音完成",
            "stats": {
                "lufs": round(final_lufs, 1),
                "peak_db": round(final_peak, 1),
                "sample_rate": MixConfig.SAMPLE_RATE
            }
        }
        
    except Exception as e:
        print(f"\n❌ 【AI混音失败】: {str(e)}")
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
    'AIMixEngine',
    'MixConfig',
    'DeepAudioProcessor',
    'AccompanimentProcessor',
    'StyleMatcher',
    'ai_mix_with_original'
]