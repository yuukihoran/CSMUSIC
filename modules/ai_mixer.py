"""
AI 专业贴唱混音模块 V4.0 (重构版)
=============================
完整的混音引擎，整合所有专业处理环节
基于研究成果重构，集成专业级人声处理算法

混音流程（专业标准）：
1. 智能降噪 - 多方案降噪处理
2. 轻微修音 - 自动音高校正（可选）
3. 低切滤波 - 去除低频噪声
4. 去齿音 - 专业多段压缩实现
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
- audio_effects: 专业音频效果器封装
- librosa: 音频分析
- pyloudnorm: 响度标准化
- matchering: 风格匹配
"""
import os
import numpy as np
import soundfile as sf
import librosa
import warnings
from typing import Dict, Optional, Tuple, List
warnings.filterwarnings('ignore')

# 导入专业音频效果器
from .audio_effects import (
    ProfessionalVocalChain, VocalChainConfig, AccompanimentConfig,
    MultibandCompressor, DeEsser, ResonanceController, PitchCorrector,
    NoiseReducer, SidechainCompressor, LoudnessNormalizer,
    ensure_mono, ensure_stereo, pedalboard_process_mono,
    HAS_PEDALBOARD, HAS_PYWORLD, HAS_NOISEREDUCE, HAS_MATCHERING,
    HAS_SCIPY, HAS_PYLOUDNORM, HAS_LIBROSA
)

# 尝试导入 matchering 用于风格匹配
try:
    import matchering as mg
    HAS_MATCHERING = True
except ImportError:
    HAS_MATCHERING = False
    print("【提示】Matchering 未安装，风格匹配功能将受限")

# 尝试导入 pedalboard
if HAS_PEDALBOARD:
    from pedalboard import (
        Pedalboard, Compressor, Reverb, HighpassFilter, LowpassFilter,
        PeakFilter, LowShelfFilter, HighShelfFilter, Limiter, NoiseGate,
        Delay, Gain
    )

# 配置导入
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import SAMPLE_RATE, PROJECT_DIR
except ImportError:
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
#                           智能调性检测
# ============================================================================
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


def detect_bpm(audio: np.ndarray, sr: int) -> float:
    """检测音频 BPM"""
    try:
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        return float(tempo)
    except:
        return 120.0


# ============================================================================
#                           伴奏处理模块
# ============================================================================
class AccompanimentProcessor:
    """伴奏处理器"""
    
    def __init__(self, sr: int = MixConfig.SAMPLE_RATE):
        self.sr = sr
        self.sidechain = SidechainCompressor(sr)
    
    def apply_mid_side_eq(self, audio: np.ndarray) -> np.ndarray:
        """Mid/Side EQ - 为人声让出空间"""
        print("【伴奏 Mid/Side EQ】为人声让出空间")
        
        if audio.ndim == 1:
            audio = np.vstack([audio, audio])
        
        if not HAS_PEDALBOARD:
            return audio
        
        try:
            # 转换为 Mid/Side
            mid = (audio[0] + audio[1]) / 2
            side = (audio[0] - audio[1]) / 2
            
            # 在 Mid 通道挖一个坑给人声
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
            print("【侧链压缩】处理中...")
            result = self.sidechain.process(result, vocal, reduction_db=3.0)
            print("  ✅ 侧链压缩完成")
        
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
#                           AI混音引擎
# ============================================================================
class AIMixEngine:
    """
    AI 混音引擎 V4.0
    ================
    整合所有处理环节的完整混音引擎
    基于专业音频效果器封装模块
    """
    
    def __init__(self, sr: int = MixConfig.SAMPLE_RATE):
        self.sr = sr
        
        # 创建配置
        vocal_config = VocalChainConfig(
            sample_rate=sr,
            low_cut_hz=MixConfig.LOW_CUT_HZ,
            deess_threshold_db=MixConfig.DEESS_THRESHOLD,
            deess_frequency=MixConfig.DEESS_FREQUENCY,
            deess_ratio=MixConfig.DEESS_RATIO,
            comp1_threshold_db=MixConfig.COMP1_THRESHOLD,
            comp1_ratio=MixConfig.COMP1_RATIO,
            comp1_attack_ms=MixConfig.COMP1_ATTACK,
            comp1_release_ms=MixConfig.COMP1_RELEASE,
            comp2_threshold_db=MixConfig.COMP2_THRESHOLD,
            comp2_ratio=MixConfig.COMP2_RATIO,
            comp2_attack_ms=MixConfig.COMP2_ATTACK,
            comp2_release_ms=MixConfig.COMP2_RELEASE,
            eq_low_shelf_hz=MixConfig.EQ1_LOW_SHELF_HZ,
            eq_low_shelf_gain_db=MixConfig.EQ1_LOW_SHELF_GAIN,
            eq_presence_hz=MixConfig.EQ1_PRESENCE_HZ,
            eq_presence_gain_db=MixConfig.EQ1_PRESENCE_GAIN,
            eq_air_hz=MixConfig.EQ1_AIR_HZ,
            eq_air_gain_db=MixConfig.EQ1_AIR_GAIN,
            reverb_room_size=MixConfig.REVERB_ROOM_SIZE,
            reverb_damping=MixConfig.REVERB_DAMPING,
            reverb_wet_level=MixConfig.REVERB_WET_LEVEL,
            limiter_threshold_db=MixConfig.LIMITER_THRESHOLD,
            target_lufs=MixConfig.TARGET_LOUDNESS_LUFS
        )
        
        # 初始化处理器
        self.vocal_processor = ProfessionalVocalChain(vocal_config)
        self.acc_processor = AccompanimentProcessor(sr)
        self.style_matcher = StyleMatcher(sr)
        self.loudness_normalizer = LoudnessNormalizer(sr)
    
    def balance_volumes(self, vocal: np.ndarray, 
                        accompaniment: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """音量平衡"""
        print("【音量平衡】")
        
        try:
            acc_stereo = ensure_stereo(accompaniment)
            acc_lufs = self.loudness_normalizer.measure_loudness(acc_stereo)
            
            if np.isinf(acc_lufs) or np.isnan(acc_lufs):
                acc_lufs = -18.0
            
            target_vocal_lufs = acc_lufs + MixConfig.VOCAL_TO_ACC_OFFSET
            
            vocal_stereo_temp = ensure_stereo(vocal)
            vocal_lufs = self.loudness_normalizer.measure_loudness(vocal_stereo_temp)
            
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
            # 响度标准化
            current_lufs = self.loudness_normalizer.measure_loudness(mix)
            
            if not np.isinf(current_lufs) and not np.isnan(current_lufs):
                gain_db = MixConfig.TARGET_LOUDNESS_LUFS - current_lufs
                mix = mix * (10 ** (gain_db / 20))
                print(f"  响度调整: {gain_db:.1f} dB")
            
            # 限制器
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
#                           对外接口
# ============================================================================
def ai_mix_with_original(original_song_path: str,
                         user_acc_path: str,
                         user_vocal_path: str,
                         project_id: str,
                         options: dict = None) -> dict:
    """
    AI 混音主接口 V4.0 (重构版)
    
    完整混音流程：
    1. 加载音频
    2. 检测调性
    3. 处理人声（降噪、修音、去齿音、压缩、EQ、共振控制、混响）
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
        print(f"🚀 【AI混音 V4.0】工程 {project_id} 启动...")
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
        bpm = detect_bpm(ensure_mono(user_acc), MixConfig.SAMPLE_RATE)
        print(f"  检测到调性: {key}大调, BPM: {bpm:.0f}")
        
        # ===== 第三步：处理人声 =====
        print("\n【步骤 3/7】处理人声...")
        engine = AIMixEngine(sr=MixConfig.SAMPLE_RATE)
        processed_vocal = engine.vocal_processor.process(
            user_vocal,
            key=key,
            enable_pitch_correction=options["enable_pitch_correction"],
            pitch_correction_strength=options["pitch_correction_strength"],
            enable_denoise=options["enable_denoise"],
            enable_deess=True,
            enable_reverb=True
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
        
        final_lufs = engine.loudness_normalizer.measure_loudness(final_audio)
        final_peak = 20 * np.log10(np.max(np.abs(final_audio)) + 1e-10)
        
        print(f"\n{'='*60}")
        print(f"✅ 【AI混音完成】")
        print(f"{'='*60}")
        print(f"  📁 输出文件: {result_file_name}")
        print(f"  📊 最终响度: {final_lufs:.1f} LUFS")
        print(f"  📊 峰值电平: {final_peak:.1f} dB")
        print(f"  📊 采样率: {MixConfig.SAMPLE_RATE} Hz")
        print(f"  📊 调性: {key}大调")
        print(f"{'='*60}\n")
        
        return {
            "code": 0,
            "result_file": result_file_name,
            "msg": "混音完成",
            "stats": {
                "lufs": float(final_lufs),
                "peak_db": float(final_peak),
                "sample_rate": int(MixConfig.SAMPLE_RATE),
                "key": key,
                "bpm": float(bpm)
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
                "key": "C",
                "bpm": 120.0
            }
        }


# ============================================================================
#                           导出
# ============================================================================
__all__ = [
    'AIMixEngine',
    'MixConfig',
    'AccompanimentProcessor',
    'StyleMatcher',
    'ai_mix_with_original',
    'detect_key',
    'detect_bpm'
]