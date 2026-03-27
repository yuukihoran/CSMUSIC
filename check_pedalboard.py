import sys
print(f"当前 Python 版本: {sys.version}")
print(f"当前 Python 路径: {sys.executable}")

print("\n--- 1. 尝试导入 pedalboard 主模块 ---")
try:
    import pedalboard
    print(f"✅ pedalboard 主模块导入成功！版本: {pedalboard.__version__}")
except Exception as e:
    print(f"❌ 主模块导入失败: {e}")
    exit()

print("\n--- 2. 尝试导入核心组件 ---")
required_components = [
    "Pedalboard", "Compressor", "Reverb", "HighpassFilter", 
    "LowpassFilter", "PeakFilter", "LowShelfFilter", "HighShelfFilter",
    "Limiter", "NoiseGate", "MultibandCompressor"
]

failed = []
for comp in required_components:
    try:
        exec(f"from pedalboard import {comp}")
        print(f"  ✅ {comp}")
    except Exception as e:
        print(f"  ❌ {comp} 导入失败: {e}")
        failed.append(comp)

print("\n--- 3. 尝试导入 DeEsser ---")
try:
    from pedalboard import DeEsser
    print("✅ DeEsser 导入成功")
except Exception as e:
    print(f"⚠️ DeEsser 导入失败 (这是已知的 Windows 旧版本问题): {e}")

print("\n" + "="*50)
if not failed:
    print("🎉 所有核心组件正常！问题出在启动命令上。")
else:
    print(f"❌ 以下组件导入失败: {failed}")
print("="*50)