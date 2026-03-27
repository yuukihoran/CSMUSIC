import os
import uuid
import webbrowser
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from threading import Timer
import shutil

# 导入配置
from config import PROJECT_NAME, BASE_DIR, STATIC_DIR, UPLOAD_DIR, PROJECT_DIR

# 导入模块
from modules.song_splitter import split_song
from modules.ai_mixer import ai_mix_with_original
from modules.project_manager import create_project, get_all_projects

# 初始化FastAPI
app = FastAPI(title=PROJECT_NAME)

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 确保目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# 支持的音频格式
SUPPORTED_FORMATS = {"wav", "mp3", "flac", "m4a", "ogg"}

# ====================== 工具函数 ======================
def save_upload_file(file: UploadFile, prefix: str) -> str:
    """保存上传文件"""
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {file_ext}，仅支持{SUPPORTED_FORMATS}"
        )
    
    file_id = str(uuid.uuid4())[:8]
    save_name = f"{prefix}_{file_id}.{file_ext}"
    save_path = os.path.join(UPLOAD_DIR, save_name)
    
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return save_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

def clean_temp_files():
    """清理过期临时文件"""
    import time
    now = time.time()
    for file in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, file)
        if os.path.isfile(file_path) and now - os.path.getctime(file_path) > 3600:
            try:
                os.remove(file_path)
                print(f"【清理】删除过期临时文件: {file_path}")
            except:
                pass

# ====================== 接口定义 ======================
# 首页
@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse(
            content="<h1>前端页面未找到</h1><p>请确保index.html文件在static目录下</p>",
            status_code=404
        )
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# 1. 歌曲拆分接口
@app.post("/api/split")
async def api_split_song(original_song: UploadFile = File(...)):
    try:
        song_save_path = save_upload_file(original_song, "split")
        project_id = create_project("歌曲拆分", [original_song.filename])
        
        result = split_song(song_save_path, project_id)
        if result["code"] != 0:
            raise HTTPException(status_code=500, detail=result["msg"])
        
        return {
            "code": 0,
            "project_id": project_id,
            "acc_file": result["acc_file"],
            "vocal_file": result["vocal_file"],
            "msg": result["msg"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"拆分失败: {str(e)}")

# 2. AI混音接口（完整专业流程）
@app.post("/api/mix")
async def api_mix(
    user_acc: UploadFile = File(..., description="用户伴奏文件"),
    user_vocal: UploadFile = File(..., description="用户干声文件"),
    original_song: UploadFile = File(None, description="原曲文件（可选，用于风格匹配）"),
    enable_pitch_correction: bool = Form(True, description="是否启用修音"),
    pitch_correction_strength: float = Form(0.3, description="修音强度"),
    enable_denoise: bool = Form(True, description="是否启用降噪"),
    enable_sidechain: bool = Form(True, description="是否启用侧链压缩")
):
    """
    AI 专业混音接口
    
    完整流程：
    1. 轻微修音 - 自动检测调性，校正音高
    2. 深度降噪 - 多方案降噪
    3. 去齿音 - MultibandCompressor 等效实现
    4. 两次压缩 + 两次EQ
    5. 动态共振控制
    6. 混响空间感
    7. 伴奏侧链处理
    8. 风格匹配（可选）
    9. 母带处理输出
    
    参数:
        user_acc: 用户伴奏文件
        user_vocal: 用户干声文件（需已对齐）
        original_song: 原曲文件（可选，用于风格匹配学习）
        enable_pitch_correction: 是否启用轻微修音
        pitch_correction_strength: 修音强度 (0.0-1.0)
        enable_denoise: 是否启用深度降噪
        enable_sidechain: 是否启用侧链压缩
    """
    try:
        # 保存上传文件
        acc_save_path = save_upload_file(user_acc, "mix_acc")
        vocal_save_path = save_upload_file(user_vocal, "mix_vocal")
        
        # 原曲文件（可选）
        original_save_path = None
        original_filename = None
        if original_song and original_song.filename:
            original_save_path = save_upload_file(original_song, "mix_original")
            original_filename = original_song.filename
        
        # 创建工程
        file_list = [user_acc.filename, user_vocal.filename]
        if original_filename:
            file_list.insert(0, original_filename)
        project_id = create_project("AI混音", file_list)
        
        # 混音选项
        options = {
            "enable_pitch_correction": enable_pitch_correction,
            "pitch_correction_strength": pitch_correction_strength,
            "enable_denoise": enable_denoise,
            "enable_sidechain": enable_sidechain,
        }
        
        # 执行混音
        result = ai_mix_with_original(
            original_song_path=original_save_path or "",
            user_acc_path=acc_save_path,
            user_vocal_path=vocal_save_path,
            project_id=project_id,
            options=options
        )
        
        if result["code"] != 0:
            raise HTTPException(status_code=500, detail=result["msg"])
        
        return {
            "code": 0,
            "project_id": project_id,
            "result_file": result["result_file"],
            "msg": result["msg"],
            "stats": result.get("stats", {})
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"混音失败: {str(e)}")

# 3. 工程记录接口
@app.get("/api/projects")
async def api_get_projects():
    try:
        projects = get_all_projects()
        return {"code": 0, "projects": projects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取工程失败: {str(e)}")

# 4. 文件下载接口
@app.get("/api/download/{file_name}")
async def api_download(file_name: str):
    if ".." in file_name or "/" in file_name or "\\" in file_name:
        raise HTTPException(status_code=400, detail="文件名非法")
    
    file_path = os.path.join(PROJECT_DIR, file_name)
    if not os.path.exists(file_path):
        file_path = os.path.join(UPLOAD_DIR, file_name)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(
        path=file_path,
        filename=file_name,
        media_type="audio/wav" if file_name.endswith(".wav") else "audio/mpeg"
    )

# 自动打开浏览器
def open_browser():
    try:
        webbrowser.open("http://127.0.0.1:8000")
    except:
        print("【提示】请手动访问 http://127.0.0.1:8000")

# 主函数
if __name__ == "__main__":
    print(f"【{PROJECT_NAME}】服务启动中...")
    clean_temp_files()
    Timer(1, open_browser).start()
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        reload=False
    )