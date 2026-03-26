import os
import json
import time
from config import PROJECT_DIR, UPLOAD_DIR

PROJECT_INDEX = os.path.join(PROJECT_DIR, "project_index.json")

# 初始化工程索引
if not os.path.exists(PROJECT_INDEX):
    with open(PROJECT_INDEX, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

def create_project(project_type, file_list):
    """
    创建新工程
    :param project_type: 工程类型（split/mix）
    :param file_list: 工程包含的文件列表
    :return: 工程ID
    """
    import uuid
    project_id = str(uuid.uuid4())[:8]
    create_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    project_info = {
        "project_id": project_id,
        "project_type": "歌曲拆分" if project_type == "split" else "AI混音",
        "create_time": create_time,
        "file_list": file_list
    }

    # 写入索引
    with open(PROJECT_INDEX, "r", encoding="utf-8") as f:
        index = json.load(f)
    index.append(project_info)
    with open(PROJECT_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    
    print(f"【工程管理】创建新工程：{project_id}")
    return project_id

def get_all_projects():
    """获取所有工程记录"""
    with open(PROJECT_INDEX, "r", encoding="utf-8") as f:
        index = json.load(f)
    # 按创建时间倒序
    index.sort(key=lambda x: x["create_time"], reverse=True)
    return index