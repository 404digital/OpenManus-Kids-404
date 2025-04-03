from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
import os
import uuid
import time
import asyncio
from typing import Optional
import asyncio
import requests
from typing import Optional, List


# 火山引擎ASR配置
class AsrConfig:
    def __init__(self, app_id: str, access_token: str, submit_url: str, query_url: str):
        self.app_id = app_id
        self.access_token = access_token
        self.submit_url = submit_url
        self.query_url = query_url

def load_asr_config() -> AsrConfig:
    config = configparser.ConfigParser()
    config.read('config/asr.toml')
    # 去除配置值的引号并验证格式
    app_id = config['asr']['app_id'].strip('"').strip("'")
    access_token = config['asr']['access_token'].strip('"').strip("'")
    submit_url = config['asr']['submit_url'].strip('"').strip("'")
    query_url = config['asr']['query_url'].strip('"').strip("'")

    print(f"加载ASR配置: app_id={app_id} access_token={access_token[:3]}... submit_url={submit_url}")

    return AsrConfig(
        app_id=app_id,
        access_token=access_token,
        submit_url=submit_url,
        query_url=query_url
    )

from pydantic import BaseModel
import requests
import uuid
import json
from fastapi import status

from minio import Minio
from minio.error import S3Error
import tempfile
import configparser
from pathlib import Path
from datetime import datetime

class MinioConfig:
    def __init__(self, base_url: str, access_key: str, secret_key: str):
        self.base_url = base_url
        self.access_key = access_key
        self.secret_key = secret_key

def load_minio_config() -> MinioConfig:
    config = configparser.ConfigParser()
    config.read('config/minio.toml')
    # 去除配置值的引号并验证格式
    base_url = config['minio']['base_url'].strip('"').strip("'")
    access_key = config['minio']['access_key'].strip('"').strip("'")
    secret_key = config['minio']['secret_key'].strip('"').strip("'")

    print(f"加载Minio配置: base_url={base_url} access_key={access_key[:3]}... secret_key={secret_key[:3]}...")  # 安全打印部分信息

    return MinioConfig(
        base_url=base_url,
        access_key=access_key,
        secret_key=secret_key
    )

def get_minio_client() -> Minio:
    cfg = load_minio_config()
    print("cfg", cfg.base_url)
    return Minio(
        endpoint=cfg.base_url,
        access_key=cfg.access_key,
        secret_key=cfg.secret_key,
        secure=True
    )

class AsrReq(BaseModel):
    url: str

class Word(BaseModel):
    text: str
    start_time: int
    end_time: int
    blank_duration: int | None = None

class Utterance(BaseModel):
    text: str
    start_time: int
    end_time: int
    definite: bool | None = None
    words: list[Word] | None = None

class AudioInfo(BaseModel):
    duration: int

class Result(BaseModel):
    text: str | None = None
    utterances: list[Utterance] | None = None

class QueryResponse(BaseModel):
    result: Result | None = None
    audio_info: AudioInfo | None = None

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录（ui目录）
app.mount("/static", StaticFiles(directory="ui"), name="static")

# 注册ASR路由
@app.post("/api/volcengine/asr")
async def handle_asr(request: AsrReq):
    """处理语音识别请求"""
    user_id = str(uuid.uuid4())

    try:
        # 提交识别任务
        request_id = await submit_task(request.url, user_id)
        print("request_id:", request_id)
        # 轮询查询结果（最多等待5秒）
        start_time = time.time()
        while time.time() - start_time < 5:
            result = await query_result(request_id)
            print("asr 查询:",result)

            if result !=None and result.result and result.result.text:
                return {
                    "success":True,
                    "code": 0,
                    "data": result.result.dict(exclude_unset=True)
                }

            await asyncio.sleep(0.1)

        return {
            "success": False,
            "code": -1,
            "message": "ASR结果查询超时"
        }

    except Exception as e:
        # 确保返回符合FastAPI响应格式
        print("错误:",e)
        content = {
            "code": e.status_code,
            "message": e.detail
        }

@app.get("/", response_class=HTMLResponse)
async def serve_spa():
    try:
        with open("ui/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Index file not found")

# 处理前端路由的fallback（支持SPA路由）
@app.get("/{full_path:path}", response_class=HTMLResponse)
async def catch_all(full_path: str):
    file_path = os.path.join("ui", full_path)

    # 优先返回真实存在的静态文件（带正确MIME类型）
    if os.path.isfile(file_path):
        try:
            # 根据文件扩展名设置精确的MIME类型
            content_type = "text/plain; charset=utf-8"
            if file_path.endswith(".css"):
                content_type = "text/css; charset=utf-8"
            elif file_path.endswith(".js"):
                content_type = "application/javascript; charset=utf-8"
            elif file_path.endswith(".png"):
                content_type = "image/png"
            elif file_path.endswith(".jpg") or file_path.endswith(".jpeg"):
                content_type = "image/jpeg"
            elif file_path.endswith(".svg"):
                content_type = "image/svg+xml"
            elif file_path.endswith(".woff2"):
                content_type = "font/woff2"
            elif file_path.endswith(".html"):
                content_type = "text/html; charset=utf-8"

            # 根据文件类型选择读取模式
            if content_type.startswith("text") or "application/javascript" in content_type:
                with open(file_path, "r", encoding='utf-8') as f:
                    return Response(
                        content=f.read(),
                        media_type=content_type,
                        headers={"Cache-Control": "public, max-age=3600"}
                    )
            else:
                with open(file_path, "rb") as f:
                    return Response(
                        content=f.read(),
                        media_type=content_type,
                        headers={"Cache-Control": "public, max-age=3600"}
                    )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")

    # 处理SPA路由回退
    try:
        with open("ui/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="未找到页面")


async def submit_task(audio_url: str, user_id: str):
    """提交ASR任务"""
    cfg = load_asr_config()
    headers = {
        "X-Api-App-Key": cfg.app_id,
        "X-Api-Access-Key": cfg.access_token,
        "X-Api-Resource-Id": "volc.bigasr.auc",
        "X-Api-Request-Id": str(uuid.uuid4()),
        "X-Api-Sequence": "-1",
        "Content-Type": "application/json"
    }

    payload = {
        "user": {"uid": user_id},
        "audio": {
            "format": "mp3",
            "url": audio_url
        },
        "request": {
            "model_name": "bigmodel",
            "enable_itn": True,
            "enable_punc": True
        }
    }

    try:
        response = requests.post(cfg.submit_url, headers=headers, json=payload)
        response.raise_for_status()

        # 检查响应头状态
        status_code = response.headers.get("X-Api-Status-Code", "")
        if status_code != "20000000":
            error_msg = response.headers.get("X-Api-Message", "Unknown error")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"ASR提交失败: {status_code} - {error_msg}"
            )


        return response.headers["X-Api-Request-Id"]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"ASR服务请求失败: {str(e)}"
        )

async def query_result(request_id: str):
    """查询ASR结果"""
    cfg = load_asr_config()
    headers = {
        "X-Api-App-Key": cfg.app_id,
        "X-Api-Access-Key": cfg.access_token,
        "X-Api-Resource-Id": "volc.bigasr.auc",
        "X-Api-Request-Id": request_id,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(cfg.query_url, headers=headers, json={})
        response.raise_for_status()
        print("ASR查询, 响应:", response)
        # 检查响应头状态
        status_code = response.headers.get("X-Api-Status-Code", "")
        print("status code:", status_code)
        if status_code == "20000001":
            return None
        if status_code != "20000000":
            error_msg = response.headers.get("X-Api-Message", "Unknown error")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"ASR查询失败: {status_code} - {error_msg}"
            )


        return QueryResponse(**response.json())

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"ASR查询请求失败: {str(e)}"
        )

# Minio文件上传配置
MAX_FILE_SIZE = 10_000_000  # 10MB限制
DEFAULT_BUCKET = "voices"

from fastapi import UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse

@app.post("/api/minio/upload")
async def minio_upload(
    file: UploadFile = File(...),
    bucket: str = Form(DEFAULT_BUCKET),
    client: Minio = Depends(get_minio_client),
    cfg: MinioConfig = Depends(load_minio_config)
):
    """处理文件上传到Minio存储"""
    try:
        # 读取文件内容并验证大小
        print("开始处理文件上传...")
        content = await file.read()
        await file.seek(0)  # 重置文件指针
        file_size = len(content)
        print(f"文件大小: {file_size}字节")

        if file_size > MAX_FILE_SIZE:
            print("文件大小超过限制")
            return JSONResponse(
                {"success": False, "message": "文件大小超过10MB限制"},
                status_code=400
            )

        # 生成存储路径
        file_ext = Path(file.filename).suffix[1:] if file.filename else ""
        print(f"文件扩展名: {file_ext}")
        if not file_ext:
            return JSONResponse(
                {"success": False, "message": "无效的文件扩展名"},
                status_code=400
            )

        date_path = datetime.now().strftime("%Y-%m-%d")
        object_name = f"{date_path}/{uuid.uuid4()}.{file_ext}"
        print(f"生成对象名称: {object_name}")

        # 异步检查存储桶是否存在
        print("检查存储桶是否存在...")
        # bucket_exists = await asyncio.to_thread(client.bucket_exists, bucket)
        # if not bucket_exists:
        #     print("存储桶不存在")
        #     return JSONResponse(
        #         {"success": False, "message": "存储桶不存在"},
        #         status_code=400
        #     )

        # 使用异步上下文管理临时文件
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            print(f"创建临时文件: {tmp_file.name}")

            # 异步写入文件内容
            await asyncio.to_thread(tmp_file.write, content)
            await asyncio.to_thread(tmp_file.flush)

            # 异步上传到Minio（带超时机制）
            print("开始上传到Minio...")
            try:

                client.fput_object(
                    bucket_name=bucket,
                    object_name=object_name,
                    file_path=tmp_file.name,
                    content_type=file.content_type)


                print("上传成功完成")

            except Exception as e:
                print(f"上传过程中发生错误: {str(e)}")
                import traceback
                traceback.print_exc()  # 打印完整堆栈跟踪
                raise
        print("上传完成.......")
        return JSONResponse({
            "success": True,
            "size": file_size,
            "bucket": bucket,
            "file": f"https://{cfg.base_url}/{bucket}/{object_name}"
        })

    except S3Error as e:
        return JSONResponse(
            {"success": False, "message": f"存储服务错误: {str(e)}"},
            status_code=500
        )
    except Exception as e:
        return JSONResponse(
            {"success": False, "message": f"上传失败: {str(e)}"},
            status_code=500
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8181,
        access_log=False
    )
