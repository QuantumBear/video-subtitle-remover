# -*- coding: utf-8 -*-
"""VSR 去字幕服务:FastAPI 包裹层,核心处理逻辑在 vsr_pipeline.py。

设计(按需求:无消息队列,全局并发限制):
  - 不引入 Redis/Celery 等外部队列组件,任务状态存进程内存表
  - 全局并发限制:同时处理中的任务数 <= MAX_CONCURRENCY(环境变量
    VSR_MAX_CONCURRENCY 或 --concurrency,默认 5);槽满时 POST 直接
    返回 503,由客户端自行重试(不做排队)
  - 每个任务一个子进程跑 vsr_pipeline.py CLI:模型隔离、崩溃不传染、
    天然与 FastAPI 进程解耦,服务重启不杀正在跑的任务由子进程独立保证有限
  - 接口:
      POST /tasks            上传视频,创建处理任务(并发满 → 503)
      GET  /tasks/{id}       查询状态与进度
      GET  /tasks/{id}/result 下载处理结果
      DELETE /tasks/{id}     清理任务文件
      GET  /health           健康检查(含并发占用)

启动:
  uvicorn service.server:app --host 0.0.0.0 --port 8000
  环境变量:VSR_MAX_CONCURRENCY(默认 5)、VSR_WORK_DIR(默认 ./service/work)
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_CLI = os.path.join(BASE_DIR, 'vsr_pipeline.py')
FFMPEG = os.path.join(BASE_DIR, 'backend', 'ffmpeg', 'macos', 'ffmpeg')

MAX_CONCURRENCY = 5
WORK_DIR = './service/work'
_progress_re = re.compile(r'进度 (\d+)/(\d+) \((.*)\)')


class Task:
    """一个处理任务:状态 + 文件路径 + 子进程句柄 + 进度信息。"""

    def __init__(self, task_id, input_path, output_path, region, white_glyph_check):
        self.id = task_id
        self.input_path = input_path
        self.output_path = output_path
        self.region = region
        self.white_glyph_check = white_glyph_check
        self.status = 'running'          # running / done / failed
        self.progress = '0/0'            # 已处理/总帧
        self.detail = ''                 # 修复/补擦统计等细节
        self.error = ''
        self.created = datetime.now(timezone.utc).isoformat()
        self.proc = None

    def snapshot(self):
        return {
            'task_id': self.id,
            'status': self.status,
            'progress': self.progress,
            'detail': self.detail,
            'error': self.error,
            'created': self.created,
        }


app = FastAPI(title='VSR 去字幕服务', version='1.0.0')
_tasks = {}              # task_id -> Task
_slot_lock = threading.Lock()
_running = 0             # 当前占用并发槽的任务数


def _slot_acquire():
    """占一个并发槽;满则返回 False。"""
    global _running
    with _slot_lock:
        if _running >= MAX_CONCURRENCY:
            return False
        _running += 1
        return True


def _slot_release():
    global _running
    with _slot_lock:
        _running -= 1


def _run_task(task: Task):
    """子进程执行流水线,逐行读 stdout 更新进度,结束后释放并发槽。"""
    try:
        cmd = [sys.executable, PIPELINE_CLI,
               '-i', task.input_path, '-o', task.output_path]
        if task.region:
            cmd += ['-c', *[str(x) for x in task.region]]
        if not task.white_glyph_check:
            cmd += ['--no-white-glyph-check']
        task.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=BASE_DIR)
        for line in task.proc.stdout:
            m = _progress_re.search(line)
            if m:
                task.progress = f'{m.group(1)}/{m.group(2)}'
                task.detail = m.group(3)
        code = task.proc.wait()
        if code == 0 and os.path.exists(task.output_path):
            task.status = 'done'
        else:
            task.status = 'failed'
            tail = (task.detail or '')[-300:]
            task.error = f'流水线退出码 {code} {tail}'
    except Exception as e:  # 任何异常都归为失败,并保证槽被释放
        task.status = 'failed'
        task.error = str(e)
    finally:
        _slot_release()


def _validate_region(region: str):
    """'ymin,ymax,xmin,xmax' → 四元组;空串返回 None。"""
    if not region or not region.strip():
        return None
    parts = [p.strip() for p in region.split(',') if p.strip()]
    if len(parts) != 4:
        raise HTTPException(400, 'region 格式应为 ymin,ymax,xmin,xmax')
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        raise HTTPException(400, 'region 必须是 4 个整数')


@app.post('/tasks', status_code=202)
async def create_task(file: UploadFile = File(...),
                      region: str = Form(''),
                      white_glyph_check: bool = Form(True)):
    """上传视频创建处理任务。并发槽满返回 503(不排队,客户端自行重试)。"""
    if not _slot_acquire():
        raise HTTPException(503, f'并发已满({MAX_CONCURRENCY}),请稍后重试')
    try:
        region_tuple = _validate_region(region)
        task_id = uuid.uuid4().hex[:12]
        task_dir = os.path.join(WORK_DIR, task_id)
        os.makedirs(task_dir, exist_ok=True)
        input_path = os.path.join(task_dir, 'input' + os.path.splitext(file.filename or '')[1])
        output_path = os.path.join(task_dir, 'output.mp4')
        with open(input_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        if os.path.getsize(input_path) == 0:
            raise HTTPException(400, '上传文件为空')
        task = Task(task_id, input_path, output_path, region_tuple, white_glyph_check)
        _tasks[task_id] = task
        threading.Thread(target=_run_task, args=(task,), daemon=True).start()
        return {'task_id': task_id, 'status': 'running'}
    except HTTPException:
        _slot_release()
        raise
    except Exception as e:
        _slot_release()
        raise HTTPException(500, str(e))


@app.get('/tasks/{task_id}')
def get_task(task_id: str):
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(404, '任务不存在')
    snap = task.snapshot()
    snap['result_ready'] = task.status == 'done'
    return snap


@app.get('/tasks/{task_id}/result')
def get_result(task_id: str):
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(404, '任务不存在')
    if task.status == 'running':
        raise HTTPException(409, '任务处理中')
    if task.status != 'done' or not os.path.exists(task.output_path):
        raise HTTPException(404, f'结果不可用: {task.error or "输出文件缺失"}')
    return FileResponse(task.output_path, media_type='video/mp4',
                        filename=f'no_sub_{task_id}.mp4')


@app.delete('/tasks/{task_id}')
def delete_task(task_id: str):
    task = _tasks.pop(task_id, None)
    if task is None:
        raise HTTPException(404, '任务不存在')
    shutil.rmtree(os.path.dirname(task.input_path), ignore_errors=True)
    return {'deleted': task_id}


@app.get('/health')
def health():
    return {'status': 'ok', 'running': _running, 'max_concurrency': MAX_CONCURRENCY,
            'tasks_total': len(_tasks)}


def main():
    global MAX_CONCURRENCY, WORK_DIR
    ap = argparse.ArgumentParser(description='VSR 去字幕 FastAPI 服务')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--concurrency', type=int,
                    default=int(os.environ.get('VSR_MAX_CONCURRENCY', 5)),
                    help='全局并发上限,默认 5(也可用环境变量 VSR_MAX_CONCURRENCY)')
    ap.add_argument('--work-dir', default=os.environ.get('VSR_WORK_DIR', './service/work'))
    args = ap.parse_args()
    MAX_CONCURRENCY = max(1, args.concurrency)
    WORK_DIR = args.work_dir
    os.makedirs(WORK_DIR, exist_ok=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
