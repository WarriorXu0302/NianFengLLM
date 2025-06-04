# Gunicorn高并发配置文件
# 使用方法: gunicorn -c gunicorn_config.py App.app:app

import multiprocessing
import os

# 服务器绑定配置
bind = "0.0.0.0:8080"

# 工作进程数 - 根据CPU核心数动态调整
workers = min(multiprocessing.cpu_count() * 2 + 1, 8)  # 最多8个进程

# 工作进程类型 - 使用gevent以支持异步I/O
worker_class = "gevent"

# 每个工作进程的协程数量
worker_connections = 1000

# 请求超时配置
timeout = 120
keepalive = 5

# 内存和连接配置
max_requests = 1000  # 每个worker处理完1000个请求后重启，防止内存泄漏
max_requests_jitter = 100  # 随机抖动避免同时重启
worker_tmp_dir = "/dev/shm"  # 使用内存文件系统提高性能

# 日志配置
accesslog = "./logs/access.log"
errorlog = "./logs/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# 预载应用
preload_app = True

# 守护进程模式
daemon = False

# 用户权限（生产环境中使用）
# user = "www-data"
# group = "www-data"

# 进程文件
pidfile = "./logs/gunicorn.pid"

# 创建日志目录
def when_ready(server):
    os.makedirs("./logs", exist_ok=True)
    server.log.info("服务器准备就绪，高并发模式启动完成")

# 工作进程启动回调
def worker_init(worker):
    worker.log.info(f"工作进程 {worker.pid} 启动")

# 性能优化配置
def post_fork(server, worker):
    server.log.info(f"Worker spawned (pid: {worker.pid})")

def pre_fork(server, worker):
    pass

def worker_exit(server, worker):
    server.log.info(f"Worker exited (pid: {worker.pid})")

# 优雅关闭配置
graceful_timeout = 30 