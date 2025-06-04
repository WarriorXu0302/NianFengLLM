#!/bin/bash

# 高并发农业助手启动脚本
# 使用方法: chmod +x start_high_concurrency.sh && ./start_high_concurrency.sh

echo "==================================="
echo "启动智能农业助手 - 高并发模式"
echo "==================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python环境"
    exit 1
fi

# 安装依赖
echo "检查并安装依赖..."
pip install gunicorn gevent flask flask-cors torch torchvision Pillow aiohttp nest-asyncio

# 创建必要的目录
mkdir -p logs
mkdir -p uploads
mkdir -p uploads/pdfs

# 设置环境变量
export PYTHONUNBUFFERED=1
export FLASK_ENV=production

echo "启动配置:"
echo "- CPU核心数: $(nproc)"
echo "- 工作进程数: $(python -c 'import multiprocessing; print(min(multiprocessing.cpu_count() * 2 + 1, 8))')"
echo "- 端口: 8080"
echo "- 工作模式: 高并发 (Gunicorn + Gevent)"

# 启动服务器
echo "启动服务器..."

# 方式1: 使用配置文件启动
if [ -f "gunicorn_config.py" ]; then
    echo "使用 gunicorn_config.py 配置文件启动"
    gunicorn -c gunicorn_config.py App.app:app
else
    # 方式2: 直接命令行启动
    echo "使用命令行参数启动"
    gunicorn \
        --bind 0.0.0.0:8080 \
        --workers $(python -c 'import multiprocessing; print(min(multiprocessing.cpu_count() * 2 + 1, 8))') \
        --worker-class gevent \
        --worker-connections 1000 \
        --timeout 120 \
        --keepalive 5 \
        --max-requests 1000 \
        --max-requests-jitter 100 \
        --preload \
        --access-logfile ./logs/access.log \
        --error-logfile ./logs/error.log \
        --log-level info \
        --pid ./logs/gunicorn.pid \
        App.app:app
fi

echo "服务器已停止" 