from flask import Flask, request, jsonify, send_from_directory
import torch
from torchvision import transforms
from PIL import Image
import json
import os
import asyncio
import logging
import nest_asyncio
from flask_cors import CORS
from _003_SfEfficientNet import SfEfficientNet
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import atexit
import signal
import sys
import requests
import aiohttp
import PyPDF2
import glob
import shutil
from datetime import datetime
import numpy as np
import time
import threading
from functools import lru_cache, wraps
import hashlib
import pickle
import uuid

# Import LightRAG components
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

# Apply nest_asyncio to allow asyncio to work in Flask
nest_asyncio.apply()

# Configure logging
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger()

# Configure application
app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})  # 在生产环境中应该限制origins

# 高并发配置
app.config['OLLAMA_HOST'] = "http://localhost:11435"
app.config['LLM_MODEL'] = "deepseek-r1:7b"  # 对话使用deepseek模型
app.config['RAG_MODEL'] = "llama2:7b"       # RAG检索使用llama2模型
app.config['EMBEDDING_MODEL'] = "nomic-embed-text"  # 默认嵌入模型
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# 高并发配置 - 增加线程池大小
MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4)  # 根据CPU核心数动态调整
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# 异步HTTP会话池
http_session = None
session_lock = threading.Lock()

# 缓存机制 (扩展版 - 添加对话历史管理)
class SimpleCache:
    def __init__(self, max_size=1000, ttl=300):  # 5分钟TTL
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.RLock()
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                # 检查是否过期
                if time.time() - self.access_times[key] < self.ttl:
                    self.access_times[key] = time.time()
                    return self.cache[key]
                else:
                    # 过期删除
                    del self.cache[key]
                    del self.access_times[key]
            return None
    
    def set(self, key, value):
        with self.lock:
            # 如果缓存满了，删除最旧的项目
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times, key=self.access_times.get)
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def delete(self, key):
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                del self.access_times[key]

# 对话历史管理类
class ConversationManager:
    def __init__(self, max_conversations=1000, max_history_per_conversation=50):
        self.conversations = {}  # conversation_id -> list of messages
        self.access_times = {}
        self.max_conversations = max_conversations
        self.max_history_per_conversation = max_history_per_conversation
        self.lock = threading.RLock()
    
    def add_message(self, conversation_id, role, content):
        """添加消息到对话历史"""
        with self.lock:
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []
            
            # 添加新消息
            self.conversations[conversation_id].append({
                'role': role,
                'content': content,
                'timestamp': time.time()
            })
            
            # 限制历史长度
            if len(self.conversations[conversation_id]) > self.max_history_per_conversation:
                # 保留系统消息和最近的消息
                messages = self.conversations[conversation_id]
                system_messages = [msg for msg in messages if msg['role'] == 'system']
                recent_messages = [msg for msg in messages if msg['role'] != 'system'][-self.max_history_per_conversation:]
                self.conversations[conversation_id] = system_messages + recent_messages
            
            self.access_times[conversation_id] = time.time()
            
            # 如果对话数量超过限制，删除最旧的对话
            if len(self.conversations) > self.max_conversations:
                oldest_id = min(self.access_times, key=self.access_times.get)
                del self.conversations[oldest_id]
                del self.access_times[oldest_id]
    
    def get_conversation_history(self, conversation_id, include_system=True):
        """获取对话历史"""
        with self.lock:
            if conversation_id not in self.conversations:
                return []
            
            history = self.conversations[conversation_id]
            if not include_system:
                history = [msg for msg in history if msg['role'] != 'system']
            
            self.access_times[conversation_id] = time.time()
            return history.copy()
    
    def clear_conversation(self, conversation_id):
        """清除指定对话"""
        with self.lock:
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
                del self.access_times[conversation_id]
    
    def get_conversation_summary(self, conversation_id):
        """获取对话摘要信息"""
        with self.lock:
            if conversation_id not in self.conversations:
                return None
            
            history = self.conversations[conversation_id]
            return {
                'conversation_id': conversation_id,
                'message_count': len(history),
                'last_activity': self.access_times.get(conversation_id, 0),
                'created': history[0]['timestamp'] if history else None
            }
    
    def list_conversations(self):
        """列出所有对话"""
        with self.lock:
            return list(self.conversations.keys())

# 全局缓存实例 (更新)
response_cache = SimpleCache(max_size=500, ttl=300)  # 响应缓存
rag_cache = SimpleCache(max_size=200, ttl=600)      # RAG查询缓存
model_cache = SimpleCache(max_size=100, ttl=1800)   # 模型预测缓存

# 全局对话管理器
conversation_manager = ConversationManager()

# 生成对话ID的函数
def generate_conversation_id():
    """生成唯一的对话ID"""
    return str(uuid.uuid4())[:8]

# Configure working directories
WORKING_DIR = "../examples/Agriculture_LightRAG"
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(os.path.join(app.static_folder), exist_ok=True)

# Configure device for PyTorch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_classes = 39

# Image transformation for plant disease detection
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Initialize RAG instance with connection pooling
rag_instance = None
loop = None
rag_lock = threading.RLock()

# 高并发的HTTP会话管理 - 修复版本
async def get_http_session():
    """获取HTTP会话，支持事件循环重创建"""
    global http_session
    
    # 检查当前事件循环状态
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            # 如果事件循环关闭，重置会话
            http_session = None
    except RuntimeError:
        # 如果没有事件循环，重置会话
        http_session = None
    
    if http_session is None or http_session.closed:
        connector = aiohttp.TCPConnector(
            limit=100,          # 总连接池大小
            limit_per_host=30,  # 每个主机的连接数
            ttl_dns_cache=300,  # DNS缓存TTL
            use_dns_cache=True,
        )
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        http_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
    return http_session

# 高并发的缓存装饰器
def cache_result(cache_instance, ttl=None):
    """缓存装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = hashlib.md5(
                (str(func.__name__) + str(args) + str(kwargs)).encode()
            ).hexdigest()
            
            # 尝试从缓存获取
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                logger.debug(f"缓存命中: {func.__name__}")
                return cached_result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result)
            logger.debug(f"缓存存储: {func.__name__}")
            return result
        return wrapper
    return decorator

# 安全关闭资源的函数 (改进版)
def cleanup_resources():
    """清理所有资源，关闭线程池和异步循环"""
    global executor, rag_instance, loop, http_session
    
    logger.info("正在清理应用资源...")
    
    # 清理缓存
    try:
        response_cache.clear()
        rag_cache.clear()
        model_cache.clear()
        logger.info("缓存清理完成")
    except Exception as e:
        logger.error(f"清理缓存时出错: {str(e)}")
    
    # 关闭HTTP会话 - 改进版本
    if http_session:
        try:
            logger.info("关闭HTTP会话...")
            # 检查会话状态
            if not http_session.closed:
                # 尝试在合适的事件循环中关闭会话
                try:
                    # 检查当前是否有运行的事件循环
                    current_loop = None
                    try:
                        current_loop = asyncio.get_running_loop()
                    except RuntimeError:
                        # 没有运行的事件循环
                        pass
                    
                    if current_loop and not current_loop.is_closed():
                        # 在当前循环中关闭
                        current_loop.run_until_complete(http_session.close())
                    else:
                        # 创建临时循环来关闭会话
                        temp_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(temp_loop)
                        try:
                            temp_loop.run_until_complete(http_session.close())
                        finally:
                            temp_loop.close()
                except Exception as close_error:
                    logger.warning(f"关闭HTTP会话时出错: {str(close_error)}")
                    # 强制标记为已关闭
                    try:
                        http_session._closed = True
                    except:
                        pass
            http_session = None
            logger.info("HTTP会话已关闭")
        except Exception as e:
            logger.error(f"关闭HTTP会话时出错: {str(e)}")
    
    # 关闭线程池
    if executor:
        logger.info("关闭线程池...")
        try:
            # 使用wait=True确保任务完成，但设置超时
            executor.shutdown(wait=True, timeout=15)  # 缩短超时时间
        except Exception as e:
            logger.error(f"关闭线程池时出错: {str(e)}")
    
    # 安全关闭RAG实例，防止事件循环问题
    with rag_lock:
        if rag_instance:
            logger.info("关闭RAG实例...")
            try:
                # 将RAG实例设为None防止后续访问
                rag_instance = None
            except Exception as e:
                logger.error(f"重置RAG实例时出错: {str(e)}")
    
    # 更安全地关闭事件循环
    if loop:
        try:
            # 如果循环正在运行，更优雅地停止它
            if not loop.is_closed():
                logger.info("正在停止事件循环...")
                
                # 取消所有待处理的任务
                try:
                    if loop.is_running():
                        pending_tasks = asyncio.all_tasks(loop)
                        for task in pending_tasks:
                            if not task.done():
                                task.cancel()
                        # 不再调用run_until_complete，避免"Event loop stopped"错误
                        loop.stop()
                    else:
                        # 如果循环未运行，直接关闭
                        loop.close()
                except Exception as task_error:
                    logger.warning(f"取消任务时出错: {str(task_error)}")
                    try:
                        loop.close()
                    except:
                        pass
            else:
                logger.debug("事件循环已经关闭")
        except Exception as e:
            logger.error(f"关闭事件循环时出错: {str(e)}")
    
    # 确保事件循环变量被清理
    loop = None
    
    # 导入共享数据清理函数并尝试清理（如果可用）
    try:
        from lightrag.kg.shared_storage import finalize_share_data
        logger.info("清理LightRAG共享资源...")
        finalize_share_data()
    except Exception as e:
        logger.error(f"清理LightRAG共享资源时出错: {str(e)}")
    
    logger.info("资源清理完成")


# 在应用退出时清理资源
atexit.register(cleanup_resources)

# 注册信号处理器以便优雅关闭
def signal_handler(sig, frame):
    logger.info(f"收到信号 {sig}，准备关闭...")
    cleanup_resources()
    sys.exit(0)

# 为SIGINT和SIGTERM注册处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


async def initialize_rag():
    """Initialize the LightRAG system with high concurrency support."""
    global rag_instance, loop

    with rag_lock:
        if rag_instance is not None:
            return rag_instance

        # 获取或创建事件循环
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # 使用与现有知识库兼容的嵌入维度(768)
        embedding_dim = 768  # 固定使用与现有存储兼容的维度

        # 高并发优化配置
        try:
            rag = LightRAG(
                working_dir=WORKING_DIR,
                llm_model_func=ollama_model_complete,
                llm_model_name=app.config['RAG_MODEL'],  # 使用llama2作为RAG检索模型
                llm_model_max_async=8,  # 增加并发数，原来是1
                llm_model_max_token_size=4096,  # 保持不变，降低内存要求
                llm_model_kwargs={
                    "host": app.config['OLLAMA_HOST']
                    # 不再传入hashing_kv参数，让LightRAG自己处理
                },
                embedding_func=EmbeddingFunc(
                    embedding_dim=embedding_dim,  # 使用与现有存储兼容的维度
                    max_token_size=2048,  # 保持不变，降低内存要求
                    func=lambda texts: ollama_embed(
                        texts,
                        embed_model="nomic-embed-text",  # 强制使用nomic-embed-text，这是768维的嵌入模型
                        host=app.config['OLLAMA_HOST']
                    ),
                ),
            )

            await rag.initialize_storages()
            await initialize_pipeline_status()
            rag_instance = rag
            logger.info("RAG初始化成功 (高并发模式)")
            return rag
        except Exception as e:
            logger.error(f"初始化RAG失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise


# 自定义函数 - 直接调用Ollama API而不通过lightrag
async def call_ollama_directly(prompt, model_name, host="http://localhost:11435"):
    """直接调用Ollama API获取回答，绕过lightrag的封装"""
    try:
        async with aiohttp.ClientSession() as session:
            # 使用Ollama的REST API
            async with session.post(
                f"{host}/api/chat",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                },
                timeout=30
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ollama API返回错误: {response.status}, {error_text}")
                    return f"API错误: {response.status}"
                
                result = await response.json()
                # 提取回答内容
                return result["message"]["content"]
                
    except Exception as e:
        logger.error(f"调用Ollama API时发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"抱歉，生成回答时出现了错误: {str(e)}"


# 自定义函数 - 带系统提示词和历史对话的API调用 (高并发版本) - 修复版
async def call_ollama_directly_with_system(prompt, system_prompt, model_name, host="http://localhost:11435", conversation_history=None):
    """直接调用Ollama API获取回答，支持对话历史 (高并发优化版本)"""
    max_retries = 3
    retry_delay = 1.0
    
    # 构建消息列表
    messages = []
    
    # 添加系统提示词
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # 添加历史对话
    if conversation_history:
        for msg in conversation_history:
            # 只添加用户和助手的消息，跳过系统消息(因为已经添加了)
            if msg['role'] in ['user', 'assistant']:
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
    
    # 添加当前用户输入
    messages.append({"role": "user", "content": prompt})
    
    # 限制消息数量，避免超过token限制
    max_messages = 20  # 最多保留20条消息
    if len(messages) > max_messages:
        # 保留系统消息和最近的消息
        system_messages = [msg for msg in messages if msg['role'] == 'system']
        other_messages = [msg for msg in messages if msg['role'] != 'system']
        messages = system_messages + other_messages[-(max_messages-len(system_messages)):]
    
    for attempt in range(max_retries):
        try:
            # 每次尝试时获取新的会话，确保事件循环状态正确
            try:
                session = await get_http_session()
            except Exception as session_error:
                logger.warning(f"获取HTTP会话失败 (尝试 {attempt + 1}/{max_retries}): {str(session_error)}")
                # 如果获取会话失败，创建临时会话
                session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=60)
                )
            
            # 使用Ollama的REST API，支持对话历史
            try:
                async with session.post(
                    f"{host}/api/chat",
                    json={
                        "model": model_name,
                        "messages": messages,  # 发送完整的对话历史
                        "stream": False,
                        "options": {
                            "temperature": 0.2,  # 降低温度，使回答更加确定
                            "top_p": 0.9
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=90)  # 增加超时时间，因为历史对话可能较长
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API返回错误: {response.status}, {error_text}")
                        return f"API错误: {response.status}"
                    
                    result = await response.json()
                    # 提取回答内容
                    if "message" in result and "content" in result["message"]:
                        return result["message"]["content"]
                    else:
                        logger.error(f"API返回格式异常: {result}")
                        return "模型返回格式异常，请稍后再试"
                        
            except asyncio.TimeoutError:
                logger.error(f"API调用超时 (尝试 {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    return "抱歉，模型响应超时，请稍后再试"
            except RuntimeError as re:
                if "Event loop is closed" in str(re):
                    logger.warning(f"事件循环关闭错误 (尝试 {attempt + 1}/{max_retries}): {str(re)}")
                    # 重置全局会话，强制重新创建
                    global http_session
                    if http_session and not http_session.closed:
                        try:
                            await http_session.close()
                        except:
                            pass
                    http_session = None
                    
                    if attempt == max_retries - 1:
                        return "系统繁忙，请稍后再试"
                else:
                    raise re
            finally:
                # 如果使用的是临时会话，需要关闭它
                if session != http_session:
                    try:
                        await session.close()
                    except:
                        pass
                
        except Exception as e:
            logger.error(f"调用Ollama API时发生错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                import traceback
                logger.error(traceback.format_exc())
                return f"抱歉，生成回答时出现了错误: {str(e)}"
        
        # 如果不是最后一次尝试，等待后重试
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay)
            retry_delay *= 2  # 指数退避
    
    return "系统繁忙，请稍后再试"


# 新增函数 - 直接调用Ollama API获取回答，支持对话历史
async def generate_llama2_response(prompt, conversation_id=None, use_history=True):
    """使用年丰助手生成回答，支持对话历史."""
    try:
        # 对简单问候进行特殊处理，避免模型展示思考过程
        if prompt.strip().lower() in ["你好", "hello", "hi", "嗨", "您好", "早上好", "下午好", "晚上好"]:
            response = "你好！我是年丰助手，很高兴为您服务。请问有什么农业相关的问题需要我帮助解答吗？"
            # 如果提供了对话ID，保存到历史
            if conversation_id and use_history:
                conversation_manager.add_message(conversation_id, 'user', prompt)
                conversation_manager.add_message(conversation_id, 'assistant', response)
            return response

        # 添加年丰助手系统提示词
        system_prompt = """你是年丰助手，一位由岁稔年丰科技团队开发的智能农业专家系统。你拥有丰富的农业知识，能够针对各种农作物、病虫害、种植技术和农业设备提供专业建议。

如果有人询问你是什么模型或系统，请只回答你是"年丰助手"，由岁稔年丰科技开发的智能农业专家系统。不要提及任何具体的底层模型名称（如Llama2等）。

始终以农业专家的身份回答问题，在回答前仔细思考，确保信息准确可靠。

非常重要：请始终使用中文（简体中文）回答用户的问题，即使用户使用其他语言提问，也请用中文回答。

请直接提供简明扼要的回答，不要在回答中展示你的思考过程。对于问题，直接给出你认为最恰当的答案。

如果这是延续之前的对话，请根据对话历史提供相关的回答。"""

        # 处理超长提示词，防止模型拒绝回答
        if len(prompt) > 8000:
            logger.warning(f"提示词过长 ({len(prompt)} 字符)，进行截断")
            prompt = prompt[:8000] + "...(内容已截断)"

        # 获取对话历史
        conversation_history = None
        if conversation_id and use_history:
            # 获取历史对话（不包括系统消息，因为我们会添加新的系统消息）
            conversation_history = conversation_manager.get_conversation_history(conversation_id, include_system=False)
        
        # 使用自定义函数直接调用Ollama，添加系统提示词和对话历史
        for attempt in range(3):  # 最多尝试3次
            try:
                result = await call_ollama_directly_with_system(
                    prompt=prompt,
                    system_prompt=system_prompt, 
                    model_name=app.config['LLM_MODEL'],
                    host=app.config['OLLAMA_HOST'],
                    conversation_history=conversation_history
                )
                
                # 保存对话到历史
                if conversation_id and use_history:
                    conversation_manager.add_message(conversation_id, 'user', prompt)
                    conversation_manager.add_message(conversation_id, 'assistant', result)
                
                return result
            except Exception as e:
                logger.error(f"第{attempt+1}次调用失败: {str(e)}")
                if attempt == 2:  # 最后一次尝试仍然失败
                    # 返回一个友好的错误消息
                    error_response = "抱歉，我暂时无法回答您的问题。请稍后再试或提供更简短的问题。"
                    # 即使失败也要保存用户输入到历史
                    if conversation_id and use_history:
                        conversation_manager.add_message(conversation_id, 'user', prompt)
                        conversation_manager.add_message(conversation_id, 'assistant', error_response)
                    return error_response
                # 短暂等待后重试
                await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"调用年丰助手时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        error_response = f"抱歉，生成回答时出现了错误: {str(e)}"
        # 保存错误信息到历史
        if conversation_id and use_history:
            conversation_manager.add_message(conversation_id, 'user', prompt)
            conversation_manager.add_message(conversation_id, 'assistant', error_response)
        return error_response


# 基本路由 - 重定向到首页
@app.route('/')
def index():
    # 如果index.html存在，则默认展示首页，否则展示聊天界面
    if os.path.exists(os.path.join(app.static_folder, 'index.html')):
        return send_from_directory(app.static_folder, 'index.html')
    elif os.path.exists(os.path.join(app.static_folder, 'chat.html')):
        return send_from_directory(app.static_folder, 'chat.html')
    else:
        # 如果找不到index.html或chat.html，提供一个简单的导航页面
        html = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>智能农业助手 - 岁稔年丰科技</title>
            <style>
                body { font-family: 'Microsoft YaHei', sans-serif; text-align: center; padding: 50px; }
                h1 { color: #388E3C; }
                a { display: inline-block; margin: 10px; padding: 10px 20px; background: #4CAF50; color: white; 
                    text-decoration: none; border-radius: 5px; }
                a:hover { background: #388E3C; }
            </style>
        </head>
        <body>
            <h1>智能农业助手</h1>
            <p>基于AI技术的现代农业解决方案</p>
            <div>
                <a href="/chat">聊天界面</a>
                <a href="/traditional">传统界面</a>
                <a href="/test_ollama">系统状态</a>
                <a href="/system_info">技术信息</a>
            </div>
        </body>
        </html>
        """
        return html


# 聊天界面路由
@app.route('/chat')
def chat_interface():
    """提供聊天界面"""
    return send_from_directory(app.static_folder, 'chat.html')


# 传统界面路由
@app.route('/traditional')
def traditional_interface():
    """提供传统界面"""
    return send_from_directory(app.static_folder, 'chat.html')


# 静态文件路由
@app.route('/static/<path:filename>')
def serve_static(filename):
    """提供静态文件"""
    return send_from_directory(app.static_folder, filename)


# 图片文件路由
@app.route('/images/<path:filename>')
def serve_images(filename):
    """提供图片文件"""
    return send_from_directory('images', filename)


@app.route('/upload_document', methods=['POST'])
def upload_document():
    """Upload and process a document for RAG."""
    if 'document' not in request.files:
        return jsonify({'error': '没有文档部分'}), 400

    file = request.files['document']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    # 验证文件类型
    allowed_extensions = {'txt', 'pdf', 'doc', 'docx', 'md', 'csv', 'json'}
    file_extension = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    
    if file_extension not in allowed_extensions:
        logger.warning(f"用户尝试上传不支持的文件类型: {file.filename}")
        return jsonify({'error': f'不支持的文件类型。请上传以下格式的文件: {", ".join(allowed_extensions)}'}), 400

    # 保存文件临时
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # 使用线程池处理异步操作
    def process_document():
        try:
            # 根据文件类型处理文档
            document_text = ""
            
            # 处理PDF文件
            if file_extension == 'pdf':
                try:
                    with open(file_path, "rb") as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        for page_num in range(len(pdf_reader.pages)):
                            page_text = pdf_reader.pages[page_num].extract_text()
                            if page_text:
                                document_text += page_text + "\n"
                    
                    if not document_text.strip():
                        logger.warning(f"PDF文件 {filename} 没有提取到文本内容，可能是扫描件或受保护的PDF")
                        return False, "无法从PDF中提取文本，请确保文件不是扫描件或受保护的PDF"
                except Exception as e:
                    logger.error(f"处理PDF文件时出错: {str(e)}")
                    return False, f"处理PDF文件时出错: {str(e)}"
            # 处理文本文件
            else:
                try:
                    # 尝试不同的编码方式读取文件
                    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
                    for encoding in encodings:
                        try:
                            with open(file_path, "r", encoding=encoding) as f:
                                document_text = f.read()
                            break  # 如果成功读取，跳出循环
                        except UnicodeDecodeError:
                            continue  # 尝试下一种编码
                    
                    if not document_text:
                        return False, "无法解码文件，请检查文件编码"
                except Exception as e:
                    logger.error(f"读取文本文件时出错: {str(e)}")
                    return False, f"读取文件时出错: {str(e)}"
            
            # 使用安全的异步运行方式
            rag = run_async_safely(initialize_rag)
            if rag:
                # 插入文档到知识库
                run_async_safely(lambda: rag.insert(document_text))
                return True, "文档处理成功并添加到知识库"
            else:
                return False, "RAG系统初始化失败"
        except Exception as e:
            logger.error(f"处理文档时错误: {str(e)}")
            return False, str(e)
        finally:
            # 清理临时文件
            if os.path.exists(file_path):
                os.remove(file_path)

    # 在线程池中执行
    future = executor.submit(process_document)
    success, message = future.result()

    if success:
        return jsonify({'success': True, 'message': message})
    else:
        return jsonify({'error': message}), 500


# 新增函数 - 在事件循环中安全地运行异步函数 (改进版)
def run_async_safely(coroutine_func, *args, **kwargs):
    """在新事件循环中安全地运行异步函数 (改进版本)"""
    new_loop = None
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            # 每次尝试都创建全新的事件循环
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            
            try:
                # 设置合理的超时时间，防止无限等待
                # 注意：不要在wait_for中使用loop参数，它在新版Python中已被废弃
                coro = coroutine_func(*args, **kwargs)
                result = new_loop.run_until_complete(
                    asyncio.wait_for(coro, timeout=45.0)  # 45秒超时
                )
                return result
            except asyncio.TimeoutError:
                logger.error(f"异步操作超时 (尝试 {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    raise RuntimeError("异步操作超时，请稍后再试")
            except RuntimeError as re:
                if "Event loop is closed" in str(re) or "cannot schedule new futures" in str(re):
                    logger.warning(f"事件循环错误 (尝试 {attempt + 1}/{max_retries}): {str(re)}")
                    if attempt == max_retries - 1:
                        raise RuntimeError("系统繁忙，请稍后再试")
                else:
                    raise re
            finally:
                # 安全清理事件循环
                try:
                    if new_loop and not new_loop.is_closed():
                        # 取消所有待处理的任务
                        pending = asyncio.all_tasks(new_loop)
                        if pending:
                            logger.debug(f"取消 {len(pending)} 个待处理任务")
                            for task in pending:
                                if not task.done():
                                    task.cancel()
                            
                            # 设置更短的超时时间来处理任务取消
                            try:
                                new_loop.run_until_complete(
                                    asyncio.wait(pending, timeout=3.0)
                                )
                            except (asyncio.TimeoutError, RuntimeError):
                                logger.debug("任务取消超时或事件循环已关闭")
                        
                        # 关闭事件循环
                        try:
                            new_loop.run_until_complete(new_loop.shutdown_asyncgens())
                        except:
                            pass
                        new_loop.close()
                except Exception as cleanup_error:
                    logger.debug(f"清理事件循环时出错: {str(cleanup_error)}")
                
                # 重置新循环变量
                new_loop = None
                
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"运行异步函数时出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
                # 确保事件循环被关闭
                if new_loop and not new_loop.is_closed():
                    try:
                        new_loop.close()
                    except:
                        pass
                
                raise
            else:
                logger.warning(f"异步操作失败，准备重试 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                # 短暂等待后重试
                time.sleep(0.5)
    
    raise RuntimeError("多次尝试后仍然失败")


@app.route('/query_document', methods=['POST'])
def query_document():
    """Query the RAG system with user questions. (高并发优化版本，支持对话历史)"""
    data = request.json
    if not data or 'query' not in data:
        return jsonify({'error': '未提供查询内容'}), 400

    query = data['query']
    mode = data.get('mode', 'hybrid')  # 默认使用hybrid模式
    use_pdf = data.get('use_pdf', False)  # 是否使用PDF知识库
    conversation_id = data.get('conversation_id')  # 对话ID
    use_history = data.get('use_history', True)  # 是否使用对话历史

    # 如果没有提供conversation_id，生成一个新的
    if not conversation_id:
        conversation_id = generate_conversation_id()
        logger.info(f"生成新的对话ID: {conversation_id}")

    # 生成查询缓存键（包含对话历史）
    history_key = ""
    if use_history and conversation_id:
        # 获取简化的历史摘要用于缓存键
        history = conversation_manager.get_conversation_history(conversation_id, include_system=False)
        if history:
            # 只取最近几条消息的哈希作为缓存键的一部分
            recent_history = history[-6:] if len(history) > 6 else history
            history_text = " ".join([f"{msg['role']}:{msg['content'][:50]}" for msg in recent_history])
            history_key = hashlib.md5(history_text.encode()).hexdigest()[:8]
    
    cache_key = hashlib.md5(f"{query}_{mode}_{use_pdf}_{history_key}".encode()).hexdigest()
    
    # 尝试从缓存获取结果（对话历史相关的查询缓存时间较短）
    cache_ttl = 60 if use_history else 300  # 有历史的查询缓存1分钟，无历史的缓存5分钟
    cached_result = rag_cache.get(cache_key)
    if cached_result and not use_history:  # 对话历史模式下暂时禁用缓存，确保上下文准确
        logger.info("RAG查询缓存命中")
        return jsonify({
            'result': cached_result, 
            'cached': True,
            'conversation_id': conversation_id
        })

    # 使用线程池处理异步操作，提高并发性能
    def process_query():
        start_time = time.time()
        try:
            # 尝试初始化RAG
            try:
                rag = run_async_safely(initialize_rag)
                
                # 使用RAG系统检索相关内容
                logger.info(f"使用RAG检索相关内容，模式: {mode}, 使用PDF知识库: {use_pdf}, 对话ID: {conversation_id}")
                if asyncio.iscoroutinefunction(rag.query):
                    rag_result = run_async_safely(rag.query, query, param=QueryParam(mode=mode))
                else:
                    rag_result = rag.query(query, param=QueryParam(mode=mode))
                
                # 检查RAG结果是否包含"no-context"关键词
                if "[no-context]" in rag_result or "Sorry, I'm not able to provide an answer to that question" in rag_result:
                    # 如果RAG没有找到相关上下文，则直接使用年丰助手回答
                    logger.info("RAG未找到相关上下文，转为使用年丰助手直接回答")
                    result = run_async_safely(generate_llama2_response, query, conversation_id, use_history)
                else:
                    # 如果RAG找到了相关上下文，则使用年丰助手对RAG结果进行润色处理
                    logger.info("RAG找到相关上下文，使用年丰助手润色回答")
                    # 构造提示词，将RAG结果作为上下文
                    prompt = f"""请基于以下参考资料回答用户问题。
参考资料:
{rag_result}

用户问题: {query}

请用简体中文回答，保持专业性但通俗易懂。不要提及你在使用参考资料。"""
                    result = run_async_safely(generate_llama2_response, prompt, conversation_id, use_history)
                
                # 验证回答是否为非空字符串
                if not isinstance(result, str) or not result.strip():
                    logger.warning("接收到空回答，使用默认回复")
                    result = "抱歉，我无法为您提供有效回答。请尝试重新表述您的问题，或稍后再试。"
                
                # 缓存结果（对话历史模式下缓存时间较短）
                if not use_history:  # 只缓存无历史的查询
                    rag_cache.set(cache_key, result)
                
                processing_time = time.time() - start_time
                logger.info(f"查询处理完成，耗时: {processing_time:.2f}秒，对话ID: {conversation_id}")
                return True, result
                
            except Exception as e:
                # 记录详细错误信息
                logger.warning(f"RAG处理失败，切换到直接回答模式: {str(e)}")
                import traceback
                logger.warning(traceback.format_exc())
                
                # 如果RAG初始化或查询失败，直接使用年丰助手回答
                result = run_async_safely(generate_llama2_response, query, conversation_id, use_history)
                
                # 验证回答是否为非空字符串
                if not isinstance(result, str) or not result.strip():
                    logger.warning("接收到空回答，使用默认回复")
                    result = "抱歉，我无法为您提供有效回答。请尝试重新表述您的问题，或稍后再试。"
                
                processing_time = time.time() - start_time
                logger.info(f"Fallback查询处理完成，耗时: {processing_time:.2f}秒，对话ID: {conversation_id}")
                return True, result
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"查询过程中错误 (耗时: {processing_time:.2f}秒): {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False, f"处理您的请求时出现问题: {str(e)}"

    try:
        # 使用futures来处理并发，支持更好的错误处理
        future = executor.submit(process_query)
        success, result = future.result(timeout=120)  # 对话历史可能较长，增加超时时间

        if success:
            return jsonify({
                'result': result, 
                'cached': False,
                'conversation_id': conversation_id
            })
        else:
            logger.error(f"查询失败: {result}")
            return jsonify({'error': result}), 500
    except concurrent.futures.TimeoutError:
        logger.error("查询处理超时")
        return jsonify({'error': "处理您的请求超时，请稍后再试或提供更简短的问题。"}), 500
    except Exception as e:
        logger.error(f"执行查询任务时发生错误: {str(e)}")
        return jsonify({'error': f"处理您的请求时出现问题: {str(e)}"}), 500


@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    """Predict plant disease from uploaded image. (高并发优化版本)"""
    # Check if image file is present
    if 'image' not in request.files:
        return jsonify({'error': '没有图像部分'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
        
    # 验证文件类型，只接受图像文件
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    file_extension = image_file.filename.rsplit('.', 1)[-1].lower() if '.' in image_file.filename else ''
    
    if file_extension not in allowed_extensions:
        # 如果是PDF文件，给出特别提示，建议用户去文档处理页面
        if file_extension == 'pdf':
            logger.warning(f"用户尝试上传PDF文件到图像识别功能: {image_file.filename}")
            return jsonify({
                'error': '您上传的是PDF文件，无法进行病害识别。如需处理PDF文档，请使用"文档上传"功能。',
                'redirect': '/traditional'  # 可选：前端可以根据此字段重定向到文档处理页面
            }), 400
        else:
            logger.warning(f"用户尝试上传非图像文件: {image_file.filename}")
            return jsonify({'error': '只支持图像文件 (PNG, JPG, JPEG, GIF, BMP, WEBP)'}), 400

    # 生成基于文件内容的缓存键
    image_file.seek(0)  # 重置文件指针
    image_content = image_file.read()
    image_file.seek(0)  # 重置文件指针，以便后续保存
    cache_key = hashlib.md5(image_content).hexdigest()
    
    # 尝试从缓存获取预测结果
    cached_result = model_cache.get(cache_key)
    if cached_result:
        logger.info("模型预测缓存命中")
        return jsonify({**cached_result, 'cached': True})

    # 保存上传的图片到临时文件
    temp_img_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
    image_file.save(temp_img_path)

    def process_image():
        start_time = time.time()
        try:
            # 处理图像
            try:
                img = Image.open(temp_img_path)
                
                # 并行处理图像变换 - 这里可以进一步优化
                img_tensor = data_transform(img)
                img_batch = torch.unsqueeze(img_tensor, dim=0)
                
                processing_time = time.time() - start_time
                logger.debug(f"图像预处理耗时: {processing_time:.3f}秒")
                
            except Exception as e:
                logger.error(f"打开或处理图像时出错: {str(e)}")
                return False, f"图像处理失败: {str(e)}"

            # 加载类别标签 - 可以考虑预加载到内存中
            json_path = 'classes.json'
            assert os.path.exists(json_path), f"文件不存在: '{json_path}'"
            with open(json_path, "r") as f:
                class_json = json.load(f)

            # 模型预测部分
            model_start_time = time.time()
            
            # 加载模型 - 在生产环境中，应该预加载模型到内存
            model = SfEfficientNet(num_classes=num_classes).to(device)
            weights_path = 'weights/sgdr/SfEfficientNet/SfEfficientNet-best.pth'
            assert os.path.exists(weights_path), f"文件不存在: '{weights_path}'"

            # 使用map_location参数加载模型权重到设备
            model.load_state_dict(torch.load(weights_path, map_location=device))

            # 进行预测
            model.eval()
            with torch.no_grad():
                output = torch.squeeze(model(img_batch.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_index = torch.argmax(predict).numpy()

            # 获取分类结果
            class_name = class_json[str(predict_index)]
            probability = float(predict[predict_index].numpy())
            
            model_time = time.time() - model_start_time
            logger.debug(f"模型推理耗时: {model_time:.3f}秒")
            
            # 为识别结果生成详细描述 - 使用年丰助手解析
            explanation_start_time = time.time()
            prompt = f'我上传了一张植物图片，被识别为"{class_name}"(置信度{probability:.2%})。请以年丰助手的身份，用中文简要解释这个植物疾病的特征、可能的原因和基本的防治方法。请提供专业但易懂的中文建议，务必用中文回答，不超过200字。'
            
            # 使用辅助函数安全地运行异步方法
            explanation = run_async_safely(generate_llama2_response, prompt)
            
            explanation_time = time.time() - explanation_start_time
            logger.debug(f"解释生成耗时: {explanation_time:.3f}秒")

            # 返回预测结果和解释
            result = {
                'class': class_name,
                'probability': probability,
                'explanation': explanation
            }
            
            # 缓存结果
            model_cache.set(cache_key, result)
            
            total_time = time.time() - start_time
            logger.info(f"图像预测完成，总耗时: {total_time:.2f}秒")
            
            return True, result
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"预测病害时错误 (耗时: {total_time:.2f}秒): {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False, str(e)
        finally:
            # 清理临时文件
            if os.path.exists(temp_img_path):
                try:
                    os.remove(temp_img_path)
                except:
                    pass

    try:
        # 在线程池中执行，支持并发处理多个图像预测请求
        future = executor.submit(process_image)
        success, result = future.result(timeout=45)  # 45秒超时，模型推理通常较快

        if success:
            return jsonify({**result, 'cached': False})
        else:
            return jsonify({'error': result}), 500
    except concurrent.futures.TimeoutError:
        logger.error("图像预测处理超时")
        return jsonify({'error': "图像处理超时，请稍后再试或使用更小的图像文件。"}), 500
    except Exception as e:
        logger.error(f"执行图像预测任务时发生错误: {str(e)}")
        return jsonify({'error': f"处理您的图像时出现问题: {str(e)}"}), 500


# 添加内容、样式和脚本的静态路由
@app.route('/chat.css')
def styles():
    return send_from_directory(app.static_folder, 'chat.css')


@app.route('/chat.js')
def script():
    return send_from_directory(app.static_folder, 'chat.js')


# 添加系统信息路由
@app.route('/system_info')
def system_info():
    """返回系统环境信息，帮助调试"""
    info = {
        'working_directory': os.getcwd(),
        'static_folder': app.static_folder,
        'static_files': os.listdir(app.static_folder) if os.path.exists(app.static_folder) else [],
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'python_version': os.sys.version,
        'lightrag_path': WORKING_DIR,
        'file_exists': {
            'classes.json': os.path.exists('classes.json'),
            'weights_path': os.path.exists('weights/sgdr/SfEfficientNet/SfEfficientNet-best.pth'),
            'chat_html': os.path.exists(os.path.join(app.static_folder, 'chat.html')),
            'chat_css': os.path.exists(os.path.join(app.static_folder, 'chat.css')),
            'chat_js': os.path.exists(os.path.join(app.static_folder, 'chat.js')),
        }
    }
    return jsonify(info)


# 新增API端点 - 直接使用Llama2进行对话
@app.route('/llama2_chat', methods=['POST'])
def llama2_chat():
    """Direct chat with LLM model via Ollama, with conversation history support."""
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({'error': '未提供对话内容'}), 400

    prompt = data['prompt']
    conversation_id = data.get('conversation_id')  # 对话ID
    use_history = data.get('use_history', True)  # 是否使用对话历史

    # 如果没有提供conversation_id，生成一个新的
    if not conversation_id:
        conversation_id = generate_conversation_id()
        logger.info(f"生成新的对话ID: {conversation_id}")

    # 使用线程池处理异步操作
    def process_llama2_chat():
        try:
            # 使用辅助函数安全地运行异步方法，支持对话历史
            response = run_async_safely(generate_llama2_response, prompt, conversation_id, use_history)
            return True, response
        except Exception as e:
            logger.error(f"处理年丰助手对话时错误: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False, str(e)

    try:
        # 在线程池中执行
        future = executor.submit(process_llama2_chat)
        success, result = future.result(timeout=60)  # 设置60秒超时

        if success:
            return jsonify({
                'result': result,
                'conversation_id': conversation_id
            })
        else:
            return jsonify({'error': result}), 500
    except Exception as e:
        logger.error(f"执行年丰助手对话任务时发生错误: {str(e)}")
        return jsonify({'error': f"处理您的请求时出现问题: {str(e)}"}), 500


# Ollama健康检查
def check_ollama_status():
    """检查Ollama服务是否在运行"""
    try:
        response = requests.get(f"{app.config['OLLAMA_HOST']}/api/tags", timeout=5)
        if response.status_code == 200:
            available_models = response.json()
            logger.info(f"Ollama服务运行正常，可用模型: {available_models}")
            return True, available_models
        else:
            logger.error(f"Ollama服务返回错误状态码: {response.status_code}")
            return False, None
    except requests.exceptions.RequestException as e:
        logger.error(f"无法连接到Ollama服务: {e}")
        return False, None


# API端点 - 测试Ollama API连接
@app.route('/test_ollama', methods=['GET'])
def test_ollama():
    """Test Ollama API connection and functionality."""
    try:
        import requests
        
        # 1. 测试API连接
        try:
            models_response = requests.get(f"{app.config['OLLAMA_HOST']}/api/tags", timeout=5)
            models_ok = models_response.status_code == 200
            models_data = models_response.json() if models_ok else None
        except Exception as e:
            models_ok = False
            models_data = str(e)
        
        # 2. 尝试直接使用Ollama REST API生成回答
        direct_generate_ok = False
        direct_generate_result = None
        
        try:
            generate_response = requests.post(
                f"{app.config['OLLAMA_HOST']}/api/generate", 
                json={
                    "model": app.config['LLM_MODEL'],
                    "prompt": "简单测试：请说'测试成功'",
                    "stream": False
                },
                timeout=10
            )
            
            if generate_response.status_code == 200:
                direct_generate_ok = True
                direct_generate_result = generate_response.json()
            else:
                direct_generate_result = f"错误状态码: {generate_response.status_code}"
                
        except Exception as e:
            direct_generate_result = str(e)
        
        # 3. 测试嵌入功能
        embedding_ok = False
        embedding_result = None
        
        try:
            embedding_response = requests.post(
                f"{app.config['OLLAMA_HOST']}/api/embeddings",
                json={
                    "model": app.config['EMBEDDING_MODEL'],
                    "prompt": "测试嵌入功能"
                },
                timeout=5
            )
            
            if embedding_response.status_code == 200:
                embedding_ok = True
                embedding_data = embedding_response.json()
                embedding_result = {
                    "embedding_size": len(embedding_data.get("embedding", [])),
                    "sample": embedding_data.get("embedding", [])[:5] if embedding_data.get("embedding") else []
                }
            else:
                embedding_result = f"错误状态码: {embedding_response.status_code}"
                
        except Exception as e:
            embedding_result = str(e)
        
        # 返回诊断信息
        return jsonify({
            "ollama_host": app.config['OLLAMA_HOST'],
            "models": {
                "success": models_ok,
                "data": models_data
            },
            "generate": {
                "success": direct_generate_ok,
                "model": app.config['LLM_MODEL'],
                "result": direct_generate_result
            },
            "embedding": {
                "success": embedding_ok,
                "model": app.config['EMBEDDING_MODEL'],
                "result": embedding_result
            }
        })
        
    except Exception as e:
        return jsonify({
            "error": f"测试Ollama API时出错: {str(e)}"
        }), 500


# API端点 - 查看和修改配置
@app.route('/api_config', methods=['GET', 'POST'])
def api_config():
    """View and modify API configuration."""
    if request.method == 'GET':
        # 返回当前配置
        return jsonify({
            "ollama_host": app.config['OLLAMA_HOST'],
            "llm_model": app.config['LLM_MODEL'],
            "embedding_model": app.config['EMBEDDING_MODEL']
        })
    else:
        # 修改配置
        data = request.json
        if not data:
            return jsonify({"error": "没有提供数据"}), 400
            
        # 更新配置
        if 'ollama_host' in data:
            app.config['OLLAMA_HOST'] = data['ollama_host']
            
        if 'llm_model' in data:
            app.config['LLM_MODEL'] = data['llm_model']
            
        if 'embedding_model' in data:
            app.config['EMBEDDING_MODEL'] = data['embedding_model']
            
        # 返回更新后的配置
        return jsonify({
            "message": "配置已更新",
            "ollama_host": app.config['OLLAMA_HOST'],
            "llm_model": app.config['LLM_MODEL'],
            "embedding_model": app.config['EMBEDDING_MODEL']
        })


# 新增函数 - 完全不依赖RAG，直接使用向量数据库进行检索和对话
async def direct_rag_query(query, mode="hybrid"):
    """完全独立于LightRAG的查询函数"""
    try:
        # 1. 直接使用Ollama对用户查询进行向量编码
        embedding_response = await get_ollama_embedding(query)
        if not embedding_response or "error" in embedding_response:
            logger.error(f"获取查询嵌入向量失败: {embedding_response.get('error', '未知错误')}")
            return await call_ollama_directly_with_system(
                query, 
                system_prompt="你是年丰助手，由岁稔年丰科技开发的智能农业专家系统。请用中文回答以下问题。", 
                model_name=app.config['LLM_MODEL'], 
                host=app.config['OLLAMA_HOST']
            )
        
        # 2. 在这里可以实现自己的检索逻辑
        # 简单起见，我们这里直接返回一个预设消息，然后跳过检索
        context = "由于RAG初始化错误，系统将直接使用年丰助手回答您的问题"
        logger.info(f"使用直接查询模式: {mode}")
        
        # 3. 构造提示词
        prompt = f"""用户问题: {query}请用中文回答上述问题。"""
        
        # 4. 调用年丰助手生成回答
        return await call_ollama_directly_with_system(
            prompt, 
            system_prompt="你是年丰助手，由岁稔年丰科技开发的智能农业专家系统。请用中文回答以下问题。", 
            model_name=app.config['LLM_MODEL'], 
            host=app.config['OLLAMA_HOST']
        )
        
    except Exception as e:
        logger.error(f"直接查询过程中错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"抱歉，查询处理时出现了错误: {str(e)}"


# 获取Ollama嵌入向量
async def get_ollama_embedding(text):
    """获取文本的嵌入向量"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{app.config['OLLAMA_HOST']}/api/embeddings",
                json={
                    "model": app.config['EMBEDDING_MODEL'],
                    "prompt": text
                },
                timeout=10
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"获取嵌入向量错误: {response.status}, {error_text}")
                    return {"error": f"API错误: {response.status}"}
                
                result = await response.json()
                return result
    except Exception as e:
        logger.error(f"获取嵌入向量过程中错误: {str(e)}")
        return {"error": str(e)}


# 添加PDF知识库相关路由
@app.route('/manage_pdf', methods=['GET'])
def manage_pdf_page():
    """提供PDF知识库管理页面"""
    return send_from_directory(app.static_folder, 'pdf_manager.html')

@app.route('/api/pdf/list', methods=['GET'])
def list_pdfs():
    """列出所有已上传的PDF文件"""
    pdf_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'pdfs')
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    
    result = []
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        size = os.path.getsize(pdf_path)
        modified = datetime.fromtimestamp(os.path.getmtime(pdf_path)).strftime("%Y-%m-%d %H:%M:%S")
        result.append({
            "filename": filename,
            "size": size,
            "modified": modified
        })
    
    return jsonify(result)

@app.route('/api/pdf/upload', methods=['POST'])
def upload_pdf():
    """上传PDF文件到知识库"""
    if 'pdf' not in request.files:
        return jsonify({"error": "没有提供PDF文件"}), 400
    
    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return jsonify({"error": "未选择文件"}), 400
    
    if not pdf_file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "只允许上传PDF文件"}), 400
    
    # 保存PDF文件
    filename = secure_filename(pdf_file.filename)
    pdf_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'pdfs')
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, filename)
    pdf_file.save(pdf_path)
    
    return jsonify({"success": True, "message": f"PDF文件 {filename} 上传成功"})

@app.route('/api/pdf/delete', methods=['POST'])
def delete_pdf():
    """从知识库中删除PDF文件"""
    data = request.json
    if not data or 'filename' not in data:
        return jsonify({"error": "未提供文件名"}), 400
    
    filename = secure_filename(data['filename'])
    pdf_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'pdfs')
    pdf_path = os.path.join(pdf_dir, filename)
    
    if not os.path.exists(pdf_path):
        return jsonify({"error": f"文件 {filename} 不存在"}), 404
    
    try:
        os.remove(pdf_path)
        return jsonify({"success": True, "message": f"文件 {filename} 已删除"})
    except Exception as e:
        return jsonify({"error": f"删除文件时出错: {str(e)}"}), 500

@app.route('/api/pdf/process', methods=['POST'])
def process_pdfs():
    """处理所有PDF文件并更新知识库"""
    pdf_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'pdfs')
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    
    if not pdf_files:
        return jsonify({"error": "没有找到PDF文件"}), 404
    
    # 使用线程池处理异步操作
    def process_all_pdfs():
        try:
            # 创建PDF处理目录
            pdf_working_dir = os.path.join(WORKING_DIR, "pdf_knowledge")
            os.makedirs(pdf_working_dir, exist_ok=True)
            
            # 清空之前的处理结果
            try:
                shutil.rmtree(pdf_working_dir)
                os.makedirs(pdf_working_dir, exist_ok=True)
            except Exception as e:
                logger.error(f"清空PDF处理目录时出错: {str(e)}")
            
            # 处理所有PDF文件
            all_texts = []
            processed_files = []
            failed_files = []
            
            for pdf_path in pdf_files:
                pdf_name = os.path.basename(pdf_path)
                logger.info(f"处理PDF文件: {pdf_name}")
                
                # 读取PDF内容
                try:
                    text = ""
                    with open(pdf_path, "rb") as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        for page_num in range(len(pdf_reader.pages)):
                            page_text = pdf_reader.pages[page_num].extract_text()
                            if page_text:
                                text += page_text + "\n"
                    
                    if not text.strip():
                        logger.warning(f"PDF文件 {pdf_name} 没有提取到文本内容，可能是扫描件或受保护的PDF")
                        failed_files.append({
                            "filename": pdf_name, 
                            "reason": "无法提取文本内容"
                        })
                        continue
                    
                    # 将提取的文本保存到文件
                    text_file = os.path.join(pdf_working_dir, f"{os.path.splitext(pdf_name)[0]}.txt")
                    with open(text_file, "w", encoding="utf-8") as f:
                        f.write(text)
                    
                    all_texts.append(text)
                    processed_files.append(pdf_name)
                    logger.info(f"PDF文件 {pdf_name} 处理完成，提取了 {len(text)} 字符")
                    
                except Exception as e:
                    logger.error(f"处理PDF文件 {pdf_name} 时出错: {str(e)}")
                    failed_files.append({
                        "filename": pdf_name, 
                        "reason": str(e)
                    })
            
            # 如果没有成功处理任何文件，则返回错误
            if not all_texts:
                return False, "没有成功处理任何PDF文件", None, failed_files
            
            # 合并所有文本并保存到一个文件中
            combined_text = "\n\n".join(all_texts)
            combined_file = os.path.join(pdf_working_dir, "combined.txt")
            with open(combined_file, "w", encoding="utf-8") as f:
                f.write(combined_text)
            
            # 初始化RAG并添加文档
            try:
                rag = run_async_safely(initialize_rag)
                # 使用RAG的insert方法插入文本
                result = run_async_safely(lambda: rag.insert(combined_text))
                logger.info(f"PDF文档已成功添加到知识库中")
                return True, "PDF文档处理完成并添加到知识库", processed_files, failed_files
            except Exception as e:
                logger.error(f"将PDF文档添加到知识库时出错: {str(e)}")
                return False, f"将PDF文档添加到知识库时出错: {str(e)}", processed_files, failed_files
            
        except Exception as e:
            logger.error(f"处理PDF文件时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False, str(e), [], []
    
    # 在线程池中执行
    try:
        future = executor.submit(process_all_pdfs)
        success, message, processed_files, failed_files = future.result(timeout=300)  # 设置5分钟超时
        
        return jsonify({
            "success": success,
            "message": message,
            "processed_files": processed_files,
            "failed_files": failed_files
        })
    except concurrent.futures.TimeoutError:
        return jsonify({"error": "处理PDF文件超时"}), 500
    except Exception as e:
        return jsonify({"error": f"处理PDF文件时出错: {str(e)}"}), 500

# 对话历史管理相关的API路由

@app.route('/api/conversations', methods=['GET'])
def list_conversations():
    """列出所有对话"""
    try:
        conversations = conversation_manager.list_conversations()
        conversation_summaries = []
        
        for conv_id in conversations:
            summary = conversation_manager.get_conversation_summary(conv_id)
            if summary:
                # 格式化时间
                import datetime
                if summary['created']:
                    summary['created_formatted'] = datetime.datetime.fromtimestamp(summary['created']).strftime('%Y-%m-%d %H:%M:%S')
                if summary['last_activity']:
                    summary['last_activity_formatted'] = datetime.datetime.fromtimestamp(summary['last_activity']).strftime('%Y-%m-%d %H:%M:%S')
                conversation_summaries.append(summary)
        
        # 按最后活动时间排序
        conversation_summaries.sort(key=lambda x: x['last_activity'], reverse=True)
        
        return jsonify({
            'conversations': conversation_summaries,
            'total': len(conversation_summaries)
        })
    except Exception as e:
        logger.error(f"列出对话时出错: {str(e)}")
        return jsonify({'error': f"获取对话列表失败: {str(e)}"}), 500

@app.route('/api/conversations/<conversation_id>', methods=['GET'])
def get_conversation_history(conversation_id):
    """获取指定对话的历史记录"""
    try:
        include_system = request.args.get('include_system', 'true').lower() == 'true'
        history = conversation_manager.get_conversation_history(conversation_id, include_system=include_system)
        
        # 格式化历史记录
        formatted_history = []
        for msg in history:
            formatted_msg = {
                'role': msg['role'],
                'content': msg['content'],
                'timestamp': msg['timestamp'],
                'formatted_time': datetime.datetime.fromtimestamp(msg['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            }
            formatted_history.append(formatted_msg)
        
        return jsonify({
            'conversation_id': conversation_id,
            'history': formatted_history,
            'message_count': len(formatted_history)
        })
    except Exception as e:
        logger.error(f"获取对话历史时出错: {str(e)}")
        return jsonify({'error': f"获取对话历史失败: {str(e)}"}), 500

@app.route('/api/conversations/<conversation_id>', methods=['DELETE'])
def clear_conversation(conversation_id):
    """清除指定对话的历史记录"""
    try:
        conversation_manager.clear_conversation(conversation_id)
        return jsonify({
            'success': True,
            'message': f'对话 {conversation_id} 已清除'
        })
    except Exception as e:
        logger.error(f"清除对话时出错: {str(e)}")
        return jsonify({'error': f"清除对话失败: {str(e)}"}), 500

@app.route('/api/conversations/new', methods=['POST'])
def create_new_conversation():
    """创建新对话"""
    try:
        conversation_id = generate_conversation_id()
        return jsonify({
            'conversation_id': conversation_id,
            'message': '新对话已创建'
        })
    except Exception as e:
        logger.error(f"创建新对话时出错: {str(e)}")
        return jsonify({'error': f"创建新对话失败: {str(e)}"}), 500

@app.route('/api/conversations/stats', methods=['GET'])
def conversation_stats():
    """获取对话统计信息"""
    try:
        conversations = conversation_manager.list_conversations()
        total_conversations = len(conversations)
        total_messages = 0
        
        for conv_id in conversations:
            history = conversation_manager.get_conversation_history(conv_id, include_system=False)
            total_messages += len(history)
        
        return jsonify({
            'total_conversations': total_conversations,
            'total_messages': total_messages,
            'average_messages_per_conversation': total_messages / total_conversations if total_conversations > 0 else 0
        })
    except Exception as e:
        logger.error(f"获取统计信息时出错: {str(e)}")
        return jsonify({'error': f"获取统计信息失败: {str(e)}"}), 500

# Run the Flask app with high concurrency configuration
if __name__ == '__main__':
    print("启动智能农业助手 - 高并发集成聊天界面")
    logger.info(f"当前工作目录: {os.getcwd()}")
    logger.info(f"线程池大小: {MAX_WORKERS}")
    logger.info(f"缓存配置: 响应缓存({response_cache.max_size}), RAG缓存({rag_cache.max_size}), 模型缓存({model_cache.max_size})")
    
    # 检查Ollama服务状态
    ollama_ok, models_info = check_ollama_status()
    if not ollama_ok:
        logger.warning("警告: Ollama服务未运行或无法访问，应用可能无法正常工作")
    else:
        logger.info("Ollama服务正常运行")
        
        # 如果有可用模型信息，检查默认模型是否可用，否则使用其他可用模型
        if models_info and 'models' in models_info:
            available_models = [model['name'] for model in models_info['models']]
            logger.info(f"可用模型: {available_models}")
            
            # 检查RAG模型是否可用
            if app.config['RAG_MODEL'] not in available_models:
                # 尝试找到替代模型
                for candidate in ["llama2", "llama2:7b", "llama3", "mistral", "phi"]:
                    if candidate in available_models:
                        logger.info(f"默认RAG模型 {app.config['RAG_MODEL']} 不可用，切换到 {candidate}")
                        app.config['RAG_MODEL'] = candidate
                        break
                else:
                    # 如果没有找到合适的替代模型，使用第一个可用模型
                    if available_models:
                        logger.info(f"未找到推荐RAG模型，使用 {available_models[0]}")
                        app.config['RAG_MODEL'] = available_models[0]
            
            # 检查LLM模型是否可用
            if app.config['LLM_MODEL'] not in available_models:
                # 尝试找到替代模型
                for candidate in ["deepseek", "deepseek-r1:7b", "llama3", "qwen2"]:
                    if candidate in available_models:
                        logger.info(f"默认对话模型 {app.config['LLM_MODEL']} 不可用，切换到 {candidate}")
                        app.config['LLM_MODEL'] = candidate
                        break
                else:
                    # 如果没有找到合适的替代模型，使用第一个可用模型
                    if available_models:
                        logger.info(f"未找到推荐对话模型，使用 {available_models[0]}")
                        app.config['LLM_MODEL'] = available_models[0]
            
            # 检查嵌入模型是否可用
            if app.config['EMBEDDING_MODEL'] not in available_models:
                # 尝试找到替代嵌入模型
                for candidate in ["nomic-embed-text", "mxbai-embed", "all-minilm", "bge-large"]:
                    if candidate in available_models:
                        logger.info(f"默认嵌入模型 {app.config['EMBEDDING_MODEL']} 不可用，切换到 {candidate}")
                        app.config['EMBEDDING_MODEL'] = candidate
                        break
                else:
                    # 如果没有找到合适的替代嵌入模型，使用RAG模型
                    logger.info(f"未找到推荐嵌入模型，使用RAG模型 {app.config['RAG_MODEL']} 进行嵌入")
                    app.config['EMBEDDING_MODEL'] = app.config['RAG_MODEL']
        
        logger.info(f"最终使用的模型 - 对话: {app.config['LLM_MODEL']}, RAG: {app.config['RAG_MODEL']}, 嵌入: {app.config['EMBEDDING_MODEL']}")
    
    # 生产环境推荐配置提示
    logger.info("=" * 60)
    logger.info("高并发部署建议:")
    logger.info("1. 生产环境推荐使用: gunicorn -w 4 -k gevent --timeout 120 App.app:app")
    logger.info("2. 或使用: uwsgi --http :8080 --wsgi-file App/app.py --callable app --processes 4 --threads 2")
    logger.info("3. 配合Nginx反向代理获得更好性能")
    logger.info("4. 建议启用Redis缓存替代内存缓存")
    logger.info("=" * 60)
    
    try:
        # 启动服务器 - 开发环境配置
        # 注意：在生产环境中，应该使用 gunicorn 或 uwsgi
        app.run(
            host='0.0.0.0', 
            port=8080, 
            debug=False,  # 高并发环境下关闭debug模式
            threaded=True,  # 启用多线程
            use_reloader=False  # 关闭自动重载，提高性能
        )
    except Exception as e:
        logger.error(f"服务器运行出错: {str(e)}")
    finally:
        logger.info("服务器关闭，执行最终清理...")
        cleanup_resources()