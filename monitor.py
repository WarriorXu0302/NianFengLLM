#!/usr/bin/env python3
"""
高并发农业助手性能监控和管理工具
用于监控应用性能、缓存状态和系统资源使用情况
"""

import requests
import time
import json
import sys
import psutil
import threading
from datetime import datetime
import os

class PerformanceMonitor:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.stats = {
            'requests_sent': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_response_time': 0,
            'min_response_time': float('inf'),
            'max_response_time': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        self.running = False
    
    def check_server_status(self):
        """检查服务器状态"""
        try:
            response = requests.get(f"{self.base_url}/system_info", timeout=5)
            if response.status_code == 200:
                return True, response.json()
            return False, f"HTTP {response.status_code}"
        except Exception as e:
            return False, str(e)
    
    def test_query_performance(self, query="你好", num_requests=10):
        """测试查询性能"""
        print(f"\n开始性能测试: {num_requests} 个请求")
        print("-" * 50)
        
        for i in range(num_requests):
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/query_document",
                    json={"query": f"{query} {i}"},
                    timeout=30
                )
                end_time = time.time()
                response_time = end_time - start_time
                
                self.stats['requests_sent'] += 1
                self.stats['total_response_time'] += response_time
                self.stats['min_response_time'] = min(self.stats['min_response_time'], response_time)
                self.stats['max_response_time'] = max(self.stats['max_response_time'], response_time)
                
                if response.status_code == 200:
                    self.stats['successful_requests'] += 1
                    result = response.json()
                    if result.get('cached', False):
                        self.stats['cache_hits'] += 1
                    else:
                        self.stats['cache_misses'] += 1
                    print(f"请求 {i+1}: {response_time:.2f}s {'(缓存)' if result.get('cached') else '(新)'}")
                else:
                    self.stats['failed_requests'] += 1
                    print(f"请求 {i+1}: 失败 (HTTP {response.status_code})")
                    
            except Exception as e:
                self.stats['failed_requests'] += 1
                print(f"请求 {i+1}: 错误 - {str(e)}")
        
        self.print_stats()
    
    def test_concurrent_requests(self, query="你好", num_threads=5, requests_per_thread=5):
        """测试并发请求性能"""
        print(f"\n开始并发测试: {num_threads} 个线程，每个线程 {requests_per_thread} 个请求")
        print("-" * 60)
        
        def worker_thread(thread_id):
            for i in range(requests_per_thread):
                start_time = time.time()
                try:
                    response = requests.post(
                        f"{self.base_url}/query_document",
                        json={"query": f"{query} T{thread_id}-{i}"},
                        timeout=30
                    )
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    self.stats['requests_sent'] += 1
                    self.stats['total_response_time'] += response_time
                    self.stats['min_response_time'] = min(self.stats['min_response_time'], response_time)
                    self.stats['max_response_time'] = max(self.stats['max_response_time'], response_time)
                    
                    if response.status_code == 200:
                        self.stats['successful_requests'] += 1
                        result = response.json()
                        if result.get('cached', False):
                            self.stats['cache_hits'] += 1
                        else:
                            self.stats['cache_misses'] += 1
                        print(f"线程{thread_id}-请求{i+1}: {response_time:.2f}s {'(缓存)' if result.get('cached') else '(新)'}")
                    else:
                        self.stats['failed_requests'] += 1
                        print(f"线程{thread_id}-请求{i+1}: 失败")
                        
                except Exception as e:
                    self.stats['failed_requests'] += 1
                    print(f"线程{thread_id}-请求{i+1}: 错误 - {str(e)}")
        
        # 创建并启动线程
        threads = []
        start_time = time.time()
        
        for thread_id in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        print(f"\n并发测试完成，总耗时: {total_time:.2f}秒")
        self.print_stats()
    
    def monitor_system_resources(self, duration=60):
        """监控系统资源使用情况"""
        print(f"\n开始监控系统资源 ({duration} 秒)")
        print("-" * 50)
        
        start_time = time.time()
        while time.time() - start_time < duration:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            
            # 磁盘I/O
            disk_io = psutil.disk_io_counters()
            
            # 网络I/O
            net_io = psutil.net_io_counters()
            
            print(f"{datetime.now().strftime('%H:%M:%S')} | "
                  f"CPU: {cpu_percent:5.1f}% | "
                  f"内存: {memory.percent:5.1f}% | "
                  f"磁盘读: {disk_io.read_bytes/1024/1024:6.1f}MB | "
                  f"网络接收: {net_io.bytes_recv/1024/1024:6.1f}MB")
            
            time.sleep(5)
    
    def print_stats(self):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("性能统计报告")
        print("=" * 60)
        
        if self.stats['requests_sent'] > 0:
            avg_response_time = self.stats['total_response_time'] / self.stats['requests_sent']
            success_rate = (self.stats['successful_requests'] / self.stats['requests_sent']) * 100
            cache_hit_rate = (self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])) * 100 if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
            
            print(f"总请求数: {self.stats['requests_sent']}")
            print(f"成功请求: {self.stats['successful_requests']}")
            print(f"失败请求: {self.stats['failed_requests']}")
            print(f"成功率: {success_rate:.2f}%")
            print(f"平均响应时间: {avg_response_time:.2f}秒")
            print(f"最短响应时间: {self.stats['min_response_time']:.2f}秒")
            print(f"最长响应时间: {self.stats['max_response_time']:.2f}秒")
            print(f"缓存命中率: {cache_hit_rate:.2f}%")
            print(f"缓存命中: {self.stats['cache_hits']}")
            print(f"缓存未命中: {self.stats['cache_misses']}")
        else:
            print("没有统计数据")
        
        print("=" * 60)
    
    def reset_stats(self):
        """重置统计数据"""
        self.stats = {
            'requests_sent': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_response_time': 0,
            'min_response_time': float('inf'),
            'max_response_time': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

def main():
    monitor = PerformanceMonitor()
    
    print("智能农业助手性能监控工具")
    print("=" * 40)
    
    # 检查服务器状态
    status, info = monitor.check_server_status()
    if status:
        print("✓ 服务器状态: 正常")
        print(f"✓ 服务器信息: {info.get('device', 'N/A')}")
    else:
        print(f"✗ 服务器状态: {info}")
        return
    
    while True:
        print("\n选择测试类型:")
        print("1. 顺序性能测试")
        print("2. 并发性能测试")
        print("3. 系统资源监控")
        print("4. 服务器状态检查")
        print("5. 重置统计数据")
        print("0. 退出")
        
        choice = input("\n请选择 (0-5): ").strip()
        
        if choice == '1':
            num_requests = int(input("请求数量 (默认10): ") or "10")
            query = input("测试查询 (默认'你好'): ") or "你好"
            monitor.test_query_performance(query, num_requests)
            
        elif choice == '2':
            num_threads = int(input("线程数 (默认5): ") or "5")
            requests_per_thread = int(input("每线程请求数 (默认5): ") or "5")
            query = input("测试查询 (默认'你好'): ") or "你好"
            monitor.test_concurrent_requests(query, num_threads, requests_per_thread)
            
        elif choice == '3':
            duration = int(input("监控时长/秒 (默认60): ") or "60")
            monitor.monitor_system_resources(duration)
            
        elif choice == '4':
            status, info = monitor.check_server_status()
            if status:
                print("✓ 服务器状态: 正常")
                print(json.dumps(info, indent=2, ensure_ascii=False))
            else:
                print(f"✗ 服务器状态: {info}")
                
        elif choice == '5':
            monitor.reset_stats()
            print("统计数据已重置")
            
        elif choice == '0':
            print("退出监控工具")
            break
            
        else:
            print("无效选择，请重试")

if __name__ == "__main__":
    main() 