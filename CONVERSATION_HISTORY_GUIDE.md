# 对话历史功能使用指南

## 概述

年丰助手现已支持对话历史记忆功能，模型能够记住之前的对话内容，实现更自然、连贯的多轮对话体验。

## 功能特性

### 1. 自动对话管理
- 自动生成唯一对话ID
- 智能记忆管理（最多50条消息/对话）
- 自动清理过期对话（最多1000个对话）

### 2. 上下文感知
- 模型记住用户之前提到的信息
- 支持基于历史的后续问题
- 智能理解指代关系

### 3. 灵活控制
- 可选择是否使用对话历史
- 支持创建新对话
- 可清除特定对话历史

## API接口使用

### 1. 带历史的对话 `/llama2_chat`

```json
POST /llama2_chat
{
  "prompt": "你好，我是李农夫，我种植了10亩玉米",
  "conversation_id": "abc12345",  // 可选，不提供会自动生成
  "use_history": true             // 可选，默认true
}

响应:
{
  "result": "您好李农夫！很高兴认识您...",
  "conversation_id": "abc12345"
}
```

### 2. 带历史的RAG查询 `/query_document`

```json
POST /query_document
{
  "query": "玉米叶子黄了怎么办？",
  "conversation_id": "abc12345",
  "use_history": true,
  "mode": "hybrid"
}

响应:
{
  "result": "针对您种植的玉米出现叶子发黄的问题...",
  "conversation_id": "abc12345",
  "cached": false
}
```

### 3. 对话管理接口

#### 创建新对话
```json
POST /api/conversations/new

响应:
{
  "conversation_id": "def67890",
  "message": "新对话已创建"
}
```

#### 获取对话历史
```json
GET /api/conversations/{conversation_id}?include_system=true

响应:
{
  "conversation_id": "abc12345",
  "history": [
    {
      "role": "user",
      "content": "你好，我是李农夫...",
      "timestamp": 1625097600,
      "formatted_time": "2025-06-04 08:00:00"
    },
    {
      "role": "assistant", 
      "content": "您好李农夫！很高兴认识您...",
      "timestamp": 1625097605,
      "formatted_time": "2025-06-04 08:00:05"
    }
  ],
  "message_count": 2
}
```

#### 列出所有对话
```json
GET /api/conversations

响应:
{
  "conversations": [
    {
      "conversation_id": "abc12345",
      "message_count": 6,
      "last_activity": 1625097600,
      "created": 1625097500,
      "last_activity_formatted": "2025-06-04 08:00:00",
      "created_formatted": "2025-06-04 07:58:20"
    }
  ],
  "total": 1
}
```

#### 清除对话
```json
DELETE /api/conversations/{conversation_id}

响应:
{
  "success": true,
  "message": "对话 abc12345 已清除"
}
```

#### 获取统计信息
```json
GET /api/conversations/stats

响应:
{
  "total_conversations": 5,
  "total_messages": 25,
  "average_messages_per_conversation": 5.0
}
```

## 使用示例

### 1. 基础多轮对话

```python
import requests

base_url = "http://localhost:8080"

# 第一轮：用户介绍
response = requests.post(f"{base_url}/llama2_chat", json={
    "prompt": "你好，我是张农民，种植了20亩水稻"
})
conversation_id = response.json()['conversation_id']

# 第二轮：询问问题
response = requests.post(f"{base_url}/llama2_chat", json={
    "prompt": "水稻出现病斑怎么办？",
    "conversation_id": conversation_id
})

# 第三轮：基于历史的后续问题
response = requests.post(f"{base_url}/llama2_chat", json={
    "prompt": "针对我的情况，用药量应该是多少？",
    "conversation_id": conversation_id
})
# 模型会记住用户种植水稻，面积20亩
```

### 2. RAG查询与历史结合

```python
# 第一轮：一般性查询
response = requests.post(f"{base_url}/query_document", json={
    "query": "什么是病虫害综合防治？",
    "conversation_id": conversation_id
})

# 第二轮：基于历史的深入查询
response = requests.post(f"{base_url}/query_document", json={
    "query": "你刚才提到的生物防治具体怎么实施？",
    "conversation_id": conversation_id
})
# 模型会结合之前关于综合防治的回答
```

### 3. 无历史对话

```python
# 禁用历史，每次都是独立对话
response = requests.post(f"{base_url}/llama2_chat", json={
    "prompt": "我种植的是什么作物？",
    "conversation_id": conversation_id,
    "use_history": False  # 禁用历史
})
# 模型不会知道用户之前提到的作物信息
```

## 前端集成示例

### JavaScript示例

```javascript
class ConversationManager {
    constructor(baseUrl = 'http://localhost:8080') {
        this.baseUrl = baseUrl;
        this.currentConversationId = null;
    }

    // 创建新对话
    async createNewConversation() {
        const response = await fetch(`${this.baseUrl}/api/conversations/new`, {
            method: 'POST'
        });
        const data = await response.json();
        this.currentConversationId = data.conversation_id;
        return data.conversation_id;
    }

    // 发送消息
    async sendMessage(message, useHistory = true) {
        const response = await fetch(`${this.baseUrl}/llama2_chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                prompt: message,
                conversation_id: this.currentConversationId,
                use_history: useHistory
            })
        });
        
        const data = await response.json();
        this.currentConversationId = data.conversation_id;
        return data.result;
    }

    // 获取对话历史
    async getHistory() {
        if (!this.currentConversationId) return [];
        
        const response = await fetch(`${this.baseUrl}/api/conversations/${this.currentConversationId}`);
        const data = await response.json();
        return data.history;
    }

    // 清除当前对话
    async clearConversation() {
        if (this.currentConversationId) {
            await fetch(`${this.baseUrl}/api/conversations/${this.currentConversationId}`, {
                method: 'DELETE'
            });
            this.currentConversationId = null;
        }
    }
}

// 使用示例
const chatManager = new ConversationManager();

// 开始新对话
await chatManager.createNewConversation();

// 发送消息
const response1 = await chatManager.sendMessage("你好，我是农民李四");
const response2 = await chatManager.sendMessage("我种植的玉米出现病虫害");
const response3 = await chatManager.sendMessage("针对我的情况有什么建议？");

// 查看历史
const history = await chatManager.getHistory();
console.log("对话历史:", history);
```

## 性能优化

### 1. 缓存策略
- 有历史的查询缓存时间较短（1分钟）
- 无历史的查询缓存时间较长（5分钟）
- 模型预测结果缓存（30分钟）

### 2. 内存管理
- 每个对话最多保留50条消息
- 系统最多保留1000个对话
- 自动清理最旧的对话和消息

### 3. 并发处理
- 支持多用户同时对话
- 线程安全的对话管理
- 事件循环优化

## 测试验证

使用提供的测试脚本验证功能：

```bash
# 测试对话历史功能
python test_conversation_history.py

# 测试并发和事件循环修复
python test_fix.py
```

## 注意事项

1. **Token限制**: 每个请求最多包含20条历史消息，避免超过模型token限制
2. **隐私安全**: 对话历史存储在内存中，服务重启后会丢失
3. **性能考虑**: 启用历史会稍微增加响应时间，但提供更好的对话体验
4. **缓存影响**: 对话历史会影响缓存效果，相同问题在不同上下文下可能产生不同回答

## 未来扩展

- [ ] 持久化存储（数据库/Redis）
- [ ] 对话摘要和压缩
- [ ] 更智能的上下文管理
- [ ] 用户身份管理
- [ ] 对话分析和统计 