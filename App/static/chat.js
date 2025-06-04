// static/chat.js
document.addEventListener('DOMContentLoaded', function() {
    // 获取DOM元素
    const chatMessages = document.getElementById('chat-messages');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const uploadButton = document.getElementById('upload-button');
    const imageUpload = document.getElementById('image-upload');
    const processingIndicator = document.getElementById('processing-indicator');
    const clearChatButton = document.getElementById('clear-chat-btn');

    // 图片预览相关元素
    const imagePreviewModal = document.getElementById('image-preview-modal');
    const previewImage = document.getElementById('preview-image');
    const cancelUploadButton = document.getElementById('cancel-upload');
    const confirmUploadButton = document.getElementById('confirm-upload');
    const closeModalButton = document.querySelector('.close-modal');

    // 当前选中的图片文件
    let currentImageFile = null;

    // API基础URL
    const API_BASE_URL = 'http://localhost:8080';

    // 初始化自动调整输入框高度
    initializeTextareaAutoresize();

    // 发送按钮点击事件
    sendButton.addEventListener('click', sendMessage);

    // 输入框按下Enter键发送消息（但Shift+Enter换行）
    messageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // 上传按钮点击事件
    uploadButton.addEventListener('click', function() {
        imageUpload.click();
    });

    // 图片选择事件
    imageUpload.addEventListener('change', function(e) {
        if (e.target.files && e.target.files[0]) {
            // 保存当前选择的文件
            currentImageFile = e.target.files[0];

            // 使用FileReader预览图片
            const reader = new FileReader();
            reader.onload = function(event) {
                previewImage.src = event.target.result;
                // 显示预览模态框
                imagePreviewModal.style.display = 'flex';
            };
            reader.readAsDataURL(currentImageFile);
        }
    });

    // 取消上传按钮
    cancelUploadButton.addEventListener('click', function() {
        imagePreviewModal.style.display = 'none';
        currentImageFile = null;
        imageUpload.value = '';
    });

    // 确认上传按钮
    confirmUploadButton.addEventListener('click', function() {
        imagePreviewModal.style.display = 'none';
        if (currentImageFile) {
            processAndSendImage(currentImageFile);
        }
    });

    // 关闭模态框按钮
    closeModalButton.addEventListener('click', function() {
        imagePreviewModal.style.display = 'none';
        currentImageFile = null;
        imageUpload.value = '';
    });

    // 清空聊天按钮
    clearChatButton.addEventListener('click', function() {
        if (confirm('确定要清空所有聊天记录吗？')) {
            // 移除所有消息，只保留欢迎消息
            while (chatMessages.children.length > 1) {
                chatMessages.removeChild(chatMessages.lastChild);
            }
        }
    });

    // 发送消息函数
    function sendMessage() {
        const message = messageInput.value.trim();
        if (message) {
            // 添加用户消息到界面
            addMessage('user', message);
            // 清空输入框
            messageInput.value = '';
            // 重置输入框高度
            messageInput.style.height = 'auto';
            // 发送到后端处理
            processUserMessage(message);
        }
    }

    // 处理并发送图片
    function processAndSendImage(imageFile) {
        // 显示用户图片消息
        addImageMessage('user', URL.createObjectURL(imageFile));

        // 显示加载指示器
        showProcessingIndicator();

        // 创建FormData对象
        const formData = new FormData();
        formData.append('image', imageFile);

        // 发送到植物病害识别API
        fetch(`${API_BASE_URL}/predict_disease`, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(response.statusText);
            }
            return response.json();
        })
        .then(data => {
            // 处理分类结果
            if (data.error) {
                throw new Error(data.error);
            }

            // 获取图片分类结果、概率和模型生成的解释
            const classResult = data.class;
            const probability = (data.probability * 100).toFixed(2);
            const explanation = data.explanation || ''; // 获取润色后的解释

            // 创建结果文本
            let resultText = `【年丰分析结果】\n检测类别：**${classResult}**\n置信度：${probability}%\n\n${explanation}`;
            
            // 添加助手回复消息
            addMessage('assistant', resultText);
            
            // 隐藏加载指示器
            hideProcessingIndicator();
        })
        .catch(error => {
            console.error('图片处理错误:', error);
            // 添加错误消息
            addMessage('assistant', `很抱歉，处理图片时出现了问题：${error.message || '未知错误'}`);
            // 隐藏加载指示器
            hideProcessingIndicator();
        });
    }

    // 处理用户文本消息
    function processUserMessage(message) {
        // 显示加载指示器
        showProcessingIndicator();

        // 发送到查询API - 现在后端已经集成了RAG和直接对话的逻辑
        // 如果RAG返回no-context，后端会自动切换到Llama2直接回答
        fetch(`${API_BASE_URL}/query_document`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: message
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(response.statusText);
            }
            return response.json();
        })
        .then(data => {
            // 添加助手回复消息
            if (data.error) {
                throw new Error(data.error);
            }
            addMessage('assistant', data.result);
            // 隐藏加载指示器
            hideProcessingIndicator();
        })
        .catch(error => {
            console.error('处理消息错误:', error);
            // 添加错误消息
            addMessage('assistant', `很抱歉，处理您的消息时出现了问题：${error.message || '未知错误'}`);
            // 隐藏加载指示器
            hideProcessingIndicator();
        });
    }

    // 添加文本消息到聊天界面
    function addMessage(sender, text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';

        const icon = document.createElement('i');
        icon.className = sender === 'user' ? 'fas fa-user' : 'fas fa-robot';
        avatar.appendChild(icon);

        const content = document.createElement('div');
        content.className = 'message-content';

        // 处理文本中的换行符和Markdown格式的加粗
        let formattedText = text.replace(/\n/g, '<br>');
        // 简单的Markdown加粗处理 **text** -> <strong>text</strong>
        formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        content.innerHTML = `<p>${formattedText}</p>`;

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);

        chatMessages.appendChild(messageDiv);

        // 滚动到底部
        scrollToBottom();
    }

    // 添加图片消息到聊天界面
    function addImageMessage(sender, imageUrl) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';

        const icon = document.createElement('i');
        icon.className = sender === 'user' ? 'fas fa-user' : 'fas fa-robot';
        avatar.appendChild(icon);

        const content = document.createElement('div');
        content.className = 'message-content with-image';

        const image = document.createElement('img');
        image.src = imageUrl;
        image.alt = "上传的图片";

        content.appendChild(image);

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);

        chatMessages.appendChild(messageDiv);

        // 滚动到底部
        scrollToBottom();
    }

    // 显示处理指示器
    function showProcessingIndicator() {
        processingIndicator.style.display = 'flex';
    }

    // 隐藏处理指示器
    function hideProcessingIndicator() {
        processingIndicator.style.display = 'none';
    }

    // 滚动到底部
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // 初始化文本框自动调整高度
    function initializeTextareaAutoresize() {
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    }
});