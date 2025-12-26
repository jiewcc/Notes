### 启动

```bash
python3 -m sglang.launch_server \
    --model-path /root/.cache/huggingface/DeepSeek-R1-Distill-Qwen-1.5B \
    --host 0.0.0.0 \
    --port 30000
```

### 网页对话

```bash
python3 -m http.server 3000
```

| 部分          | 解释                              |
| ------------- | --------------------------------- |
| `python3`     | Python 3 解释器                   |
| `-m`          | 模块模式，运行指定模块            |
| `http.server` | Python 标准库中的 HTTP 服务器模块 |
| `3000`        | 服务器监听的端口号                |

```bash
# 创建优化的单文件界面
cd ~
mkdir -p deepseek-chat && cd deepseek-chat

cat > index.html << 'HTML'
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepSeek 聊天界面</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #4c51bf 0%, #667eea 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .chat-area {
            height: 60vh;
            overflow-y: auto;
            padding: 20px;
        }
        .message {
            margin: 10px 0;
            max-width: 80%;
        }
        .user-msg {
            margin-left: auto;
            background: #667eea;
            color: white;
            padding: 10px 15px;
            border-radius: 15px 15px 5px 15px;
        }
        .bot-msg {
            background: #f3f4f6;
            padding: 10px 15px;
            border-radius: 5px 15px 15px 15px;
        }
        .input-area {
            padding: 20px;
            border-top: 1px solid #e5e7eb;
            display: flex;
            gap: 10px;
        }
        #userInput {
            flex: 1;
            padding: 12px;
            border: 2px solid #e5e7eb;
            border-radius: 10px;
            font-size: 16px;
        }
        #userInput:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            background: #667eea;
            color: white;
            border: none;
            padding: 0 25px;
            border-radius: 10px;
            cursor: pointer;
            font-weight: bold;
        }
        .typing {
            display: flex;
            gap: 5px;
            padding: 10px;
        }
        .typing span {
            width: 8px;
            height: 8px;
            background: #667eea;
            border-radius: 50%;
            animation: bounce 1.4s infinite;
        }
        .typing span:nth-child(2) { animation-delay: 0.2s; }
        .typing span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 DeepSeek AI 聊天助手</h1>
            <p>连接到: http://localhost:30000</p>
        </div>
        <div id="chat" class="chat-area"></div>
        <div id="typing" class="typing" style="display:none">
            <span></span><span></span><span></span>
        </div>
        <div class="input-area">
            <input type="text" id="userInput" placeholder="输入您的问题..." autocomplete="off">
            <button onclick="sendMessage()">发送</button>
        </div>
    </div>

    <script>
        const API_URL = 'http://localhost:30000/v1/chat/completions';
        const MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B';

        function addMessage(text, sender) {
            const chat = document.getElementById('chat');
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${sender}-msg`;
            msgDiv.textContent = text;
            chat.appendChild(msgDiv);
            chat.scrollTop = chat.scrollHeight;
        }

        async function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            if (!message) return;
            
            addMessage(message, 'user');
            input.value = '';
            input.disabled = true;
            
            document.getElementById('typing').style.display = 'flex';
            
            try {
                const response = await fetch(API_URL, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: MODEL,
                        messages: [{ role: 'user', content: message }],
                        max_tokens: 1000,
                        temperature: 0.7
                    })
                });
                
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                
                const data = await response.json();
                if (data.choices?.[0]?.message?.content) {
                    addMessage(data.choices[0].message.content, 'bot');
                } else {
                    addMessage('抱歉，没有收到有效的回复。', 'bot');
                }
            } catch (error) {
                addMessage(`错误: ${error.message}`, 'bot');
            } finally {
                input.disabled = false;
                input.focus();
                document.getElementById('typing').style.display = 'none';
            }
        }

        document.getElementById('userInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

        // 初始消息
        setTimeout(() => {
            addMessage('👋 您好！我是 DeepSeek AI 助手。有什么可以帮您的吗？', 'bot');
        }, 500);
    </script>
</body>
</html>
HTML

# 启动服务器
python3 -m http.server 3000
# 访问 http://localhost:3000
```

