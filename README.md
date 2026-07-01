# 数字人视频生成服务 (Digital Human Video Generation Service)

基于 **SoulX-FlashHead**（数字人口播视频生成）和 **LongCat-AudioDiT**（语音克隆/TTS）构建的完整 REST API 服务。

## 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                    FastAPI Server (:5000)                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ Avatars  │  │  Tasks   │  │  TTS     │  │ Download │ │
│  │  CRUD    │  │  Queue   │  │ Generate │  │  Files   │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┘ │
│       │              │              │                      │
│  ┌────▼──────────────▼──────────────▼─────────────────┐  │
│  │              Background Worker                      │  │
│  │  ┌─────────────────┐    ┌────────────────────────┐ │  │
│  │  │ LongCat-AudioDiT │    │  SoulX-FlashHead      │ │  │
│  │  │ (Voice Cloning)  │───►│  (Talking Head Gen)   │ │  │
│  │  │ TTS + Clone      │    │  Image + Audio → Video│ │  │
│  │  └─────────────────┘    └────────────────────────┘ │  │
│  └─────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

## 工作流程

```
用户输入文本 + 选择数字人
         │
         ▼
┌─────────────────────┐
│ 1. LongCat-AudioDiT  │  ← 根据数字人的声音样本，对输入文本进行语音克隆
│    TTS / Voice Clone  │
└─────────┬───────────┘
          │ 生成的音频
          ▼
┌─────────────────────┐
│ 2. SoulX-FlashHead   │  ← 用数字人照片 + 生成的语音，合成口播视频
│    Talking Head Gen   │
└─────────┬───────────┘
          │ MP4 视频
          ▼
      下载 / 查看
```

## 安装与部署

### 前置条件

- Python 3.10+
- NVIDIA GPU + CUDA (RTX 4090 推荐，24GB 显存足够)
- FFmpeg
- 约 20GB 磁盘空间（模型权重 + 代码）

### 📌 RTX 4090 一键部署（国内镜像加速版）

完整复制以下命令即可：

```bash
# ========== 1. 克隆项目 ==========
git clone git@github.com:yanhuicsdn/digital-human.git
cd digital-human

# ========== 2. 创建 Conda 环境 ==========
conda create -n digital-human python=3.10 -y
conda activate digital-human

# ========== 3. 安装 PyTorch（国内清华源加速）==========
pip install torch==2.7.1 torchvision==0.22.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# ========== 4. 安装项目依赖 ==========
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# ========== 5. 安装 FlashAttention（4090 推理加速必装）==========
pip install ninja
pip install flash_attn==2.8.0.post2 --no-build-isolation \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

# ========== 6. 安装 ModelScope 并下载模型权重 ==========
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

# SoulX-FlashHead 1.3B（含 Lite & Pro，HF 镜像下载）
export HF_ENDPOINT=https://hf-mirror.com
pip install "huggingface_hub[cli]" -i https://pypi.tuna.tsinghua.edu.cn/simple
huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B \
  --local-dir ./models/SoulX-FlashHead-1_3B

# Wav2Vec 音频编码器（ModelScope 下载）
modelscope download --model AI-ModelScope/wav2vec2-base-960h \
  --local_dir ./models/wav2vec2-base-960h

# LongCat-AudioDiT 语音克隆模型（ModelScope 下载）
modelscope download --model meituan-longcat/LongCat-AudioDiT-1B \
  --local_dir ./models/LongCat-AudioDiT-1B

# ========== 7. 下载 SoulX-FlashHead 源码 ==========
git clone --depth 1 https://github.com/Soul-AILab/SoulX-FlashHead.git /tmp/fh
cp -r /tmp/fh/flash_head ./
# configs 已在 flash_head 子目录中

# ========== 8. 启动服务（Lite 模型，RTX 4090 最优）==========
export FLASHHEAD_MODEL_TYPE=lite
./run.sh
```

服务启动后：
- API 文档: http://localhost:5000/docs
- ReDoc: http://localhost:5000/redoc

### 分步说明（如需自定义）

#### 创建 Conda 环境

```bash
conda create -n digital-human python=3.10 -y
conda activate digital-human
```

#### 安装 PyTorch

```bash
pip install torch==2.7.1 torchvision==0.22.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 安装项目依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### FlashAttention（4090 推理加速，必装）

```bash
pip install ninja
pip install flash_attn==2.8.0.post2 --no-build-isolation \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 下载模型权重（ModelScope + HF 镜像）

```bash
# 安装 ModelScope（国内模型下载工具）
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

# SoulX-FlashHead 1.3B（仅在 HuggingFace 发布，走 HF 镜像）
export HF_ENDPOINT=https://hf-mirror.com
pip install "huggingface_hub[cli]" -i https://pypi.tuna.tsinghua.edu.cn/simple
huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B \
  --local-dir ./models/SoulX-FlashHead-1_3B

# Wav2Vec 音频编码器（ModelScope 下载）
modelscope download --model AI-ModelScope/wav2vec2-base-960h \
  --local_dir ./models/wav2vec2-base-960h

# LongCat-AudioDiT 语音克隆模型（ModelScope 下载）
modelscope download --model meituan-longcat/LongCat-AudioDiT-1B \
  --local_dir ./models/LongCat-AudioDiT-1B
```

#### 下载 FlashHead 源码

```bash
git clone --depth 1 https://github.com/Soul-AILab/SoulX-FlashHead.git /tmp/fh
cp -r /tmp/fh/flash_head ./
```

#### 启动服务

```bash
export FLASHHEAD_MODEL_TYPE=lite
./run.sh
# 或 python start.py
```

## API 文档

### 系统

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 健康检查 / 服务信息 |

### 数字人管理 (Avatar)

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/avatars` | 获取所有数字人列表 |
| GET | `/api/avatar_image/{avatar_id}` | 获取数字人图片 |
| GET | `/api/avatar_audio/{avatar_id}` | 获取数字人音频 |
| POST | `/api/avatar/create` | 创建新数字人（multipart） |
| DELETE | `/api/avatar/{avatar_id}` | 删除数字人 |

### 文本生成 (LLM)

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/generate_text` | 使用 LLM 生成文本 |
| POST | `/api/generate_text_stream` | 流式生成文本 (SSE) |

### 任务管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/submit` | 提交视频生成任务 |
| GET | `/api/status/{task_id}` | 查看任务状态 |
| GET | `/api/tasks` | 获取所有任务 |
| DELETE | `/api/tasks/{task_id}` | 删除排队中的任务 |
| GET | `/api/queue` | 获取队列统计信息 |

### 文件下载

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/download/{filename}` | 下载生成的文件（视频/音频） |

## API 使用示例

### 1. 创建数字人

```bash
curl -X POST http://localhost:5000/api/avatar/create \
  -F "name=张三" \
  -F "description=测试数字人" \
  -F "prompt_text=今天天气真好，适合出去玩。" \
  -F "image=@/path/to/portrait.jpg" \
  -F "audio=@/path/to/voice_sample.wav"
```

### 2. 提交视频生成任务

```bash
curl -X POST http://localhost:5000/api/submit \
  -H "Content-Type: application/json" \
  -d '{
    "text": "大家好，今天我们来聊聊人工智能的发展趋势。",
    "avatar_id": "YOUR_AVATAR_ID"
  }'
```

### 3. 查看任务状态

```bash
curl http://localhost:5000/api/status/YOUR_TASK_ID
```

### 4. 下载生成结果

```bash
curl -o output.mp4 http://localhost:5000/download/YOUR_VIDEO_FILENAME
```

### 5. 生成文本（可选 - 使用 LLM）

```bash
curl -X POST http://localhost:5000/api/generate_text \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "写一段关于人工智能的介绍",
    "api_key": "sk-xxx",
    "model_id": "qwen-turbo"
  }'
```

## 任务状态说明

| 状态 | 说明 |
|------|------|
| `queued` | 排队中，等待处理 |
| `processing` | 正在处理（TTS生成 → 视频生成） |
| `completed` | 已完成，可下载视频 |
| `failed` | 处理失败，查看 error 字段了解原因 |

## 配置说明（环境变量）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HOST` | `0.0.0.0` | 服务监听地址 |
| `PORT` | `5000` | 服务端口 |
| `FLASHHEAD_CKPT_DIR` | `./models/SoulX-FlashHead-1_3B` | FlashHead 模型路径 |
| `FLASHHEAD_WAV2VEC_DIR` | `./models/wav2vec2-base-960h` | Wav2Vec 模型路径 |
| `FLASHHEAD_MODEL_TYPE` | `lite` | FlashHead 模型类型 (`lite`/`pro`) |
| `AUDIODIT_MODEL_DIR` | `./models/LongCat-AudioDiT-1B` | AudioDiT 模型路径 |
| `LLM_API_KEY` | `""` | LLM API 密钥 |
| `LLM_BASE_URL` | `https://dashscope.aliyuncs.com/...` | LLM API 基础 URL |
| `LLM_DEFAULT_MODEL` | `qwen-turbo` | 默认 LLM 模型 |

## 项目结构

```
digital-human-service/
├── app/                        # FastAPI 应用
│   ├── main.py                 # 入口 / 路由注册
│   ├── config.py               # 配置
│   ├── models.py               # Pydantic 数据模型
│   ├── database.py             # JSON 文件数据库
│   ├── storage.py              # 文件存储管理
│   ├── routers/
│   │   ├── index.py            # GET /
│   │   ├── avatars.py          # 数字人 CRUD
│   │   ├── text_generation.py  # LLM 文本生成
│   │   ├── tasks.py            # 任务队列管理
│   │   └── download.py         # 文件下载
│   └── services/
│       ├── tts_service.py      # LongCat-AudioDiT 集成
│       ├── video_service.py    # SoulX-FlashHead 集成
│       └── task_queue.py       # 后台任务处理
├── data/                       # 数据目录
│   ├── avatars/                # 数字人图片/音频
│   ├── generated_audio/        # TTS 生成的音频
│   ├── generated_videos/       # 生成的口播视频
│   └── db.json                 # 数据库文件
├── models/                     # 模型权重（需手动下载）
│   ├── SoulX-FlashHead-1_3B/
│   ├── wav2vec2-base-960h/
│   └── LongCat-AudioDiT-1B/
├── flash_head/                 # SoulX-FlashHead 源码
├── utils.py                    # 工具函数
├── requirements.txt            # Python 依赖
├── run.sh                      # 启动脚本
├── start.py                    # 启动入口
└── README.md
```

## 模型说明

### SoulX-FlashHead

- **Lite 模型**: 单 RTX 4090 即可实现实时（25+ FPS）推理，最高支持 3 路并发
- **Pro 模型**: 生成更高质量视频，单卡 10.8 FPS，双 RTX 5090 可实时
- 更多信息: [SoulX-FlashHead GitHub](https://github.com/Soul-AILab/SoulX-FlashHead)

### LongCat-AudioDiT

- **1B 模型**: 高质量语音克隆/TTS，直接操作波形潜空间
- **3.5B 模型**: 更高音质（需更多显存）
- Seed 基准测试 SOTA 性能
- 更多信息: [LongCat-AudioDiT GitHub](https://github.com/meituan-longcat/LongCat-AudioDiT)

## 原创 API 参考

原始 API 服务定义（本系统复现自）：

```
GET  /                           # 健康检查
GET  /api/avatars                # 数字人列表
GET  /api/avatar_image/{id}      # 获取数字人图片
POST /api/avatar/create          # 创建数字人
DELETE /api/avatar/{id}          # 删除数字人
POST /api/generate_text          # LLM 文本生成
POST /api/generate_text_stream   # 流式文本生成
GET  /api/tasks                  # 所有任务
DELETE /api/tasks/{id}           # 删除任务
GET  /api/queue                  # 队列状态
POST /api/submit                 # 提交任务
GET  /api/status/{id}            # 任务状态
GET  /download/{filename}        # 下载文件
```

## 许可证

本项目整合了两个开源模型的成果：
- **SoulX-FlashHead**: Apache 2.0
- **LongCat-AudioDiT**: MIT
- **本服务代码**: MIT
