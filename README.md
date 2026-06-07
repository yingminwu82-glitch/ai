# AI 美甲虚拟试戴 · Nail Try-On

基于通义千问图像编辑 + 多模态推荐 + MediaPipe 指甲定位的 AI 美甲虚拟试戴系统

## 项目简介

用户上传一张手部照片, 系统自动:
1. **MediaPipe Hands 提取 21 个 3D 关键点**, 定位可见指甲数量和大致位置
2. **千问图像编辑 qwen-image-2.0-pro 优先生成真实试戴图**, 只重绘指甲区域, 保持手部和背景不变
3. **多模态大模型 qwen-vl-plus** 分析手部特征, 从 25 款款式库中推荐合适款式
4. **款式参考图 + 手图双图输入**, 让 AI 按真实款式生成曲面贴合、凝胶高光、甲沟阴影和透视遮挡
5. **OpenCV/SAM 本地渲染兜底**, 当图像编辑接口不可用时仍可返回试戴结果
6. 输出试戴图 + 适配度分析 + Chatbot 多轮对话 (款式咨询/搭配/换款)

**前端**: 单页 SPA, 纯 HTML/CSS/JS, 无构建依赖
**后端**: Flask + 阿里云 DashScope SDK + MediaPipe + OpenCV/SAM

## 启动方式

### 一键启动 (推荐)

```bash
cd nail-tryon
./start.sh
```

脚本会:
- 检查 Python 依赖 (缺啥装啥)
- 检查 .env (含 DASHSCOPE_API_KEY)
- 启动 Flask on :8000 (Flask 同时服务前端 + API)

### 手动启动

```bash
# 1. 装依赖
pip3 install flask flask-cors dashscope python-dotenv opencv-python-headless numpy mediapipe==0.10.14

# 2. 配 API Key
cat > backend/.env <<EOF
DASHSCOPE_API_KEY=你的key
# 可选: 关闭千问图像编辑, 仅使用本地兜底渲染
USE_QWEN_IMAGE_EDIT=1
# 可选: 图像编辑超时秒数
QWEN_IMAGE_EDIT_TIMEOUT=180
EOF

# 3. 启动
cd backend
python3 app.py
# 访问 http://localhost:8000/index.html
```

## 项目结构

```
nail-tryon/
├── backend/
│   ├── app.py                 # Flask + API / AI try-on 主逻辑
│   └── .env                   # API key (不提交)
├── frontend/
│   ├── index.html             # 单页 (选款式 → 上传 → 试戴)
│   ├── styles.css             # 衬线 + 暖色, "去 AI 感"
│   └── app.js                 # 流程编排 + Chatbot
├── assets/
│   ├── styles/                # 14 款原始款式图 (style-01..14.png)
│   ├── nail-tips/             # 14 款 × 8 位置指甲 tip 裁剪
│   ├── samples/               # 测试手图
│   └── test/                  # 本地测试输出 (不提交)
├── models/                    # SAM 权重 (不提交)
├── start.sh                   # 一键启动
└── README.md
```

## 关键 API

| 端点 | 方法 | 作用 |
|---|---|---|
| `/api/health` | GET | 健康检查 (含款式数 / 千问图像编辑开关) |
| `/api/styles` | GET | 25 款款式库 (14 原品 + 11 补充色卡) |
| `/api/recommend` | POST | AI 看手图推荐 (multipart `hand_image`) |
| `/api/tryon` | POST | 试戴主流程 (multipart `hand_image`, `style_id`; 优先 AI 图像编辑) |
| `/api/chat` | POST | Chatbot 多轮对话 (json: `message`, `style_id`, `history`) |

## 核心流程

### 1. 千问图像编辑真实试戴 (v11 主流程)
- `/api/tryon` 和 `/api/recommend_tryon` 默认优先调用 `qwen-image-2.0-pro`
- 输入顺序: 款式参考图在前, 用户手图在最后, 以保留手图输出比例
- Prompt 强约束: 只改指甲/甲床区域, 不改变手指、皮肤纹理、背景、光照和构图
- 生成目标: 曲面贴合、真实凝胶质感、透明层次、高光、边缘厚度、甲沟阴影和透视遮挡
- 返回字段 `render_engine` 可确认当前使用 `qwen-image-edit:<model>` 还是 `local-opencv`

## 本地兜底算法

### 1. 指甲定位 (v3: MediaPipe 3D 关键点)
- MediaPipe Hands 提取 21 个 3D 关键点
- 用 5 个指尖 (4/8/12/16/20) + DIP (3/7/11/15/19) 估算甲面中心 + 尺寸
- 拇指特殊处理 (指甲更大, 角度垂直)

### 2. 3D 曲率感知 Shading (v4 新增)
- **曲率计算** (0-1 范围):
  - z 偏移: `tip.z - dip.z` (MediaPipe 深度差) → 归一化
  - 关节角: tip→dip→pip 形成的角度 → 弯曲度
  - 取两者最大值
- **Shading 动态调整** (随曲率):
  - 高光宽度: 38% → 23% (越凸越窄)
  - 高光强度: 32% → 50% (越凸越亮)
  - 边缘阴影深度: +30 → +50 (越凸越深)
  - 月牙线明显度: +0 → +15%
- **效果**: 8.5/10 立体感 (vs 旧版 2/10 平面贴纸)

### 3. 像素级指甲分割 (v6: GrabCut + v7: SAM)
- **v6 GrabCut** (主备): 在指甲 ROI 内做 GrabCut + 凸包 + 连通域 + 羽化
- **v7 SAM** (预训练 ViT-B, 主用): Meta Segment Anything Model
  - 加载 358MB 预训练权重 (`models/sam_vit_b.pth`)
  - 整张图作 box 提示, 返回 评分 0.88 的 mask (覆盖率 45%)
  - 涵盖率 15-60% + 评分最高 双约束选最佳
- **v8 HSV 光照匹配**: 拿手部周围皮肤 V 通道均值, 缩放贴图 V 通道, 避免 halo

### 4. 透视变形 (4 角点)
- 根据手指方向 (tip→dip 向量) 计算 4 角点
- tip 端略窄 (95%), dip 端略宽 (105%)
- 远端小 (85%), 近端大 (115%) 模拟甲面 3D 透视

### 5. 真 PNG 指甲贴图 (v8: SAM 抠图 + HSV 融合)
- 从 `assets/nail-tips/tip-XX-{position}.png` 读取
- SAM 抠出指甲区域 (背景/皮肤移除)
- HSV 光照匹配 → 贴图颜色跟用户手部一致
- 透视变换 warp 到用户指甲位置
- alpha 混合 → 贴图自然融入
- **效果**: 7.5-8.5/10 (v8 跟 v4 合成 接近)

## 25 款说明

| 范围 | 数量 | 数据来源 |
|---|---|---|
| 1-14 | 14 款 | 原始评测数据 (有原图 + tip 资产) |
| 15-25 | 11 款 | **补充色卡** (xlsx 评估数据未提供, 颜色根据同类色系估算) |

- 15-25 款 UI 显示 `.style-card--supplement` 样式 (角标提示)
- 资产补充: 把 xlsx 里的 11 款原图放到 `assets/styles/style-15..25.png` 即可激活

## 已知限制

| 项 | 现状 | 未来改进 |
|---|---|---|
| 真实贴图 | ✅ SAM ViT-B 抠图 + HSV 融合 (7.5-8.5/10) | 增强 specular 高光 |
| 3D 曲面对准 | 假 3D (2D 椭圆 + 假 shading) | 估计指甲法线, 重建 3D mesh |
| 复杂场景 (双手交叠) | 定位失效 | 多手分离 + SAM |
| 25 款 | 11 款为补充色卡 | 等待原评估 xlsx 数据 |
| Chatbot 持久化 | 无 (刷新丢) | localStorage 上下文记忆 |

## License
