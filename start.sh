#!/bin/bash
# AI 美甲试戴 一键启动脚本
# 用法: ./start.sh

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 端口
BACKEND_PORT=8000

# 检查 Python
if ! command -v python3 >/dev/null 2>&1; then
  echo "❌ python3 未安装"
  exit 1
fi

# 检查后端 .env
if [ ! -f backend/.env ]; then
  echo "❌ backend/.env 不存在, 需要 DASHSCOPE_API_KEY"
  exit 1
fi

# 检查后端依赖
if ! python3 -c "import flask, mediapipe, cv2, dashscope" 2>/dev/null; then
  echo "📦 安装后端依赖..."
  pip3 install flask flask-cors dashscope python-dotenv opencv-python-headless numpy mediapipe==0.10.14
fi

# 检查端口占用
if lsof -i :$BACKEND_PORT >/dev/null 2>&1; then
  echo "⚠️  端口 $BACKEND_PORT 已被占用, 关闭旧进程"
  lsof -ti :$BACKEND_PORT | xargs kill -9 2>/dev/null || true
  sleep 1
fi

# 启动后端 (Flask 统一服务前端 + API, 不需要单独前端 server)
echo "🚀 启动后端: http://localhost:$BACKEND_PORT"
cd backend
python3 app.py > /tmp/nail-backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# 健康检查
STARTED=0
for i in $(seq 1 20); do
  if curl -s http://localhost:$BACKEND_PORT/api/health | grep -q '"ok":true'; then
    STARTED=1
    break
  fi
  sleep 1
done

if [ "$STARTED" = "1" ]; then
  echo "✅ 后端启动成功"
  echo ""
  echo "📍 访问: http://localhost:$BACKEND_PORT/index.html"
  echo "📍 API:  http://localhost:$BACKEND_PORT/api/health"
  echo "📍 日志: tail -f /tmp/nail-backend.log"
  echo "📍 停止: kill $BACKEND_PID"
else
  echo "❌ 后端启动失败, 检查日志: tail -f /tmp/nail-backend.log"
  kill $BACKEND_PID 2>/dev/null || true
  exit 1
fi
