#!/bin/bash

PROJECT_DIR="$(pwd)"

echo "🚀 InsightFace Docker 컨테이너 실행"
echo "현재 디렉토리: $PROJECT_DIR"

# 필수 파일들이 있는지 확인
if [ ! -f "single_insightface.py" ]; then
    echo "❌ 오류: single_insightface.py 파일을 찾을 수 없습니다."
    echo "📍 이 스크립트는 프로젝트 디렉토리에서 실행해야 합니다:"
    echo "   cd /home/ubuntu/arcface-pytorch/insight_face_package_model"
    echo "   ./run_docker.sh"
    exit 1
fi

if [ ! -d "models" ]; then
    echo "❌ 오류: models 디렉토리를 찾을 수 없습니다."
    echo "📍 이 스크립트는 프로젝트 디렉토리에서 실행해야 합니다."
    exit 1
fi

if [ ! -d "pair" ]; then
    echo "❌ 오류: pair 디렉토리를 찾을 수 없습니다."
    echo "📍 이 스크립트는 프로젝트 디렉토리에서 실행해야 합니다."
    exit 1
fi

MODEL_NAME=${1:-auraface}
DATA_PATH=${2:-./pair}

echo "모델: $MODEL_NAME"
echo "데이터: $DATA_PATH"

sudo docker run --gpus all --rm \
  -v "$PROJECT_DIR":/app/workspace \
  -u $(id -u):$(id -g) \
  insightface-gpu:latest \
  bash -c "cd /app/workspace && python single_insightface.py --model_name $MODEL_NAME --data_path $DATA_PATH" 2>&1 | tee output.log


if [ -f "insightface_evaluation_results.xlsx" ]; then
    sudo chown $USER:$USER ./*.xlsx ./*.txt ./*.png 2>/dev/null || true
fi

echo "✅ 실행 완료. 로그: output.log"
