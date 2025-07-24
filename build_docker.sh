#!/bin/bash

echo "🚀 InsightFace GPU 도커 이미지 빌드 시작..."
echo "================================================"
echo "📋 빌드 환경 (로컬 .venv와 동일):"
echo "   - Base Image: pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel"
echo "   - Python: 3.10.12"
echo "   - PyTorch: 2.7.1+cu126"
echo "   - CUDA: 12.6"
echo "   - cuDNN: 9.x"
echo "   - ONNX Runtime GPU: 1.22.0"
echo "   - InsightFace: 0.7.3"
echo ""

# 도커 이미지 빌드
docker build -t insightface-gpu:cuda12.6-py310 .

if [ $? -eq 0 ]; then
    echo "✅ 도커 이미지 빌드 완료!"
    echo ""
    echo "📋 사용법:"
    echo "1. 단일 실행:"
    echo "   docker run --gpus all -v \$(pwd)/models:/app/models -v \$(pwd)/man:/app/man insightface-gpu:cuda12.6-py310"
    echo ""
    echo "2. 대화형 모드:"
    echo "   docker run --gpus all -it -v \$(pwd)/models:/app/models -v \$(pwd)/man:/app/man insightface-gpu:cuda12.6-py310 bash"
    echo ""
    echo "3. 데이터셋 평가:"
    echo "   docker run --gpus all -v \$(pwd)/models:/app/models -v \$(pwd)/pair:/app/pair insightface-gpu:cuda12.6-py310 python single_insightface.py --model_name antelopev2"
    echo ""
    echo "4. Docker Compose 사용:"
    echo "   docker-compose up"
    echo ""
    echo "🔄 이미지 정보:"
    docker images insightface-gpu:cuda12.6-py310
    echo ""
    echo "🧪 환경 검증 명령어:"
    echo "docker run --gpus all --rm insightface-gpu:cuda12.6-py310 python -c \\"
    echo "  import torch, onnxruntime as ort; \\"
    echo "  print(f'PyTorch: {torch.__version__}'); \\"
    echo "  print(f'CUDA available: {torch.cuda.is_available()}'); \\"
    echo "  print(f'CUDA version: {torch.version.cuda}'); \\"
    echo "  print(f'ONNX Runtime providers: {ort.get_available_providers()}'); \\"
    echo "  print(f'GPU count: {torch.cuda.device_count()}')\\"
else
    echo "❌ 도커 이미지 빌드 실패!"
    echo ""
    echo "🔍 문제 해결:"
    echo "1. NVIDIA Container Toolkit 설치 확인: docker run --rm --gpus all nvidia/cuda:12.6-base-ubuntu22.04 nvidia-smi"
    echo "2. Docker 그룹 권한 확인: groups \$USER"
    echo "3. 로그 확인: docker logs <container_id>"
    exit 1
fi
