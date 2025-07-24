FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV HOME=/tmp

RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxrender1 \
    libgomp1 \
    libgl1-mesa-glx \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /tmp/matplotlib && chmod 777 /tmp/matplotlib

RUN python -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

COPY requirements_docker.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt

COPY . .


RUN echo '#!/bin/bash\nexec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 8000

# 엔트리포인트 설정
ENTRYPOINT ["/entrypoint.sh"]

# 기본 명령어
CMD ["python3", "single_insightface.py", "--help"]
