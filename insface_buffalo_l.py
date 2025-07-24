
from insightface.app import FaceAnalysis
import cv2
import sklearn
import sklearn.metrics
import os

import os


"""
# cuDNN 라이브러리들을 CUDA 시스템 디렉토리로 복사
sudo cp /home/ubuntu/arcface-pytorch/insight_face_package_model/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib/* /usr/local/cuda-12.2/lib64/

# cuBLAS 라이브러리들을 CUDA 시스템 디렉토리로 복사
sudo cp /home/ubuntu/arcface-pytorch/insight_face_package_model/.venv/lib/python3.10/site-packages/nvidia/cublas/lib/* /usr/local/cuda-12.2/lib64/

# 시스템 라이브러리 캐시 업데이트
sudo ldconfig

"""

# 🚀 GPU 가속을 위한 완전한 환경변수 설정
def setup_gpu_environment():
    
    # CUDA 경로 설정
    cuda_root = '/usr/local/cuda-12.2'
    cuda_bin = f'{cuda_root}/bin'
    cuda_lib = f'{cuda_root}/lib64'
    
    # 가상환경 내 NVIDIA 라이브러리 경로
    venv_nvidia_base = '/home/ubuntu/arcface-pytorch/insight_face_package_model/.venv/lib/python3.10/site-packages/nvidia'
    cudnn_lib = f'{venv_nvidia_base}/cudnn/lib'
    cublas_lib = f'{venv_nvidia_base}/cublas/lib'
    
    # PATH 환경변수 설정
    current_path = os.environ.get('PATH', '')
    if cuda_bin not in current_path:
        os.environ['PATH'] = f'{cuda_bin}:{current_path}'
    
    # LD_LIBRARY_PATH 환경변수 설정 (라이브러리 검색 경로)
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_ld_path = f'{cuda_lib}:{cudnn_lib}:{cublas_lib}:{current_ld_path}'
    os.environ['LD_LIBRARY_PATH'] = new_ld_path
    
    # CUDA 관련 추가 환경변수
    os.environ['CUDA_HOME'] = cuda_root
    os.environ['CUDA_ROOT'] = cuda_root
    os.environ['CUDA_PATH'] = cuda_root
    
    # GPU 메모리 설정 (선택적)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU 0번 사용
    
    print("🚀 GPU 환경 설정 완료")
    print(f"  ✅ CUDA Path: {cuda_lib}")
    print(f"  ✅ cuDNN Path: {cudnn_lib}")
    print(f"  ✅ cuBLAS Path: {cublas_lib}")
    print(f"  ✅ GPU Device: {os.environ['CUDA_VISIBLE_DEVICES']}")
    return True

# GPU 환경 설정 실행
setup_gpu_environment()

"""

HEAD : ArcFace 

antelopev2	ResNet-100	Glint360K	407MB	최고 성능
buffalo_l	ResNet-50	WebFace600K	326MB	높은 성능
buffalo_m	ResNet-50	WebFace600K	313MB	중간 성능
buffalo_s	MobileFaceNet	WebFace600K	159MB	빠른 속도
buffalo_sc	MobileFaceNet	WebFace600K	16MB	초경량

"""



provider_options = [
    {},  # CUDAExecutionProvider에 대한 옵션
    {'intra_op_num_threads': 0}  # CPUExecutionProvider에 대한 옵션
]

app = FaceAnalysis(name="antelopev2", 
                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                   provider_options=provider_options,
                   root=".")

image_list = os.listdir('pair/0')

image = cv2.imread(f'pair/0/{image_list[0]}')
compare = cv2.imread(f'pair/0/{image_list[5]}')

if image is None or compare is None:
    raise ValueError("이미지 파일을 찾을 수 없습니다. 경로를 확인하세요.")

app.prepare(ctx_id=0, det_size=(640, 640))
# faces = app.get(image)
# c = app.get(compare)

faces = app.get(image)
c = app.get(compare)

value = faces[0]
value2 = c[0]
embedding_vector = value['embedding']
compare_vector = value2['embedding']

embedding_vector = embedding_vector.reshape(1, -1)
compare_vector = compare_vector.reshape(1, -1)

print(embedding_vector.shape)
print('='*10)



print("유사도 : ",sklearn.metrics.pairwise.cosine_similarity(embedding_vector, compare_vector)[0][0])