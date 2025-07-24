import os
import itertools
import random
import numpy as np
import pandas as pd
from deepface import DeepFace
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix
import logging
import traceback
import pickle
import argparse
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt


try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
LOG_FILE = os.path.join(script_dir, "single_deepface.txt")

logging.basicConfig(
    filename=LOG_FILE, level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s', filemode='w'
)

def get_all_embeddings(identity_map, model, detector_backend, dataset_name, head ='ArcFace' , use_cache=True, batch_size=10240):
    """임베딩을 추출하거나 캐시에서 로드 (배치 처리 기능 추가)"""
    cache_file = os.path.join(script_dir, f"embeddings_cache_{dataset_name}_{head}_single_batch.pkl")
    
    if use_cache and os.path.exists(cache_file):
        print(f"\n캐시 파일 '{cache_file}'에서 임베딩을 로드합니다...")
        with open(cache_file, 'rb') as f:
            embeddings = pickle.load(f)
        print("임베딩 로드 완료.")
        return embeddings

    all_images = sorted(list(set(itertools.chain.from_iterable(identity_map.values()))))
    print(f"\n총 {len(all_images)}개의 이미지에 대해 임베딩을 새로 추출합니다 (배치 크기: {batch_size})...")
    
    embeddings = {}
    
    for i in tqdm(range(0, len(all_images), batch_size), desc="임베딩 추출"):
        batch_paths = all_images[i:i+batch_size]
        
        # DeepFace.represent()는 배치를 지원하지 않으므로 개별 처리
        for img_path in batch_paths:
            try:
                embedding_obj = DeepFace.represent(
                    img_path=img_path, 
                    model_name=head,
                    detector_backend=detector_backend, 
                    enforce_detection=False,
                    normalization='ArcFace',
                    align=True,
                )
                embeddings[img_path] = embedding_obj[0]['embedding']
            except Exception as e:
                logging.warning(f"임베딩 추출 오류: {img_path}. 제외됩니다. 오류: {e}")
                embeddings[img_path] = None

    if use_cache:
        print(f"\n추출된 임베딩을 캐시 파일 '{cache_file}'에 저장합니다...")
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print("캐시 저장 완료.")
    return embeddings

def collect_scores_from_embeddings(pairs, embeddings, is_positive):
    distances, labels = [], []
    label = 1 if is_positive else 0
    desc = "동일 인물 쌍 계산" if is_positive else "다른 인물 쌍 계산"
    for img1_path, img2_path in tqdm(pairs, desc=desc):
        emb1, emb2 = embeddings.get(img1_path), embeddings.get(img2_path)
        if emb1 is not None and emb2 is not None:
            distances.append(cosine(emb1, emb2))
            labels.append(label)
    return distances, labels

def save_results_to_excel(excel_path, model_name, detector_backend, roc_auc, eer, tar_at_far_results, target_fars, metrics):
    """결과를 Excel 파일에 저장합니다."""
    new_data = {
        "model_name": [model_name], "detector_backend": [detector_backend],
        "roc_auc": [f"{roc_auc:.4f}"], "eer": [f"{eer:.4f}"],
        "accuracy": [f"{metrics['accuracy']:.4f}"], "recall": [f"{metrics['recall']:.4f}"],
        "f1_score": [f"{metrics['f1_score']:.4f}"], "tp": [metrics['tp']],
        "tn": [metrics['tn']], "fp": [metrics['fp']], "fn": [metrics['fn']]
    }
    for far in target_fars:
        new_data[f"tar_at_far_{far*100:g}%"] = [f"{tar_at_far_results.get(far, 0):.4f}"]
    
    new_df = pd.DataFrame(new_data)
    try:
        df = pd.read_excel(excel_path)
    except FileNotFoundError:
        df = pd.DataFrame()

    updated_df = pd.concat([df, new_df], ignore_index=True)
    updated_df.to_excel(excel_path, index=False)
    print(f"\n평가 결과가 '{excel_path}' 파일에 저장되었습니다.")

def plot_roc_curve(fpr, tpr, roc_auc, model_name, excel_path):
    """ROC 커브를 그리고 파일로 저장합니다."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (TAR)')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plot_filename = os.path.splitext(excel_path)[0] + "_roc_curve.png"
    plt.savefig(plot_filename)
    print(f"ROC 커브 그래프가 '{plot_filename}' 파일로 저장되었습니다.")

def main(args):
    # --- 1단계: 데이터셋 스캔 ---
    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(f"데이터셋 경로를 찾을 수 없습니다: {args.data_path}")

    identity_map = {}
    for person_folder in os.listdir(args.data_path):
        person_path = os.path.join(args.data_path, person_folder)
        if os.path.isdir(person_path):
            images = [os.path.join(person_path, f) for f in os.listdir(person_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(images) > 1:
                identity_map[person_folder] = images
    
    if not identity_map:
        raise ValueError("데이터셋에서 2개 이상의 이미지를 가진 인물을 찾지 못했습니다.")
    print(f"총 {len(identity_map)}명의 인물, {sum(len(v) for v in identity_map.values())}개의 이미지를 찾았습니다.")

    # --- 2단계: 평가 쌍 생성 ---
    print("\n평가에 사용할 동일 인물/다른 인물 쌍을 생성합니다...")
    
    positive_pairs = []
    for imgs in tqdm(identity_map.values(), desc="동일 인물 쌍 생성"):
        positive_pairs.extend(itertools.combinations(imgs, 2))

    num_positive_pairs = len(positive_pairs)
    
    identities = list(identity_map.keys())
    negative_pairs_set = set()
    if len(identities) > 1:
        with tqdm(total=num_positive_pairs, desc="다른 인물 쌍 생성") as pbar:
            while len(negative_pairs_set) < num_positive_pairs:
                id1, id2 = random.sample(identities, 2)
                pair = (random.choice(identity_map[id1]), random.choice(identity_map[id2]))
                sorted_pair = tuple(sorted(pair))
                negative_pairs_set.add(sorted_pair)
                pbar.update(len(negative_pairs_set) - pbar.n)  # 현재 세트 크기만큼 진행 상황 업데이트
    negative_pairs = list(negative_pairs_set)

    print(f"- 동일 인물 쌍: {len(positive_pairs)}개, 다른 인물 쌍: {len(negative_pairs)}개")

    # --- 3단계: 모델 빌드 ---
    print(f"\n모델({args.model_name})을 빌드하고 GPU에 로드합니다...")
    model = DeepFace.build_model(args.model_name)
    print("모델이 성공적으로 빌드되었습니다.")

    # --- 4단계: 임베딩 추출 또는 캐시 로드 ---
    dataset_name = os.path.basename(os.path.normpath(args.data_path))
    embeddings = get_all_embeddings(identity_map, model, args.detector_backend, dataset_name, head=args.model_name, use_cache=not args.no_cache, batch_size=args.batch_size)

    # --- 5단계: 점수 수집 ---
    print("\n미리 계산된 임베딩으로 거리를 계산합니다...")
    pos_distances, pos_labels = collect_scores_from_embeddings(positive_pairs, embeddings, is_positive=True)
    neg_distances, neg_labels = collect_scores_from_embeddings(negative_pairs, embeddings, is_positive=False)
    distances = np.array(pos_distances + neg_distances)
    labels = np.array(pos_labels + neg_labels) 

    print("\n--- 최종 평가 결과 ---")
    if labels.size > 0:
        scores = -distances
        fpr, tpr, thresholds = roc_curve(labels, scores) # Roc 커브 계산
        roc_auc = auc(fpr, tpr)
        frr = 1 - tpr
        eer_index = np.nanargmin(np.abs(fpr - frr))
        eer = fpr[eer_index]
        eer_threshold = -thresholds[eer_index]

        tar_at_far_results = {far: np.interp(far, fpr, tpr) for far in args.target_fars}
        
        predictions = (distances < eer_threshold).astype(int)
        cm = confusion_matrix(labels, predictions)
        if cm.size == 1:
             if labels[0] == 0: tn, fp, fn, tp = cm[0][0], 0, 0, 0
             else: tn, fp, fn, tp = 0, 0, 0, cm[0][0]
        else:
            tn, fp, fn, tp = cm.ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        metrics = {"accuracy": accuracy, "recall": recall, "f1_score": f1_score, "tp": tp, "tn": tn, "fp": fp, "fn": fn}

        print(f"사용된 모델: {args.model_name}, 전체 평가 쌍: {len(labels)} 개")
        print(f"[주요 성능] ROC-AUC: {roc_auc:.4f}, EER: {eer:.4f} (거리 임계값: {eer_threshold:.4f})")
        print(f"[상세 지표] Accuracy: {metrics['accuracy']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
        for far, tar in tar_at_far_results.items():
            print(f"  - TAR @ FAR {far*100:g}%: {tar:.4f}")

        excel_path = os.path.join(script_dir, args.excel_path)
        save_results_to_excel(excel_path, args.model_name, args.detector_backend, roc_auc, eer, tar_at_far_results, args.target_fars, metrics)
        
        if args.plot_roc:
            plot_roc_curve(fpr, tpr, roc_auc, args.model_name, excel_path)
    else:
        msg = "평가를 위한 유효한 점수를 수집하지 못했습니다."
        print(msg)
        logging.error(msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-Process Face Recognition Evaluation Script")
    parser.add_argument("--data_path", type=str, default="/home/ubuntu/Face-Recognition/deep_face/dataset/ms1m-arcface", help="평가할 데이터셋의 루트 폴더")
    parser.add_argument("--model_name", type=str, default="ArcFace", help="사용할 얼굴 인식 모델 (e.g., VGG-Face, Facenet, ArcFace)")
    parser.add_argument("--detector_backend", type=str, default="retinaface", help="사용할 얼굴 탐지 백엔드")
    parser.add_argument("--excel_path", type=str, default="single_evaluation_results.xlsx", help="결과를 저장할 Excel 파일 이름")
    parser.add_argument("--target_fars", nargs='+', type=float, default=[0.01, 0.001, 0.0001], help="TAR을 계산할 FAR 목표값들")
    parser.add_argument("--batch_size", type=int, default=10240, help="임베딩 추출 시 사용할 배치 크기")
    parser.add_argument("--no-cache", action="store_true", help="이 플래그를 사용하면 기존 임베딩 캐시를 무시하고 새로 추출합니다.")
    parser.add_argument("--plot-roc", action="store_true", help="이 플래그를 사용하면 ROC 커브 그래프를 파일로 저장합니다.", default=True)
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        error_message = traceback.format_exc()
        logging.error(f"스크립트 실행 중 처리되지 않은 예외 발생:\n{error_message}")
        print(f"\n치명적인 오류가 발생했습니다. '{LOG_FILE}' 파일에 상세 내역이 기록되었습니다.")
