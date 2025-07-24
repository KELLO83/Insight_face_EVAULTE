import os
import itertools
import random
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
import cv2
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix
import logging
import traceback
import pickle
import argparse
import matplotlib.pyplot as plt

# --- 로깅 및 경로 설정 ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
LOG_FILE = os.path.join(script_dir, "LOG.txt")

logging.basicConfig(
    filename=LOG_FILE, level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s', filemode='w'
)

model_info = {
        "antelopev2": "ResNet-100 + Glint360K (407MB)",
        "buffalo_l": "ResNet-50 + WebFace600K (326MB)", 
        "buffalo_m": "ResNet-50 + WebFace600K (313MB)",
        "buffalo_s": "MobileFaceNet + WebFace600K (159MB)",
        "buffalo_sc": "MobileFaceNet + WebFace600K (16MB)",
    }

def get_all_embeddings_insightface(identity_map, face_app, model_name, dataset_name, use_cache=True):
    """InsightFace를 사용한 임베딩 추출 (normed_embedding 사용)"""
    cache_file = os.path.join(script_dir, f"embeddings_cache_{dataset_name}_{model_name}_insightface.pkl")
    
    if use_cache and os.path.exists(cache_file):
        print(f"\n캐시 파일 '{cache_file}'에서 임베딩을 로드합니다...")
        with open(cache_file, 'rb') as f:
            embeddings = pickle.load(f)
        print("임베딩 로드 완료.")
        return embeddings

    all_images = sorted(list(set(itertools.chain.from_iterable(identity_map.values()))))
    print(f"\n총 {len(all_images)}개의 이미지에 대해 임베딩을 새로 추출합니다...")
    
    embeddings = {}
    
    for img_path in tqdm(all_images, desc="임베딩 추출"):
        try:
            image = cv2.imread(img_path)
            if image is None:
                logging.warning(f"이미지 로드 실패: {img_path}")
                embeddings[img_path] = None
                continue
            
            image_rgb = image[:, :, ::-1]
            faces = face_app.get(image_rgb)
            
            if len(faces) == 0:
                logging.warning(f"얼굴을 찾을 수 없음: {img_path}")
                embeddings[img_path] = None
            else:
                embeddings[img_path] = faces[0].normed_embedding
                
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
    """임베딩으로 유사도를 계산합니다 (np.dot 사용)."""
    similarities, labels = [], []
    label = 1 if is_positive else 0
    desc = "동일 인물 쌍 계산" if is_positive else "다른 인물 쌍 계산"
    for img1_path, img2_path in tqdm(pairs, desc=desc):
        emb1, emb2 = embeddings.get(img1_path), embeddings.get(img2_path)
        if emb1 is not None and emb2 is not None:
            similarities.append(np.dot(emb1, emb2))
            labels.append(label)
            
    return similarities, labels

def save_results_to_excel(excel_path, model_name, roc_auc, eer, tar_at_far_results, target_fars, metrics, total_dataset_img_len, total_class,
                           data_path, model_attr_value):
    """결과를 Excel 파일에 저장합니다."""
    new_data = {
        "model_name": [model_name],
        "roc_auc": [f"{roc_auc:.4f}"], "eer": [f"{eer:.4f}"],
        "accuracy": [f"{metrics['accuracy']:.4f}"], "recall": [f"{metrics['recall']:.4f}"],
        "f1_score": [f"{metrics['f1_score']:.4f}"], "tp": [metrics['tp']],
        "tn": [metrics['tn']], "fp": [metrics['fp']], "fn": [metrics['fn']]
    }

    for far in target_fars:
        new_data[f"tar_at_far_{far*100:g}%"] = [f"{tar_at_far_results.get(far, 0):.4f}"]

    new_data.update({
        'total_dataset_img_len': [total_dataset_img_len],
        'total_class': [total_class],
        'data_path': [data_path],
        'model_attr': [model_attr_value]
    })
    
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

    modeL_value = model_info.get(model_name, "정보 없음")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (TAR)')
    plt.title(f'ROC Curve {model_name} {modeL_value}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plot_filename = os.path.splitext(excel_path)[0] + f"_{model_name}_roc_curve.png"
    plt.savefig(plot_filename)
    print(f"ROC 커브 그래프가 '{plot_filename}' 파일로 저장되었습니다.")

def main(args):
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
                if sorted_pair not in negative_pairs_set:
                    negative_pairs_set.add(sorted_pair)
                    pbar.update(1)
    negative_pairs = list(negative_pairs_set)

    print(f"- 동일 인물 쌍: {len(positive_pairs)}개, 다른 인물 쌍: {len(negative_pairs)}개")
    print(f"\{args.model_name} 모델을 로드합니다...")

    provider_options = [
        {},  # CUDAExecutionProvider에 대한 옵션
        {'intra_op_num_threads': 0}  # CPUExecutionProvider에 대한 옵션
    ]
        
    face_app = FaceAnalysis(
        name=args.model_name,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        root=".",
        provider_options=provider_options
    )

    face_app.prepare(ctx_id=0, det_size=tuple(args.det_size))
    print("모델이 성공적으로 로드되었습니다.")

    dataset_name = os.path.basename(os.path.normpath(args.data_path))
    embeddings = get_all_embeddings_insightface(
        identity_map, face_app, args.model_name, dataset_name, 
        use_cache=not args.no_cache
    )

    pos_similarities, pos_labels = collect_scores_from_embeddings(positive_pairs, embeddings, is_positive=True)
    neg_similarities, neg_labels = collect_scores_from_embeddings(negative_pairs, embeddings, is_positive=False)
    
    scores = np.array(pos_similarities + neg_similarities)
    labels = np.array(pos_labels + neg_labels)

    print("\n--- 최종 평가 결과 ---")
    if labels.size > 0:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        frr = 1 - tpr
        eer_index = np.nanargmin(np.abs(fpr - frr))
        eer = fpr[eer_index]
        eer_threshold = thresholds[eer_index]

        tar_at_far_results = {far: np.interp(far, fpr, tpr) for far in args.target_fars}
        
        predictions = (scores >= eer_threshold).astype(int)
        cm = confusion_matrix(labels, predictions)
        
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        metrics = {"accuracy": accuracy, "recall": recall, "f1_score": f1_score, "tp": tp, "tn": tn, "fp": fp, "fn": fn}

        print(f"사용된 모델: InsightFace {args.model_name}, 전체 평가 쌍: {len(labels)} 개")
        print(f"[주요 성능] ROC-AUC: {roc_auc:.4f}, EER: {eer:.4f} (유사도 임계값: {eer_threshold:.4f})")
        print(f"[상세 지표] Accuracy: {metrics['accuracy']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
        for far, tar in tar_at_far_results.items():
            print(f"  - TAR @ FAR {far*100:g}%: {tar:.4f}")
        
        with open(LOG_FILE, 'a') as log_file:
            log_file.write(f"\nInsightFace {args.model_name} 평가 결과:\n")
            log_file.write(f"ROC-AUC: {roc_auc:.4f}, EER: {eer:.4f} (Threshold: {eer_threshold:.4f})\n")
            log_file.write(f"Accuracy: {metrics['accuracy']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1_score']:.4f}\n")
            for far, tar in tar_at_far_results.items():
                log_file.write(f"TAR @ FAR {far*100:g}%: {tar:.4f}\n")

        excel_path = os.path.join(script_dir, args.excel_path)
        total_dataset_img_len = sum(len(v) for v in identity_map.values())
        total_class = len(identity_map)
        model_attr_value = model_info.get(args.model_name, "정보 없음")
        save_results_to_excel(excel_path, f"InsightFace_{args.model_name}", roc_auc, eer, tar_at_far_results, \
                              args.target_fars, metrics, total_dataset_img_len, total_class, args.data_path, model_attr_value)
        
        
        if args.plot_roc:
            plot_roc_curve(fpr, tpr, roc_auc, args.model_name, excel_path)
    else:
        msg = "평가를 위한 유효한 점수를 수집하지 못했습니다."
        print(msg)
        logging.error(msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-Process InsightFace Evaluation Script")
    parser.add_argument("--data_path", type=str, default="/home/ubuntu/arcface-pytorch/insight_face_package_model/pair", help="평가할 데이터셋의 루트 폴더")
    parser.add_argument("--model_name", type=str, default="antelopev2", 
                       choices=["antelopev2", "buffalo_l", "buffalo_m", "buffalo_s", "buffalo_sc","auraface"],
                       help="사용할 InsightFace 모델")
    parser.add_argument("--excel_path", type=str, default="insightface_evaluation_results.xlsx", help="결과를 저장할 Excel 파일 이름")
    parser.add_argument("--target_fars", nargs='+', type=float, default=[0.01, 0.001, 0.0001], help="TAR을 계산할 FAR 목표값들")
    parser.add_argument("--det_size", nargs=2, type=int, default=[640, 640], help="얼굴 검출 크기 (width, height)")
    parser.add_argument("--no-cache", help="이 플래그를 사용하면 기존 임베딩 캐시를 무시하고 새로 추출합니다." , default=False) 
    parser.add_argument("--plot-roc", action="store_true", help="이 플래그를 사용하면 ROC 커브 그래프를 파일로 저장합니다.", default=True)
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        error_message = traceback.format_exc()
        logging.error(f"스크립트 실행 중 처리되지 않은 예외 발생:\n{error_message}")
        print(f"\n치명적인 오류가 발생했습니다. '{LOG_FILE}' 파일에 상세 내역이 기록되었습니다.")