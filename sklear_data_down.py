from sklearn.datasets import fetch_lfw_people
import os
import numpy as np
import cv2
from tqdm import tqdm

current_path = os.path.join(os.getcwd() , 'lfw')
min_persons = 3
lfw_people = fetch_lfw_people(data_home=f'{current_path}',min_faces_per_person=min_persons, resize=None)

n_samples, h, w = lfw_people.images.shape
n_classes = lfw_people.target_names.shape[0]

print("\n--- 데이터셋 정보 ---")
print(f"총 이미지 수 (n_samples): {n_samples}")
print(f"이미지 크기 (height x width): {h} x {w}")
print(f"총 인물 수 (n_classes): {n_classes}")
print(f"포함된 인물 리스트 (일부): {list(lfw_people.target_names[:10])}")

print("\n--- 인물별 사진 개수 검증 ---")
counts = np.bincount(lfw_people.target)

is_all_over_min = True
for person_name, count in zip(lfw_people.target_names, counts):
    if count < min_persons:
        print(f"[오류] {person_name}님은 사진이 {count}장밖에 없습니다.")
        is_all_over_min = False
    if list(lfw_people.target_names).index(person_name) < 20:
        print(f"- {person_name}: {count} 장")

if is_all_over_min:
    print(f"\n[성공] 모든 인물이 최소 {min_persons}장 이상의 사진을 가지고 있음을 확인했습니다.")


folder_path = 'lfw/lfw_home/lfw_funneled'
folder_list = [os.path.join(folder_path, folder) for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))]

save_path = './lfw_soring'

print(folder_list[:5])


for person_idx, person_path in enumerate(folder_list):
    
    image_list = os.listdir(person_path)
    
    if len(image_list) < min_persons:
        continue
    else:
        new_person_path = os.path.join(save_path, str(person_idx))
        os.makedirs(new_person_path, exist_ok=True)


        full_image_path_list = [os.path.join(person_path, f) for f in image_list]

        for img_idx, img_path in enumerate(full_image_path_list):
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)

            if image is not None:
                dest_file_path = os.path.join(new_person_path, f"{img_idx}.jpg")
                cv2.imwrite(dest_file_path, image)

print(f"작업 완료. 결과가 '{save_path}' 폴더에 저장되었습니다.")
    
