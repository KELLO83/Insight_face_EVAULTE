import cv2
import numpy as np
import face_alignment
from skimage import transform as trans
import torch
import matplotlib.pyplot as plt
import natsort
import os
import glob
import tqdm


""" 얼굴 정렬 랜드마크 추출 데이터 정제 """

class FaceAligner:

    def __init__(self):
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, 
            flip_input=False,
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            face_detector='sfd'
        )

        print("device" , self.fa.device)


        self.src_landmarks = np.array([
            [30.2946, 51.6963],  # 왼쪽 눈
            [65.5318, 51.5014],  # 오른쪽 눈
            [48.0252, 71.7366],  # 코 끝
            [33.5493, 92.3655],  # 왼쪽 입꼬리
            [62.7299, 92.2041]   # 오른쪽 입꼬리
        ], dtype=np.float32)
    


    def get_landmarks(self, image):

        try:
            preds = self.fa.get_landmarks(image)
            if preds is None or len(preds) == 0:
                return None
            return preds[0] 
         
        except:
            return None
    
    def align_face(self, image, landmarks=None):
        if landmarks is None:
            landmarks = self.get_landmarks(image)
            
        if landmarks is None:
            return None
        
        dst_landmarks = np.array([
            landmarks[36:42].mean(axis=0),  # 왼쪽 눈 중심
            landmarks[42:48].mean(axis=0),  # 오른쪽 눈 중심
            landmarks[30],                   # 코 끝
            landmarks[48],                   # 왼쪽 입꼬리
            landmarks[54]                    # 오른쪽 입꼬리
        ], dtype=np.float32)
        

        tform = trans.SimilarityTransform()
        tform.estimate(dst_landmarks, self.src_landmarks)

        aligned_face = trans.warp(
            image, 
            tform.inverse, 
            output_shape=(112, 112),
            preserve_range=True
        ).astype(np.uint8)
        
        return aligned_face
    
    def align_face_cv2(self, image, landmarks=None):
        if landmarks is None:
            landmarks = self.get_landmarks(image)
            
        if landmarks is None:
            return None

        dst_landmarks = np.array([
            landmarks[36:42].mean(axis=0),  # 왼쪽 눈 중심
            landmarks[42:48].mean(axis=0),  # 오른쪽 눈 중심
            landmarks[30],                   # 코 끝
            landmarks[48],                   # 왼쪽 입꼬리
            landmarks[54]                    # 오른쪽 입꼬리
        ], dtype=np.float32)
        

        M = cv2.getAffineTransform(dst_landmarks[:3], self.src_landmarks[:3])
        aligned_face = cv2.warpAffine(image, M, (112, 112))
        
        return aligned_face

def test_face_alignment():

    target_dir = 'lfw_sorting'
    save_dir = os.path.join(f'{target_dir}', 'aligned_faces')
    os.makedirs(save_dir, exist_ok=True)

    image_folder = os.path.join(f'{target_dir}','**',  '*.jpg')
    image_files = natsort.natsorted(glob.glob(image_folder))

    print(f"Found {len(image_files)} images in the folder.")
    aligner = FaceAligner()

    for image_path in tqdm.tqdm(image_files):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            continue           

        aligned_face = aligner.align_face(image)
        
        if aligned_face is None:
            print(f"Failed to align face for: {image_path}")
            continue


        relative_path_for_saving = '/'.join(image_path.split('/')[-2:])
        save_path = os.path.join(save_dir, relative_path_for_saving)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, aligned_face)


if __name__ == '__main__':
    test_face_alignment()