from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis
import numpy as np
import cv2

snapshot_download(
    "fal/AuraFace-v1",
    local_dir="models/auraface",
)
face_app = FaceAnalysis(
    name="auraface",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    root=".",
)

face_app.prepare(ctx_id=0, det_size=(640, 640))

input_image = cv2.imread("man/1.jpg")
compare_image = cv2.imread('man/2.jpg')

if input_image is None or compare_image is None:
    raise FileNotFoundError()

cv2_image = input_image.copy()
cv2_image = cv2_image[:, :, ::-1]
faces = face_app.get(cv2_image)
embedding = faces[0].normed_embedding


compare_image = compare_image[: ,: , ::-1]
cf = face_app.get(compare_image)
embedding_compare = cf[0].normed_embedding


similarity = np.dot(embedding , embedding_compare)


print("Cosine Similarity:", similarity)


