import torch
import torchvision



# celebA


# transform = transforms.Compose([
#     transforms.CenterCrop(178),         # aligned 이미지는 178×218
#     transforms.Resize((112, 112)),      # 원하는 입력 크기
#     transforms.ToTensor(),
#     transforms.Normalize([0.5]*3, [0.5]*3)
# ])

data = torchvision.datasets.CelebA(split='test', download=True , target_type='idenetity',root='data/celebA', transform=None)

