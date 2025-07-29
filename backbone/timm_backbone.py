import timm
import torch
import torch.nn as nn
import torchinfo


class resnet50():
    def __init__(self):
        self.model = timm.create_model('resnet50', pretrained=True , num_classes = 512)
        self.model = self.model.to(device='cuda:0')


class resnet101():
    def __init__(self):
        self.model = timm.create_model('resnet101', pretrained=True , num_classes = 512)
        self.model = self.model.to(device='cuda:0')

class resnet152():
    def __init__(self):
        self.model = timm.create_model('resnet152', pretrained=True , num_classes = 512)
        self.model = self.model.to(device='cuda:0')

class resnet50x():
    def __init__(self):
        self.model = timm.create_model('resnext50_32x4d', pretrained=True, features_only=True)
        self.model = self.model.to(device='cuda:0')

model = resnet50x()
model = model.model.to(device='cuda:0')


torchinfo.summary(model, input_size=(1, 3, 112, 112))


tensor = torch.randn(1, 3, 112, 112).to(device='cuda:0')

out = model(tensor)


for ot in out:
    print(ot.shape)  # Output shape should be (1, 512) for the feature vector