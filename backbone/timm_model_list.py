import timm
for model in timm.list_models('*resnet*',pretrained=True):
    if 'resnet' in model:
        print(model)
