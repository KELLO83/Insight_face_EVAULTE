import torch
from backbone.ir_ASIS_Resnet import Backbone
from backbone.irsnet import IResNet , IBasicBlock
import torchinfo
backbone = IResNet(IBasicBlock , [3,13,30,3])
# backbone = Backbone(
#     input_size=(112,112,3),
#     num_layers=50,
# )
backbone.eval()



# load_result = backbone.load_state_dict(weight , strict = False)

# print("누락된 가중치 : {}".format(load_result.missing_keys))
# print("예상치못한 가중치 : {}".format(load_result.unexpected_keys))
# print(load_result)
# print("="*30)


# dummy_input = torch.randn(100, 3, 112, 112)
model_info = torchinfo.summary(
    backbone,
    input_size=(100, 3, 112, 112),
    verbose=False,
    col_names=["input_size", "output_size", "num_params", "trainable","params_percent"],
    row_settings=["depth"],
    device='cuda:1',
    mode='eval'
)
print(model_info)
# backbone.eval()
# backbone = backbone.to('cuda:1')
# dummy_input = dummy_input.to(torch.float32).to('cuda:1')
# output  = backbone(dummy_input)


