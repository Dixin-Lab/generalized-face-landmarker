import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from Landmark2.utils import NestedTensor, is_main_process


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [4, 8, 16, 32]
            # 中间特征图的channel数量
            self.num_channels = [256, 512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: torch.tensor):
        xs = self.body(tensor_list)
        out: Dict[str, torch.tensor] = {}
        for name, x in xs.items():
            out[name] = x
        return out


# Backbone
class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        # norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


class feature_fusion(nn.Module):
    def __init__(self, dim_1, dim_2, target_dim):
        super(feature_fusion, self).__init__()
        self.Conv1_down_1 = conv_1x1_bn(dim_1, target_dim//2)
        self.Conv1_down_2 = conv_bn(target_dim//2, target_dim//2, 1)
        self.Conv2_down = conv_1x1_bn(dim_2, target_dim//2)
        self.feature_mix = conv_bn(target_dim, target_dim, 1)

    def forward(self, S_feature, L_feature):
        S_feature = self.Conv1_down_1(S_feature)
        S_feature = F.interpolate(S_feature, scale_factor=2.0, mode='bilinear', align_corners=True)
        S_feature = self.Conv1_down_2(S_feature)
        L_feature = self.Conv2_down(L_feature)
        O_feature = torch.cat((S_feature, L_feature), dim=1)
        O_feature = self.feature_mix(O_feature)

        return O_feature

if __name__ == "__main__":
    from utils import nested_tensor_from_tensor_list
    Resnet50 = Backbone(name='resnet50', train_backbone=True, return_interm_layers=True, dilation=False)
    input = torch.rand(2, 3, 256, 256)
    output = Resnet50(input)
    print(output['0'].size())
    feature_s = torch.rand(2, 2048, 8, 8)
    feature_l = torch.rand(2, 1024, 16, 16)
    fusion = feature_fusion(2048, 1024, 512)
    out = fusion(feature_s, feature_l)
    print(out.size())