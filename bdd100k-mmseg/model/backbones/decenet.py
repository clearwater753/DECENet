# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
# from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
# from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmseg.registry import MODELS

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_gelu = BNGeLU(nOut)
            
        self.dropout = nn.Dropout2d(p=0.001)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_gelu(output)
            
        output = self.dropout(output) 

        return output

class BNGeLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        
        self.acti = nn.GELU()

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output

class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_gelu = BNGeLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool],1)

        output = self.bn_gelu(output)

        return output

class AFCModule(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNGeLU(nIn)
        self.conv3x3 = Conv(nIn, nIn // 2, kSize, 1, padding=1, bn_acti=True)

        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1,
                             padding=(1, 0), groups=nIn // 2, bn_acti=True)
        
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1,
                              padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1,
                              padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)

        self.bn_relu_2 = BNGeLU(nIn // 2)
        
        self.conv3x3_2 = Conv(nIn // 2, nIn, 3, 1, padding=1, bn_acti=False)
        
        self.ca11 = eca_layer(nIn // 2)
        
        self.ca12 = eca_layer(nIn // 2)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv3x3(output)

        br1 = self.dconv3x1(output)
        # br1 = self.dconv1x3(br1)
        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)
        br1 = self.ca11(br1)
        br2 = self.ca12(br2)

        output = br1 + br2
        output = self.bn_relu_2(output)
        #output = self.conv3x3_2(output)
        #output = self.conv1x1(output)
        output = self.conv3x3_2(output)

        return output + input

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.b_n = nn.BatchNorm2d(1, eps=1e-3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.b_n(x)
        return self.sigmoid(x)

@MODELS.register_module()
class DECENet(nn.Module):
    
    def __init__(self, block_1=3, block_2=3, block_3 = 5):
        super().__init__()
        #print("using DECENet")

        # ---------- Encoder -------------#
        self.init_conv = nn.Sequential(
            Conv(3, 16, 3, 2, padding=1, bn_acti=True),
            Conv(16, 16, 3, 1, padding=1, bn_acti=True),
            Conv(16, 16, 3, 1, padding=1, bn_acti=True),
        )
        #
        self.bn_gelu_1 = BNGeLU(16)

        # Branch 1
        # Attention 1
        self.attention1_1 = eca_layer(16)

        # BRU Block 1
        dilation_block_1 = [2, 2, 2,]

        self.BRU_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            
            self.BRU_Block_1.add_module("AFC_Module_1_" + str(i), AFCModule(16, d=dilation_block_1[i]))
        self.bn_gelu_2 = BNGeLU(16)
        # Attention 2
        self.attention2_1 = eca_layer(16)

        # Down 1
        self.downsample_1 = DownSamplingBlock(16, 64)
        # BRU Block 2

        dilation_block_2 = [2, 2, 2]
        self.BRU_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            
            self.BRU_Block_2.add_module("AFC_Module_2_" + str(i), AFCModule(64, d=dilation_block_2[i]))
        self.bn_gelu_3 = BNGeLU(64)
        # Attention 3
        self.attention3_1 = eca_layer(64)

        # Down 2
        self.downsample_2 = DownSamplingBlock(64, 128)
        # BRU Block 3
        
        dilation_block_3 = [2,4,4,16,16]
        
        self.BRU_Block_3 = nn.Sequential()
        for i in range(0, block_3):
            
            self.BRU_Block_3.add_module("AFC_Module_3_" + str(i), AFCModule(128, d=dilation_block_3[i]))
                   
        self.bn_gelu_4 = BNGeLU(128)

        # Branch 2
        
        self.conv_sipath1 = Conv(16, 64, 1, 1, 0, bn_acti=True)
        self.conv_sipath2 = Conv(64, 64, 3, 1, 1,groups=64, bn_acti=True)
        self.conv_sipath3 = Conv(64, 32, 1, 1, 0, bn_acti=True)
        
        self.atten_sipath = SpatialAttention()
        
         # 在模型初始化时调用权重初始化函数
        self._initialize_weights()

    def forward(self, input):
                
        output0 = self.init_conv(input) # [1, 3, 512, 1024] -> [1, 16, 256, 512]
        output0 = self.bn_gelu_1(output0) # [1, 16, 256, 512] -> [1, 16, 256, 512]

        # Detail Branch
        output_sipath = self.conv_sipath1(output0) # [1, 16, 256, 512] -> [1, 64, 256, 512]
        output_sipath = self.conv_sipath2(output_sipath) # [1, 64, 256, 512] -> [1, 64, 256, 512]
        output_sipath1 = self.conv_sipath3(output_sipath) # [1, 64, 256, 512] -> [1, 32, 256, 512]
        

        output_sipath = self.atten_sipath(output_sipath1) # [1, 32, 256, 512] -> [1, 1, 256, 512]

        # Branch1
        output1 = self.attention1_1(output0) # [1, 16, 256, 512] -> [1, 16, 256, 512]

        # block1
        output1 = self.BRU_Block_1(output1) # [1, 16, 256, 512] -> [1, 16, 256, 512]
        output1 = self.bn_gelu_2(output1) # [1, 16, 256, 512] -> [1, 16, 256, 512]
        output1 = self.attention2_1(output1) # [1, 16, 256, 512] -> [1, 16, 256, 512]

        # down1
        output1 = self.downsample_1(output1) # [1, 16, 256, 512] -> [1, 64, 128, 256]

        # block2
        output1 = self.BRU_Block_2(output1) # [1, 64, 128, 256] -> [1, 64, 128, 256]
        output1 = self.bn_gelu_3(output1) # [1, 64, 128, 256] -> [1, 64, 128, 256]
        output1 = self.attention3_1(output1) # [1, 64, 128, 256] -> [1, 64, 128, 256]

        # down2
        output1 = self.downsample_2(output1) # [1, 64, 128, 256] -> [1, 128, 64, 128]

        # block3
        output2 = self.BRU_Block_3(output1) # [1, 128, 64, 128] -> [1, 128, 64, 128]
        output2 = self.bn_gelu_4(output2) # [1, 128, 64, 128] -> [1, 128, 64, 128]       

        return output2, output_sipath, output_sipath1

    def _initialize_weights(self):
        init_weight(self, nn.init.kaiming_normal_, nn.BatchNorm2d, 1e-3, 0.1, mode='fan_in')
        
def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs)

def __init_weight(module, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            conv_init(m.weight, **kwargs)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, norm_layer):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            m.eps = bn_eps
            m.momentum = bn_momentum
        elif isinstance(m, nn.Linear):
            conv_init(m.weight, **kwargs)
            nn.init.constant_(m.bias, 0)

"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DECENet(classes=19).to(device)
    model.eval()
    out = model(torch.rand(1, 3, 512, 1024).to(device))
    print(out.shape)
    # summary(model, (1,3, 512, 1024))