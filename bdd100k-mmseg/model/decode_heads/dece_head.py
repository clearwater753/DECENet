import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
from ..losses import accuracy
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

class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.relu = nn.ReLU6(inplace= True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
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

class EABranch(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads=8,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_injection=True,
                 use_cross_kv=True,
                 cross_size=12):
        super().__init__()
        in_channels_h, in_channels_l = in_channels
        out_channels_h, out_channels_l = out_channels
        assert in_channels_h == out_channels_h, "in_channels_h is not equal to out_channels_h"
        self.out_channels_l = out_channels_l
        self.proj_flag = in_channels_l != out_channels_l
        self.use_injection = use_injection
        self.use_cross_kv = use_cross_kv
        self.cross_size = cross_size
        

        self.attn_l = ExternalAttention(
            in_channels_l,
            out_channels_l,
            inter_channels=cross_size * cross_size,
            num_heads=num_heads,
            use_cross_kv=True) #use_cross_kv=False)
        
        if use_cross_kv:
            self.cross_kv = nn.Sequential(
                nn.BatchNorm2d(out_channels_h),
                nn.AdaptiveMaxPool2d(output_size=(self.cross_size, self.cross_size)),
                nn.Conv2d(out_channels_h, 2 * out_channels_l, 1, 1, 0))
            

    def forward(self, x_h, x_l):
        
        crosskv = self.cross_kv(x_h)
        cross_k, cross_v = crosskv.chunk(2, dim=1)
        
        cross_k = cross_k.permute([0, 2, 3, 1])
        cross_k = torch.reshape(cross_k,[-1, self.out_channels_l, 1, 1])
        
        cross_v = torch.reshape(cross_v,
            [-1, self.cross_size * self.cross_size, 1,
             1])  # n*out_channels_h,144,1,1
        
        x_l = x_l + self.attn_l(x_l, cross_k, cross_v)  # n,out_chs_h,h,w
        return x_l

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class ExternalAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels,
                 num_heads=8,
                 use_cross_kv=False):
        super().__init__()
        assert out_channels % num_heads == 0, \
            "out_channels ({}) should be be a multiple of num_heads ({})".format(out_channels, num_heads)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.use_cross_kv = use_cross_kv
        self.norm = nn.BatchNorm2d(in_channels)
        self.same_in_out_chs = in_channels == out_channels

        self.reatt = nn.Parameter(1e-5 * torch.ones((1, 64, 1, 1)), requires_grad=True)

        if use_cross_kv:
            assert self.same_in_out_chs, "in_channels is not equal to out_channels when use_cross_kv is True"
        
    def _act_sn(self, x):
        H,W = x.shape[2:]
        #x = x.reshape([-1, self.inter_channels, 0, 0]) * (self.inter_channels **-0.5)
        x = torch.reshape(x,[-1, self.inter_channels, H, W]) * (self.inter_channels ** -0.5)
        
        x = F.softmax(x, dim=1)# axis=1)
        #x = x.reshape([1, -1, 0, 0])
        x = torch.reshape(x,[1, -1, H, W])
        return x

    def _act_dn(self, x):
        x_shape = x.shape   #paddle.shape(x)
        h, w = x_shape[2], x_shape[3]
        #x = x.reshape([0, self.num_heads, self.inter_channels // self.num_heads, -1])
        x = torch.reshape(x,[0, self.num_heads, self.inter_channels // self.num_heads, -1])
        x = F.softmax(x, dim=3)
        x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-06)
        #x = x.reshape([0, self.inter_channels, h, w])
        x = torch.reshape(x,[0, self.inter_channels, h, w])
        return x

    def forward(self, x, cross_k=None, cross_v=None):
        """
        Args:
            x (Tensor): The input tensor.
            cross_k (Tensor, optional): The dims is (n*144, c_in, 1, 1)
            cross_v (Tensor, optional): The dims is (n*c_in, 144, 1, 1)
        """
        x = self.norm(x)
        if not self.use_cross_kv:
            x = F.conv2d(
                x,
                self.k,
                bias=None,
                stride=2 if not self.same_in_out_chs else 1,
                padding=0)  # n,c_in,h,w -> n,c_inter,h,w
            x = self._act_dn(x)  # n,c_inter,h,w
            x = F.conv2d(
                x, self.v, bias=None, stride=1,
                padding=0)  # n,c_inter,h,w -> n,c_out,h,w
        else:
            assert (cross_k is not None) and (cross_v is not None), \
                "cross_k and cross_v should no be None when use_cross_kv"
            B = x.shape[0]
            assert B > 0, "The first dim of x ({}) should be greater than 0, please set input_shape for export.py".format(
                B)
            
            H,W = x.shape[2:]
            x = torch.reshape(x,[1, -1, H, W])
            
            x = F.conv2d(
                x, cross_k, bias=None, stride=1, padding=0,
                groups=B)  # 1,n*c_in,h,w -> 1,n*144,h,w  (group=B)
            
            x = x + (self.reatt).repeat(1,B,H,W)
            x = self._act_sn(x)
            
            x = F.conv2d(
                x, cross_v, bias=None, stride=1, padding=0,
                groups=B)  # 1,n*144,h,w -> 1, n*c_in,h,w  (group=B)
            #x = x.reshape([-1, self.in_channels, 0,0])  # 1, n*c_in,h,w -> n,c_in,h,w  (c_in = c_out)
            x = torch.reshape(x,[-1, self.in_channels, H, W])

        return x

@MODELS.register_module()
class DECEHead(BaseDecodeHead):
    
    def __init__(self, classes=19, block_4 = 3, block_5 = 3, aug_weight = 0.5, **kwargs):
        super().__init__(**kwargs)
        #print("using DECENet")
        self.aug_weight = aug_weight
        # Attention 4
        self.attention4_1 = eca_layer(128)
        
        #===============================================
        self.sidehead = nn.Sequential(
            Conv(nIn=128,nOut=128,kSize=3,stride=1,padding=1),
            Conv(nIn=128,nOut=classes,kSize=1,stride=1,padding=0),
        )

        # --------------Decoder   ----------------- #
        # Up 1
        self.upsample_1 = UpsamplerBlock(128, 64)

        # BRU Block 4
        dilation_block_4 = [2,2,2]
        self.BRU_Block_4 = nn.Sequential()
        for i in range(0, block_4):
            
            self.BRU_Block_4.add_module("AFC_Module_4_" + str(i), AFCModule(64, d=dilation_block_4[i]))
        self.bn_gelu_5 = BNGeLU(64)
        
        # Up 2
        self.upsample_2 = UpsamplerBlock(64, 32)
        # BRU Block 5
        dilation_block_5 = [2,2,2]
        self.BRU_Block_5 = nn.Sequential()
        for i in range(0, block_5):
            
            self.BRU_Block_5.add_module("AFC_Module_5_" + str(i), AFCModule(32, d=dilation_block_5[i]))
        self.bn_gelu_6 = BNGeLU(32)
        self.attention6_1 = eca_layer(32)
        # Branch 2
        
        self.bn_gelu_8 = BNGeLU(33)

        self.endatten = CoordAtt(33, 33)

        self.output_conv = nn.ConvTranspose2d(33, classes, 2, stride=2, padding=0, output_padding=0, bias=True)

        self.cross2 = EABranch(in_channels=[32,64], out_channels=[32,64], num_heads=4, drop_rate=0.05, cross_size=8, use_cross_kv=True)
        
        self._init_weights()

    def forward(self, inputs):
        output2, output_sipath, output_sipath1 = inputs
        # 辅助损失头
        # need output2
        out_side = F.interpolate(self.sidehead(output2), [512, 1024], mode='bilinear', align_corners=False) # [1, 128, 64, 128] -> [1, class, 512, 1024]
        # ---------- Decoder ----------------
        
        # 开始上采样
        # need output2, output_sipath1, output_sipath
        # up1
        output = self.attention4_1(output2) # [1, 128, 64, 128] -> [1, 128, 64, 128]


        output = self.upsample_1(output) # [1, 128, 64, 128] -> [1, 64, 128, 256]

        # block4
        output = self.BRU_Block_4(output) # [1, 64, 128, 256] -> [1, 64, 128, 256]
        output = self.bn_gelu_5(output) # [1, 64, 128, 256] -> [1, 64, 128, 256]


        output = self.cross2(output_sipath1, output) # [1, 32, 256, 512], [1, 64, 256, 512] -> [1, 64, 256, 512]

        # up2
        output = self.upsample_2(output) # [1, 64, 128, 256] -> [1, 32, 256, 512]

        # block5
        output = self.BRU_Block_5(output) # [1, 32, 256, 512] -> [1, 32, 256, 512]
        output = self.bn_gelu_6(output) # [1, 32, 256, 512] -> [1, 32, 256, 512]
        output = self.attention6_1(output) # [1, 32, 256, 512] -> [1, 32, 256, 512]


        output = torch.cat((output , output_sipath),dim=1) # [1, 32, 256, 512], [1, 1, 256, 512] -> [1, 33, 256, 512]

        output = self.bn_gelu_8(output) # [1, 33, 256, 512] -> [1, 33, 256, 512]

        output = self.endatten(output) # [1, 33, 256, 512] -> [1, 33, 256, 512]

        out = self.output_conv(output) # [1, 33, 256, 512] -> [1, class, 512, 1024]

        return out, out_side
    
    def loss(self, inputs, batch_data_samples, train_cfg=None) -> dict:        
        seg_logits, seg_logits_aug = self.forward(inputs)
        
        losses = self.loss_by_feat(seg_logits, seg_logits_aug, batch_data_samples)
        return losses
    
    def loss_by_feat(self, seg_logits, seg_logits_aug, batch_data_samples) -> dict:
        seg_label = self._stack_batch_gt(batch_data_samples)        
        loss = dict()
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label) + self.aug_weight*loss_decode(
                        seg_logits_aug,
                        seg_label)
            else:
                loss[loss_decode.loss_name] = loss[loss_decode.loss_name]+loss_decode(
                    seg_logits,
                    seg_label,)+self.aug_weight*loss_decode(
                        seg_logits_aug,
                        seg_label)

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss

    def predict(self, inputs, batch_img_metas,
                test_cfg=None):
        seg_logits, seg_logits_aug = self.forward(inputs)

        return self.predict_by_feat(seg_logits, batch_img_metas)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.1
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)   



