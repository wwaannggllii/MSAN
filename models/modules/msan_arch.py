import torch
import torch.nn as nn
import models.modules.blocks as B
import torch.nn.functional as F
# from torchsummaryX import summary as summaryX

class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        '''首先可以把这个函数理解为类型转换函数，
        将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)，
        所以经过类型转换这个self.scale变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。'''
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        # print(self.scale)
        return input * self.scale

class SpatialGateW(nn.Module):
    def __init__(self, wn, gate_channel, expand = 4):
        super(SpatialGateW, self).__init__()

        self.gate_s1 = nn.Sequential(wn(nn.Conv2d(gate_channel, gate_channel*expand, kernel_size=1,stride=1,
                        padding=1//2, dilation=1)),
                                     nn.ReLU(True))
        self.gate_s2 = nn.Sequential(wn(nn.Conv2d(gate_channel, gate_channel*expand, kernel_size=3,stride=1,
                        padding=3, dilation=3)),
                                     nn.ReLU(True))
        self.gate_s3 = wn(nn.Conv2d(gate_channel*expand*2, 1, kernel_size=1, stride=1, padding=1//2, dilation=1))
        self.gate_s3_s = nn.Sigmoid()
        # self.gate_s3_s = nn.Tanh()
        # self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out1 = self.gate_s1(x)
        out2 = self.gate_s2(x)
        # print(out1.shape)
        # print(out2.shape)
        # exit()
        out = torch.cat([out1, out2], 1)
        # print(out.shape)
        # exit()
        # out = out1 + out2
        out = self.gate_s3(out) #[2, 1, 94, 94]
        out = self.gate_s3_s(out)
        # print ('----------------------------------------')
        # print ('x',x.shape)
        # print ('----------------------------------------')
        # print ('out',out.shape)
        # print ('----------------------------------------')
        # print ('out1', out1.shape)
        # print ('----------------------------------------')
        # print ('out2', out2.shape)
        # print ('----------------------------------------')
        # exit()

        out = out * x + x

        return out.expand_as(x)


class CALayerW(nn.Module):
    def __init__(self, wn, channel, expand = 4):
        super(CALayerW, self).__init__()
        # global average pooling: feature --> point

        # self.gamma = nn.Parameter(torch.zeros(1))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.gate_ss = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=1,padding=1, dilation=1),
        #                              nn.PReLU())
        # self.gate_ss = nn.Conv2d(channel, channel // reduction, kernel_size=1)


        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                wn(nn.Conv2d(channel, channel*expand, kernel_size=1,stride=1, padding=1//2, dilation=1, bias=True)),
                #nn.BatchNorm2d(in_ch),
            nn.ReLU(True),
               # nn.LeakyReLU(),
                wn(nn.Conv2d(channel*expand, channel, kernel_size=1,stride=1, padding=1//2, dilation=1, bias=True)),
                nn.Sigmoid()
        )
    def forward(self, x):
        # max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        # avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        y = self.avg_pool(x)
        # print(y.shape)
        # exit()
        y = self.conv_du(y) #torch.Size([2, 64, 1, 1])
        out = x * y + x
        return out.expand_as(x)

class MSLayer(nn.Module):
    def __init__(self, wn, channel,act=nn.ReLU(True), reduction=8):
        super(MSLayer, self).__init__()

        self.down1_1 = nn.Sequential(
               wn(nn.Conv2d(channel, channel // reduction, 3, stride=1, padding=1, dilation=1)),
               act
         )
        self.down1_2 = nn.Sequential(
               wn(nn.Conv2d(channel, channel // reduction, 3, stride=1, padding=3, dilation=3)),
               act
         )
        self.down1_3 = nn.Sequential(
               wn(nn.Conv2d(channel, channel // reduction, 3, stride=1, padding=5, dilation=5)),
               act
                  )
        self.conv_1x1_output0 = wn(nn.Conv2d((channel // reduction) * 2, channel // reduction, 1, 1, 0))
        self.conv_1x1_output1 = wn(nn.Conv2d((channel // reduction) * 2, channel // reduction, 1, 1, 0))
        self.conv_1x1_output2 = wn(nn.Conv2d((channel // reduction) * 3, channel, 1, 1, 0))

    def forward(self, x):

        out_down1 = self.down1_1(x)
        out_down2 = self.down1_2(x)


        out_down22 = self.conv_1x1_output0(torch.cat([out_down1, out_down2], dim=1))
        out_down3 = self.down1_3(x)
        # print(out_down1.shape)
        # print(out_down2.shape)
        # print(out_down3.shape)
        # exit()
        out_down33 = self.conv_1x1_output1(torch.cat([out_down22, out_down3], dim=1))
        out_down = self.conv_1x1_output2(torch.cat([out_down1, out_down22, out_down33], dim=1))
        out_down = out_down + x
        return out_down

class MSAB(nn.Module):
    def __init__(self,  wn, n_feats, act=nn.ReLU(True)):
        super(MSAB, self).__init__()
        self.K1 = Scale(1)
        self.K2 = Scale(1)

        # self.conv = wn(nn.Conv2d(n_feats, n_feats, 3, 1, 3//2))
        self.pam_attention_1_4 = SpatialGateW(wn, 24, expand=4)
        self.cam_attention_1_4 = CALayerW(wn, 24, expand=4)

        body = []
        body.append(
            wn(nn.Conv2d(16, 16, 3, padding=3 // 2)))
        body.append(act)
        self.body = nn.Sequential(*body)

        self.conv_1x1_output = wn(nn.Conv2d(n_feats, n_feats, 1, 1, 0))

        self.ms = MSLayer(wn, n_feats, act=nn.ReLU(inplace=True), reduction=1)
    def forward(self, x):

        # x = self.conv0(x)

        ms = self.ms(x)

        # out_a = self.conv(ms)  # 进一步提取特征用于分割

        # spx = torch.split(out_a, self.m_features, 1)  # # 每块大小为32,torch.Size([2, 32, 32, 32])
        x_slice_k3 = ms[:, :24, :, :]
        x_slice_k5 = ms[:, 24:48, :, :]
        x_slice_k7 = ms[:, 48:, :, :]
        attn_sam = self.pam_attention_1_4(x_slice_k3)
        attn_cam = self.cam_attention_1_4(x_slice_k5)
        ori = self.body(x_slice_k7)
        out = self.conv_1x1_output(torch.cat([attn_sam, attn_cam,ori], dim=1))
        # print(ms.shape)
        # print(x.shape)
        # print(out.shape)
        # exit()

        # out = self.K1(x) + self.K2(out)
        out = x + out

        return out


class MSAN(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_recurs, upscale_factor, norm_type=None,
                 act_type='prelu'):
        super(MSAN, self).__init__()
        self.num_recurs = num_recurs  #12
        # if upscale_factor == 2:
        #     stride = 2
        #     padding = 2
        #     projection_filter = 6
        # if upscale_factor == 3:
        #     stride = 3
        #     padding = 2
        #     projection_filter = 7
        # elif upscale_factor == 4:
        #     stride = 4
        #     padding = 2
        #     projection_filter = 8

        act = nn.ReLU(True)
        # wn = lambda x: x
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(
            [0.4488, 0.4371, 0.4040])).view([1, 3, 1, 1])
        head = []
        head.append(
            wn(nn.Conv2d(in_channels, num_features, 3, padding=3 // 2)))

        # define tail module
        out_feats = upscale_factor * upscale_factor * out_channels
        tail = []

        tail.append(
            wn(nn.Conv2d(num_features, out_feats, 3, padding=3 // 2)))
        tail.append(nn.PixelShuffle(upscale_factor))

        skip = []
        skip.append(
            wn(nn.Conv2d(in_channels, out_feats, 5, padding=5 // 2))
        )
        skip.append(nn.PixelShuffle(upscale_factor))

        block = []
        for i in range(self.num_recurs):
            block.append(MSAB(wn, num_features, act))

        self.head = nn.Sequential(*head)
        self.block = nn.Sequential(*block)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)
        #
        # deconv = []
        # deconv.append(B.DeconvBlock(num_features, num_features, projection_filter, stride=stride,
        #                              padding=padding, norm_type=None, act_type=None))
        # deconv.append(act)
        # deconv.append( wn(nn.Conv2d(num_features, out_channels, 3, padding=3 // 2)))
        # self.deconv = nn.Sequential(*deconv)

    def forward(self, x):
        x = (x - self.rgb_mean.cuda() * 255) / 127.5
        s = self.skip(x)
        x = self.head(x)
        o0 = x
        o1 = self.block(o0)
        o2 = o0 + o1

        x = self.tail(o2)
        x += s
        out = x * 127.5 + self.rgb_mean.cuda() * 255
        return out

# summaryX(MSAN(), torch.zeros((1,3,640,320)))