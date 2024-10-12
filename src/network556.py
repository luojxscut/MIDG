import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


# class InpaintGenerator(BaseNetwork):
#     def __init__(self, residual_blocks=8, init_weights=True):
#         super(InpaintGenerator, self).__init__()
#
#         self.encoder = nn.Sequential(
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
#             nn.InstanceNorm2d(64, track_running_stats=False),
#             nn.ReLU(True),
#
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(128, track_running_stats=False),
#             nn.ReLU(True),
#
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(256, track_running_stats=False),
#             nn.ReLU(True)
#         )
#
#         blocks = []
#         for _ in range(residual_blocks):
#             block = ResnetBlock(256, 2)
#             blocks.append(block)
#
#         self.middle = nn.Sequential(*blocks)
#
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(128, track_running_stats=False),
#             nn.ReLU(True),
#
#             nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(64, track_running_stats=False),
#             nn.ReLU(True),
#
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
#         )
#
#         if init_weights:
#             self.init_weights()
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.middle(x)
#         x = self.decoder(x)
#         x = (torch.tanh(x) + 1) / 2
#
#         return x


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out



class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(ch_out),
            nn.InstanceNorm2d(ch_out, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out, track_running_stats=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out, track_running_stats=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class edgeatten(nn.Module):
    def __init__(self,k_size=3):
        super(edgeatten, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.convatt0 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_mask):
        edge_mask = F.interpolate(input=edge_mask, size=(x.size()[2], x.size()[3]), mode='bilinear', align_corners=True).bool()
        m = self.avg_pool(x)
        x0 = edge_mask[:, 0, :, :].unsqueeze(1) * x
        y = self.avg_pool(x0 + (edge_mask[:, 0, :, :].unsqueeze(1) == False) * m)
        y = self.convatt0(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        x0 = x0 * y.expand_as(x0)
        return x0







class InpaintGenerator(BaseNetwork):  # 添加了空间注意力和通道注意力
    def __init__(self, img_ch=4, output_ch=3,residual_blocks=8, init_weights=True):
        super(InpaintGenerator, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)  # 64
        self.Conv2 = conv_block(ch_in=64, ch_out=128)  # 64 128
        self.Conv3 = conv_block(ch_in=128, ch_out=256)  # 128 256
        self.Conv4 = conv_block(ch_in=256, ch_out=512)  # 256 512
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)  # 512 1024

        self.fusionconv1=conv_block(ch_in=128, ch_out=64)
        self.fusionconv2=conv_block(ch_in=256, ch_out=128)

        # self.cbam1 = CBAM(channel=64)
        # self.cbam2 = CBAM(channel=128)
        # self.cbam3 = CBAM(channel=256)
        # self.cbam4 = CBAM(channel=512)
        #
        self.edge=edgeatten()
        #
        # blocks = []
        # for _ in range(residual_blocks):
        #     block = ResnetBlock(1024, 2)
        #     blocks.append(block)
        #
        # self.middle = nn.Sequential(*blocks)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)  # 1024 512
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)  # 512 256
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)  # 256 128
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)  # 128 64
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)  # 64

        if init_weights:

            self.init_weights()


    def forward(self, x,edge_mask):
        # encoding path
        x1 = self.Conv1(x)
        # edge_mask=edge*mask
        edgeatt1=self.edge(x1,edge_mask)
        x1=torch.cat([x1, edgeatt1], dim=1)
        x1=self.fusionconv1(x1)
        # x1 = self.cbam1(x1)+edgeatt1 + x1
        # x1 = edgeatt1 + x1

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        edgeatt2=self.edge(x2,edge_mask)
        x2=torch.cat([x2, edgeatt2], dim=1)
        x2=self.fusionconv2(x2)

        # x2 = self.cbam2(x2) +edgeatt2+ x2
        # x2 = edgeatt2+ x2

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        edgeatt3=self.edge(x3,edge_mask)
        # x3 = self.cbam3(x3) +edgeatt3+ x3
        # x3 = edgeatt3+ x3

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        edgeatt4=self.edge(x4,edge_mask)
        # x4 = self.cbam4(x4) + x4 +edgeatt4
        # x4 =  x4 +edgeatt4

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # x5 = self.middle(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        x = (torch.tanh(d1) + 1) / 2

        return x

class conv_block2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block2, self).__init__()
        self.conv = nn.Sequential(
            # nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False)),
            # nn.BatchNorm2d(ch_out),
            nn.InstanceNorm2d(ch_out, track_running_stats=False),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(ch_out, track_running_stats=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv2, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            spectral_norm(nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(ch_out, track_running_stats=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class EdgeGenerator(BaseNetwork):
    def __init__(self, img_ch=3, output_ch=1,residual_blocks=8, init_weights=True):

    # def __init__(self, residual_blocks=8, use_spectral_norm=True, init_weights=True):
        super(EdgeGenerator, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block2(ch_in=img_ch, ch_out=64)  # 64
        self.Conv2 = conv_block2(ch_in=64, ch_out=128)  # 64 128
        self.Conv3 = conv_block2(ch_in=128, ch_out=256)  # 128 256
        self.Conv4 = conv_block2(ch_in=256, ch_out=512)  # 256 512
        self.Conv5 = conv_block2(ch_in=512, ch_out=1024)  # 512 1024



        # self.cbam1 = CBAM(channel=64)
        # self.cbam2 = CBAM(channel=128)
        # self.cbam3 = CBAM(channel=256)
        # self.cbam4 = CBAM(channel=512)
        #
        #
        # blocks = []
        # for _ in range(residual_blocks):
        #     block = ResnetBlock(1024, 2)
        #     blocks.append(block)

        # self.middle = nn.Sequential(*blocks)

        self.Up5 = up_conv2(ch_in=1024, ch_out=512)  # 1024 512
        self.Up_conv5 = conv_block2(ch_in=1024, ch_out=512)

        self.Up4 = up_conv2(ch_in=512, ch_out=256)  # 512 256
        self.Up_conv4 = conv_block2(ch_in=512, ch_out=256)

        self.Up3 = up_conv2(ch_in=256, ch_out=128)  # 256 128
        self.Up_conv3 = conv_block2(ch_in=256, ch_out=128)

        self.Up2 = up_conv2(ch_in=128, ch_out=64)  # 128 64
        self.Up_conv2 = conv_block2(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)  # 64

        if init_weights:
            self.init_weights()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        # edge_mask=edge*mask
        # x1 = self.cbam1(x1)+edgeatt1 + x1
        # x1 = edgeatt1 + x1

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        # x2 = self.cbam2(x2) +edgeatt2+ x2
        # x2 = edgeatt2+ x2

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        # edgeatt3 = self.edge(x3, edge_mask)
        # x3 = self.cbam3(x3) +edgeatt3+ x3
        # x3 = edgeatt3+ x3

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        # edgeatt4 = self.edge(x4, edge_mask)
        # x4 = self.cbam4(x4) + x4 +edgeatt4
        # x4 =  x4 +edgeatt4

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # x5 = self.middle(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        x = (torch.tanh(d1) + 1) / 2

        return x

# class EdgeGenerator(BaseNetwork):
#     def __init__(self, residual_blocks=8, use_spectral_norm=True, init_weights=True):
#         super(EdgeGenerator, self).__init__()
#
#         self.encoder = nn.Sequential(
#             nn.ReflectionPad2d(3),
#             spectral_norm(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
#             nn.InstanceNorm2d(64, track_running_stats=False),
#             nn.ReLU(True),
#
#             spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
#             nn.InstanceNorm2d(128, track_running_stats=False),
#             nn.ReLU(True),
#
#             spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), use_spectral_norm),
#             nn.InstanceNorm2d(256, track_running_stats=False),
#             nn.ReLU(True)
#         )
#
#         blocks = []
#         for _ in range(residual_blocks):
#             block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
#             blocks.append(block)
#
#         self.middle = nn.Sequential(*blocks)
#
#         self.decoder = nn.Sequential(
#             spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
#             nn.InstanceNorm2d(128, track_running_stats=False),
#             nn.ReLU(True),
#
#             spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), use_spectral_norm),
#             nn.InstanceNorm2d(64, track_running_stats=False),
#             nn.ReLU(True),
#
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
#         )
#
#         if init_weights:
#             self.init_weights()
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.middle(x)
#         x = self.decoder(x)
#         x = torch.sigmoid(x)
#         return x


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    writer = SummaryWriter()
    InpaintGenerator=InpaintGenerator()
    edgegenerator=EdgeGenerator()
    # x = torch.randn(size = (1,3,256,256))
    x = torch.randn(size = (1,4,224,224))
    x2 = torch.randn(size = (1,3,224,224))
    edge=torch.randn(size = (1,1,224,224))
    print(InpaintGenerator)
    print(InpaintGenerator(x,edge).shape)
    # # gen = gen.to('cuda')
    # # x = x.to('cuda')
    # # summary(gen,input_size=(1,3,256,256))
    # # summary(gen,input_size=(1,1,512,512))
    # # summary(gen,input_size=(1,1,256,256))
    #
    # logger = SummaryWriter(log_dir='./log')
    # logger.add_graph(InpaintGenerator, x)
    # writer.add_graph(InpaintGenerator, x)
    edgegenerator.to('cuda')
    summary(edgegenerator,input_size=(3,224,224))