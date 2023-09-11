import torch
import torch.nn as nn
from torch.nn import functional as F


class MainFrame(nn.Module):
    def __init__(
            self,
            img_dim,
            img_time,
            in_channel,
            f_maps=[16, 32, 64],
            input_dropout_rate=0.1,
            num_layers=0
    ):
        super(MainFrame, self).__init__()
        self.img_dim = img_dim
        self.img_time = img_time
        self.f_maps = f_maps
        # 2Conv + Down
        self.encoders = self.temporalSqueeze(
            f_maps=[in_channel] + f_maps
        )

        # up + 2Conv
        self.decoders = self.temporalExcitation(
            f_maps=f_maps[::-1] + [in_channel]
        )
        #self.final_conv = nn.Conv3d(1, 1, 1)

    def temporalSqueeze(self, f_maps, num_layers=0):
        model_list = nn.ModuleList([])

        for idx in range(1, len(f_maps)):
            encoder_layer = SqueezeLayer(
                in_channels=f_maps[idx-1],
                out_channels=f_maps[idx],
            )
            model_list.append(encoder_layer)
        return model_list

    def temporalExcitation(self, f_maps):
        model_list = nn.ModuleList([])
        for idx in range(1, len(f_maps)):
            decoder_layer = ExcitationLayer(
                in_channels=f_maps[idx-1],
                out_channels=f_maps[idx],
                if_up_sample=True
            )
            model_list.append(decoder_layer)
        return model_list

    def process_by_trans(self, x):
        raise NotImplementedError("Should be implemented in child class!!")

    def forward(self, x):
        encoders_features = []
        for encoder in self.encoders:
            before_down, x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, before_down)

        x = self.process_by_trans(x)

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(x, encoder_features)
        
        #x = self.final_conv(x)

        return x


class SqueezeLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3
    ):
        super(SqueezeLayer, self).__init__()
        self.conv_net = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            if_encoder=True
        )
        self.down_sample = nn.Conv3d(out_channels, out_channels, kernel_size=(3,3,3), stride=(2,1,1), padding=(1,1,1))

    def forward(self, x):
        before_down = self.conv_net(x)
        x = self.down_sample(before_down)
        return before_down, x


class ExcitationLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            if_up_sample=True,
            kernel_size=3,
    ):
        super(ExcitationLayer, self).__init__()
        self.conv_net = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            if_encoder=False
        )
        self.if_up_sample = if_up_sample
        self.up_sample = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(4,3,3), stride=(2,1,1), padding=(1,1,1))

    def forward(self, x, encoder_features):
        if self.if_up_sample:
            x = self.up_sample(x)
        x += encoder_features
        # x = torch.cat((encoder_features, x), dim=2)
        x = self.conv_net(x)
        return x


class SingleConv(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=1
    ):
        super(SingleConv, self).__init__()
        self.add_module('Conv3d',
                        nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, stride=stride))
        self.add_module('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True))


class DoubleConv(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            if_encoder,
            kernel_size=3
    ):
        super(DoubleConv, self).__init__()
        if if_encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, padding=1))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, padding=1))
