import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetEncoderDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        backbone = cfg.MODEL
        pretrained = cfg.PRETRAINED

        self.hidden_dim = cfg.HIDDEN_DIM
        self.patch_size = cfg.PATCH_SIZE
        self.interp_mode = cfg.INTERP_MODE
        self.img_size = cfg.IMG_SIZE

        if backbone == "resnet34":
            net = models.resnet34(pretrained)
            channels = [64, 128, 256, 512]
        elif backbone == "resnet50":
            net = models.resnet50(pretrained)
            channels = [256, 512, 1024, 2048]
        elif backbone == "resnet101":
            net = models.resnet101(pretrained)
            channels = [256, 512, 1024, 2048]
        else:
            raise NotImplementedError("invalid backbone: {}".format(backbone))

        # Encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv1.weight.data = net.conv1.weight.data
        # self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        # Decoder
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(channels[0], self.hidden_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(channels[1], self.hidden_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(channels[2], 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.hidden_dim, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(channels[3], 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.hidden_dim, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )

        if cfg.FREEZE:
            # freeze encoder
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            for param in self.layer1.parameters():
                param.requires_grad = False
            for param in self.layer2.parameters():
                param.requires_grad = False
            for param in self.layer3.parameters():
                param.requires_grad = False
            for param in self.layer4.parameters():
                param.requires_grad = False

    def forward(self, inputs, outputs_size=None):
        """
        Args:
            inputs: list of images, each image is a numpy array of shape [H, W, 3]
            outputs_size: list of output size, each size is a tuple of (H, W)
        Returns:
            out_feats: multi-level features, each level is a list(len=bs) of features,
                each feature is a torch.Tensor of shape [C, H, W]
        """
        # pre-check
        assert isinstance(inputs, list), "inputs must be a list"
        assert isinstance(outputs_size, list) or outputs_size is None, \
            "outputs_size must be a list or None"
        assert len(inputs) == len(outputs_size) or outputs_size is None, \
            "inputs and outputs_size must have the same length"

        # unify input image size
        x = [
            self.interp(
                torch.from_numpy(np.array(batch)).float().unsqueeze(0).cuda(),
                size=tuple(self.img_size),
            )
            for batch in inputs
        ]
        x = torch.cat(x, dim=0)  # [B, C, H, W]
        assert x.shape[1] == 3, "image must have 3 channels"

        # check input size
        h, w = x.shape[2], x.shape[3]
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            new_h, new_w = self.get_new_size(h, w)
            x = self.interp(x, size=(new_h, new_w))

        # Encoder
        conv1_out = self.relu(self.bn1(self.conv1(x)))
        layer1_out = self.layer1(self.maxpool(conv1_out))
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)

        # Decoder
        layer1_out = self.deconv_layer1(layer1_out)
        layer2_out = self.deconv_layer2(layer2_out)
        layer3_out = self.deconv_layer3(layer3_out)
        layer4_out = self.deconv_layer4(layer4_out)

        out_feats = [
            [layer1_out[b] for b in range(layer1_out.shape[0])],  # 1/2
            [layer2_out[b] for b in range(layer2_out.shape[0])],  # 1/4
            [layer3_out[b] for b in range(layer3_out.shape[0])],  # 1/8
            [layer4_out[b] for b in range(layer4_out.shape[0])],  # 1/16
        ]
        if outputs_size is not None:
            out_feats = [
                [self.interp(feat.unsqueeze(0), size=outputs_size[b]).squeeze(0)
                 for b, feat in enumerate(level)]
                for level in out_feats
            ]

        return out_feats

    def get_new_size(self, h, w):
        new_H = (h // self.patch_size + 1) * self.patch_size
        new_W = (w // self.patch_size + 1) * self.patch_size
        return new_H, new_W

    def interp(self, x, size: tuple):
        return F.interpolate(
            x,
            size=size,
            mode=self.interp_mode,
            align_corners=False,
        )
