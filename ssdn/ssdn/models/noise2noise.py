import torch
import torch.nn as nn


class Noise2Noise(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""

        super(Noise2Noise, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1),
        )
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1),
        )
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1),
        )
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)

        # Decoder
        upsample5 = self._block3(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)

        # Final activation
        return self._block6(concat1)


# import torch
# import torch.nn.functional as F
# from torch import nn

# # Model Definition
# class Noise2Noise(nn.Module):
#     def __init__(self):
#         super(Noise2Noise, self).__init__()

#         self.enc_conv0 = nn.Conv2d(3, 48, 3, padding=1)
#         self.enc_conv1 = nn.Conv2d(48, 48, 3, padding=1)

#         self.dec_conv5 = nn.Conv2d(96, 96, 3, padding=1)
#         self.dec_conv1a = nn.Conv2d(99, 64, 3, padding=1)
#         self.dec_conv1b = nn.Conv2d(64, 32, 3, padding=1)
#         self.dec_conv1c = nn.Conv2d(32, 3, 3, padding=1)

#     def forward(self, x):
#         # enc_conv0
#         out = self.enc_conv0(x)
#         out = F.relu(out)
#         # enc_conv1
#         out = self.enc_conv1(out)
#         out = F.relu(out)
#         # pool1
#         pool1 = F.max_pool2d(out, 2)
#         # enc_conv2
#         out = self.enc_conv1(pool1)
#         out = F.relu(out)
#         # pool2
#         pool2 = F.max_pool2d(out, 2)
#         # enc_conv3
#         out = self.enc_conv1(pool2)
#         # out = F.relu(out)
#         # pool3
#         pool3 = F.max_pool2d(out, 2)
#         # enc_conv4
#         out = self.enc_conv1(pool3)
#         out = F.relu(out)
#         # pool4
#         pool4 = F.max_pool2d(out, 2)
#         # enc_conv5
#         out = self.enc_conv1(pool4)
#         out = F.relu(out)
#         # pool5
#         pool5 = F.max_pool2d(out, 2)
#         # enc_conv6
#         out = self.enc_conv1(pool4)
#         out = F.relu(out)
#         # upsample5
#         upsample5 = F.upsample(pool5, scale_factor=2)
#         # concat5
#         out = torch.cat((upsample5, pool4), dim=1)
#         # dec_conv5a
#         out = self.dec_conv5(out)
#         out = F.relu(out)
#         # dec_conv5b
#         out = self.dec_conv5(out)
#         out = self.relu(out)
#         # upsample4
#         upsample4 = F.upsample(out, scale_factor=2)
#         # concat4
#         out = torch.cat((upsample4, pool3), dim=1)
#         # dec_conv4a
#         out = self.dec_conv5(out)
#         out = F.relu(out)
#         # dec_conv4b
#         out = self.dec_conv5(out)
#         out = self.relu(out)
#         # upsample3
#         upsample3 = F.upsample(out, scale_factor=2)
#         # concat3
#         out = torch.cat((upsample3, pool2), dim=1)
#         # dec_conv3a
#         out = self.dec_conv5(out)
#         out = F.relu(out)
#         # dec_conv3a
#         out = self.dec_conv5(out)
#         out = self.relu(out)
#         # upsample2
#         upsample2 = F.upsample(out, scale_factor=2)
#         # concat2
#         out = torch.cat((upsample2, pool1), dim=1)
#         # dec_conv2a
#         out = self.dec_conv5(out)
#         out = F.relu(out)
#         # dec_conv2b
#         out = self.dec_conv5(out)
#         out = self.relu(out)
#         # upsample1
#         upsample1 = F.upsample(out, scale_factor=2)
#         # concat1
#         out = torch.cat((upsample1, x), dim=1)

#         out = self.dec_conv1a(out)
#         out = F.relu(out)
#         out = self.dec_conv1b(out)
#         out = F.relu(out)
#         out = self.dec_conv1c(out)
#         out = F.relu(out)

#         return out
