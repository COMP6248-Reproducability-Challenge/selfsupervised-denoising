import torch
import torch.nn.functional as F
from torch import nn

# Model Definition
class Noise2Noise(nn.Module):
    def __init__(self):
        super(Noise2Noise, self).__init__()

        self.enc_conv0 = nn.Conv2d(3, 48, (3,3), padding=0)
        self.enc_conv1 = nn.Conv2d(48, 48, (3,3), padding=0)

        self.dec_conv5 = nn.Conv2d(96, 96, (3,3), padding=0)
        self.dec_conv1a = nn.Conv2d(99, 64, (3,3), padding=0)
        self.dec_conv1b = nn.Conv2d(64, 32, (3,3), padding=0)
        self.dec_conv1c = nn.Conv2d(32, 3, (3,3), padding=0)

    def forward(self, x):
        # enc_conv0
        out = self.enc_conv0(x)
        out = F.relu(out)
        # enc_conv1
        out = self.enc_conv1(out)
        out = F.relu(out)
        # pool1
        pool1 = F.max_pool2d(out)
        # enc_conv2
        out = self.enc_conv1(pool1)
        out = F.relu(out)
        # pool2
        pool2 = F.max_pool2d(out)
        # enc_conv3
        out = self.enc_conv1(pool2)
        out = F.relu(out)
        # pool3
        pool3 = F.max_pool2d(out)
        # enc_conv4
        out = self.enc_conv1(pool3)
        out = F.relu(out)
        # pool4
        pool4 = F.max_pool2d(out)
        # enc_conv5
        out = self.enc_conv1(pool4)
        out = F.relu(out)
        # pool5
        pool5 = F.max_pool2d(out)
        # enc_conv6
        out = self.enc_conv1(pool4)
        out = F.relu(out)
        # upsample5
        out = F.upsample(pool5, scale_factor=2)
        # concat5
        out = torch.concat((pool5, pool4), dim=1)
        # dec_conv5a
        out = self.dec_conv5(out)
        out = F.relu(out)
        # dec_conv5b
        out = self.dec_conv5(out)
        out = self.relu(out)
        # upsample4
        out = F.upsample(out, scale_factor=2)
        # concat4
        out = torch.concat((out, pool3), dim=1)
        # dec_conv4a
        out = self.dec_conv5(out)
        out = F.relu(out)
        # dec_conv4b
        out = self.dec_conv5(out)
        out = self.relu(out)
        # upsample3
        out = F.upsample(out, scale_factor=2)
        # concat3
        out = torch.concat((out, pool2), dim=1)
        # dec_conv3a
        out = self.dec_conv5(out)
        out = F.relu(out)
        # dec_conv3a
        out = self.dec_conv5(out)
        out = self.relu(out)
        # upsample2
        out = F.upsample(out, scale_factor=2)
        # concat2
        out = torch.concat((out, pool1), dim=1)
        # dec_conv2a
        out = self.dec_conv5(out)
        out = F.relu(out)
        # dec_conv2b
        out = self.dec_conv5(out)
        out = self.relu(out)
        # upsample1
        out = F.upsample(out, scale_factor=2)
        # concat1
        out = torch.concat((out, x), dim=1)

        out = self.dec_conv1a(out)
        out = F.relu(out)
        out = self.dec_conv1b(out)
        out = F.relu(out)
        out = self.dec_conv1c(out)
        out = F.relu(out)

        return out