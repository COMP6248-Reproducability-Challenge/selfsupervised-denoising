""" PyTorch implementation of U-Net model for N2N and SSDN.
"""

import torch
import ssdn
import torch.nn as nn

from torch import Tensor

from ssdn.models.utility import Shift2d


class NoiseNetwork(nn.Module):
    """Custom U-Net architecture for Self Supervised Denoising (SSDN) and Noise2Noise (N2N).
    Base N2N implementation was made with reference to @joeylitalien's N2N implementation.
    Changes made are removal of weight sharing when blocks are reused. Usage of LeakyReLu
    over standard ReLu and incorporation of blindspot functionality.

    Unlike other typical U-Net implementations dropout is not used when the model is trained.

    When in blindspot mode the following behaviour changes occur:

        * Input batches are duplicated for rotations: 0, 90, 180, 270. This increases the
          batch size by 4x. After the encode-decode stage the rotations are undone and
          concatenated on the channel axis with the associated original image. This 4x
          increase in channel count is collapsed to the standard channel count in the
          first 1x1 kernel convolution.

        * To restrict the receptive field into the upward direction a shift is used for
          convolutions (see ShiftConv2d) and downsampling. Downsampling uses a single
          pixel shift prior to max pooling as dictated by Laine et al. This is equivalent
           to applying a shift on the upsample.

    Args:
        in_channels (int, optional): Number of input channels, this will typically be either
            1 (Mono) or 3 (RGB) but can be more. Defaults to 3.
        out_channels (int, optional): Number of channels the final convolution should output.
            Defaults to 3.
        blindspot (bool, optional): Whether to enable the network blindspot. This will
            add in rotation stages and shift stages while max pooling and during convolutions.
            A futher shift will occur after upsample. Defaults to False.
        zero_output_weights (bool, optional): Whether to initialise the weights of
                `nin_c` to zero. This is not mentioned in literature but is done as part
                of the tensorflow implementation for the parameter estimation network.
                Defaults to False.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        blindspot: bool = False,
        zero_output_weights: bool = False,
    ):
        super(NoiseNetwork, self).__init__()
        self._blindspot = blindspot
        self._zero_output_weights = zero_output_weights
        self.Conv2d = ShiftConv2d if self.blindspot else nn.Conv2d

        ####################################
        # Encode Blocks
        ####################################

        def _max_pool_block(max_pool: nn.Module) -> nn.Module:
            if blindspot:
                return nn.Sequential(Shift2d((1, 0)), max_pool)
            return max_pool

        # Layers: enc_conv0, enc_conv1, pool1
        self.encode_block_1 = nn.Sequential(
            self.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            self.Conv2d(48, 48, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            _max_pool_block(nn.MaxPool2d(2)),
        )

        # Layers: enc_conv(i), pool(i); i=2..5
        def _encode_block_2_3_4_5() -> nn.Module:
            return nn.Sequential(
                self.Conv2d(48, 48, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                _max_pool_block(nn.MaxPool2d(2)),
            )

        # Separate instances of same encode module definition created
        self.encode_block_2 = _encode_block_2_3_4_5()
        self.encode_block_3 = _encode_block_2_3_4_5()
        self.encode_block_4 = _encode_block_2_3_4_5()
        self.encode_block_5 = _encode_block_2_3_4_5()

        ####################################
        # Decode Blocks
        ####################################
        # Layers: enc_conv6, upsample5
        self.decode_block_6 = nn.Sequential(
            self.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1),
        )

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self.decode_block_5 = nn.Sequential(
            self.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            self.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1),
        )

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        def _decode_block_4_3_2() -> nn.Module:
            return nn.Sequential(
                self.Conv2d(144, 96, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                self.Conv2d(96, 96, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1),
            )

        # Separate instances of same decode module definition created
        self.decode_block_4 = _decode_block_4_3_2()
        self.decode_block_3 = _decode_block_4_3_2()
        self.decode_block_2 = _decode_block_4_3_2()

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self.decode_block_1 = nn.Sequential(
            self.Conv2d(96 + in_channels, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            self.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        ####################################
        # Output Block
        ####################################

        if self.blindspot:
            # Shift 1 pixel down
            self.shift = Shift2d((1, 0))
            # 4 x Channels due to batch rotations
            nin_a_io = 384
        else:
            nin_a_io = 96

        # nin_a,b,c, linear_act
        self.output_conv = self.Conv2d(96, out_channels, 1)
        self.output_block = nn.Sequential(
            self.Conv2d(nin_a_io, nin_a_io, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            self.Conv2d(nin_a_io, 96, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            self.output_conv,
        )

        # Initialize weights
        self._init_weights()

    @property
    def blindspot(self) -> bool:
        return self._blindspot

    def _init_weights(self):
        """Initializes weights using Kaiming  He et al. (2015).
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
        if self._zero_output_weights:
            self.output_conv.weight.zero_()

    def forward(self, x: Tensor) -> Tensor:
        if self.blindspot:
            rotated = [ssdn.utils.rotate(x, rot) for rot in (0, 90, 180, 270)]
            x = torch.cat((rotated), dim=0)

        # Encoder
        pool1 = self.encode_block_1(x)
        pool2 = self.encode_block_2(pool1)
        pool3 = self.encode_block_3(pool2)
        pool4 = self.encode_block_4(pool3)
        pool5 = self.encode_block_5(pool4)

        # Decoder
        upsample5 = self.decode_block_6(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.decode_block_5(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.decode_block_4(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.decode_block_3(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.decode_block_2(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        x = self.decode_block_1(concat1)

        # Output
        if self.blindspot:
            # Apply shift
            shifted = self.shift(x)
            # Unstack, rotate and combine
            rotated_batch = torch.chunk(shifted, 4, dim=0)
            aligned = [
                ssdn.utils.rotate(rotated, rot)
                for rotated, rot in zip(rotated_batch, (0, 270, 180, 90))
            ]
            x = torch.cat(aligned, dim=1)

        x = self.output_block(x)

        return x

    def input_wh_mul(self) -> int:
        """Multiple that both the width and height dimensions of an input must be to be
        processed by the network. This is devised from the number of pooling layers that
        reduce the input size.

        Returns:
            int: Dimension multiplier
        """
        max_pool_layers = 5
        return 2 ** max_pool_layers


class ShiftConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        """Custom convolution layer as defined by Laine et al. for restricting the
        receptive field of a convolution layer to only be upwards. For a h × w kernel,
        a downwards offset of k = [h/2] pixels is used. This is applied as a k sized pad
        to the top of the input before applying the convolution. The bottom k rows are
        cropped out for output.
        """
        super().__init__(*args, **kwargs)
        self.shift_size = (self.kernel_size[0] // 2, 0)
        # Use individual layers of shift for wrapping conv with shift
        shift = Shift2d(self.shift_size)
        self.pad = shift.pad
        self.crop = shift.crop

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad(x)
        x = super().forward(x)
        x = self.crop(x)
        return x
