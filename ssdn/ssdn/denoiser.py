from __future__ import annotations


import torch
import torch.nn as nn
import ssdn

from torch import Tensor

from ssdn.params import (
    ConfigValue,
    PipelineOutput,
    Pipeline,
    NoiseValue,
)

from ssdn.models import NoiseNetwork
from ssdn.datasets import NoisyDataset

from typing import Dict, List


class Denoiser(nn.Module):

    MODEL = "denoiser_model"
    SIGMA_ESTIMATOR = "sigma_estimation_model"
    ESTIMATED_SIGMA = "estimated_sigma"

    def __init__(
        self, cfg: Dict, device: str = None,
    ):
        super().__init__()
        # Configure device
        if device:
            device = torch.device(device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Store the denoiser configuration
        self.cfg = cfg

        # Models to use during training, these can be parallelised
        self.models = nn.ModuleDict()
        # References to models that are guaranteed to be device independent
        self._models = nn.ModuleDict()
        # Initialise networks using current configuration
        self.init_networks()
        # Learnable parameters
        self.l_params = nn.ParameterDict()
        self.init_l_params()

    def init_networks(self):
        # Calculate input and output channel count for networks
        in_channels = self.cfg[ConfigValue.IMAGE_CHANNELS]
        if self.cfg[ConfigValue.PIPELINE] == Pipeline.SSDN:
            if self.cfg[ConfigValue.DIAGONAL_COVARIANCE]:
                out_channels = in_channels * 2  # Means, diagonal of A
            else:
                out_channels = (
                    in_channels + (in_channels * (in_channels + 1)) // 2
                )  # Means, triangular A.
        else:
            out_channels = in_channels

        # Create general denoising model
        self.add_model(
            Denoiser.MODEL,
            NoiseNetwork(
                in_channels=in_channels,
                out_channels=out_channels,
                blindspot=self.cfg[ConfigValue.BLINDSPOT],
            ),
        )
        # Create separate model for variable parameter estimation
        if (
            self.cfg[ConfigValue.PIPELINE] == Pipeline.SSDN
            and self.cfg[ConfigValue.NOISE_VALUE] == NoiseValue.UNKNOWN_VARIABLE
        ):
            self.add_model(
                Denoiser.SIGMA_ESTIMATOR,
                NoiseNetwork(
                    in_channels=in_channels,
                    out_channels=1,
                    blindspot=False,
                    zero_output_weights=True,
                ),
            )

    def init_l_params(self):
        if (
            self.cfg[ConfigValue.PIPELINE] == Pipeline.SSDN
            and self.cfg[ConfigValue.NOISE_VALUE] == NoiseValue.UNKNOWN_CONSTANT
        ):
            init_value = torch.zeros((1, 1, 1, 1))
            self.l_params[Denoiser.ESTIMATED_SIGMA] = nn.Parameter(init_value)

    def get_model(self, model_id: str, parallelised: bool = True) -> nn.Module:
        model_dict = self.models if parallelised else self._models
        return model_dict[model_id]

    def add_model(self, model_id: str, model: nn.Module, parallelise: bool = True):
        self._models[model_id] = model
        if parallelise:
            parallel_model = nn.DataParallel(model)
        else:
            parallel_model = model
        # Move to master device (GPU 0 or CPU)
        parallel_model.to(self.device)
        self.models[model_id] = parallel_model

    def forward(self, data: Tensor) -> Tensor:
        """Pass an input into the denoiser for inference. This will not train
        the network. Inference will be applied using current model state with
        the configured pipeline.

        Args:
            data (Tensor): Image or batch of images to denoise in BCHW format.

        Returns:
            Tensor: Denoised image or images.
        """
        assert NoisyDataset.INPUT == 0
        inputs = [data]
        outputs = self.run_pipeline(inputs)
        return outputs[PipelineOutput.IMG_DENOISED]

    def run_pipeline(self, data: List, **kwargs):
        if self.cfg[ConfigValue.PIPELINE] == Pipeline.MSE:
            return self._mse_pipeline(data, **kwargs)
        elif self.cfg[ConfigValue.PIPELINE] == Pipeline.SSDN:
            return self._ssdn_pipeline(data, **kwargs)
        elif self.cfg[ConfigValue.PIPELINE] == Pipeline.MASK_MSE:
            return self._mask_mse_pipeline(data, **kwargs)
        else:
            raise NotImplementedError("Unsupported processing pipeline")

    def _mse_pipeline(self, data: List, **kwargs) -> Dict:
        outputs = {PipelineOutput.INPUTS: data}
        # Run the input through the model
        inp = data[NoisyDataset.INPUT].to(self.device)
        inp.requires_grad = True
        cleaned = self.models[Denoiser.MODEL](inp)
        outputs[PipelineOutput.IMG_DENOISED] = cleaned

        # If reference images are provided calculate the loss
        # as MSE whilst preserving individual loss per batch
        if len(data) >= NoisyDataset.REFERENCE:
            ref = data[NoisyDataset.REFERENCE].to(self.device)
            ref.requires_grad = True
            loss = nn.MSELoss(reduction="none")(cleaned, ref)
            loss = loss.view(loss.shape[0], -1).mean(1, keepdim=True)
            outputs[PipelineOutput.LOSS] = loss

        return outputs

    def _mask_mse_pipeline(self, data: List, **kwargs) -> Dict:
        outputs = {PipelineOutput.INPUTS: data}
        # Run the input through the model
        inp = data[NoisyDataset.INPUT].to(self.device)
        inp.requires_grad = True
        cleaned = self.models[Denoiser.MODEL](inp)
        outputs[PipelineOutput.IMG_DENOISED] = cleaned

        # If reference images are provided calculate the loss
        # as MSE whilst preserving individual loss per batch
        if len(data) >= NoisyDataset.REFERENCE:
            ref = data[NoisyDataset.REFERENCE].to(self.device)
            ref.requires_grad = True
            # TODO: change to use MSE loss only on masked pixels
            data[NoiseDataset.METADATA]['mask_coords']
            loss = nn.MSELoss(reduction="none")(cleaned, ref)
            loss = loss.view(loss.shape[0], -1).mean(1, keepdim=True)
            outputs[PipelineOutput.LOSS] = loss

        return outputs

    def _ssdn_pipeline(self, data: List, **kwargs) -> Dict:
        debug = False

        inp = data[NoisyDataset.INPUT]
        noisy_in = inp.to(self.device)

        # noisy_params_in = standard deviation of noise
        metadata = data[NoisyDataset.METADATA]
        noise_params_in = metadata[NoisyDataset.Metadata.INPUT_NOISE_VALUES]

        # config for noise params/style
        noise_style = self.cfg[ConfigValue.NOISE_STYLE]
        noise_params = self.cfg[ConfigValue.NOISE_VALUE]

        # Equivalent of blindspot_pipeline
        input_shape = metadata[NoisyDataset.Metadata.IMAGE_SHAPE]
        num_channels = self.cfg[ConfigValue.IMAGE_CHANNELS]
        assert num_channels in [1, 3]

        diagonal_covariance = self.cfg[ConfigValue.DIAGONAL_COVARIANCE]

        if debug:
            print("Image shape:", input_shape)
        if debug:
            print("Num. channels:", num_channels)

        # Clean data distribution.
        # Calculation still needed for line 175
        num_output_components = (
            num_channels + (num_channels * (num_channels + 1)) // 2
        )  # Means, triangular A.
        if diagonal_covariance:
            num_output_components = num_channels * 2  # Means, diagonal of A.
        if debug:
            print("Num. output components:", num_output_components)
        # Call the NN with the current image etc.
        net_out = self.models[Denoiser.MODEL](noisy_in)
        # net_out = net_out.type(torch.float64)
        if debug:
            print("Net output shape:", net_out.shape)
        mu_x = net_out[:, 0:num_channels, ...]  # Means (NCHW).
        A_c = net_out[
            :, num_channels:num_output_components, ...
        ]  # Components of triangular A.
        if debug:
            print("Shape of A_c:", A_c.shape)
        if num_channels == 1:
            sigma_x = A_c ** 2  # N1HW
        elif num_channels == 3:
            if debug:
                print("Shape before permute:", A_c.shape)
            A_c = A_c.permute(0, 2, 3, 1)  # NHWC
            if debug:
                print("Shape after permute:", A_c.shape)
            if diagonal_covariance:
                c00 = A_c[..., 0] ** 2
                c11 = A_c[..., 1] ** 2
                c22 = A_c[..., 2] ** 2
                zro = torch.zeros(c00.shape())
                c0 = torch.stack([c00, zro, zro], dim=-1)  # NHW3
                c1 = torch.stack([zro, c11, zro], dim=-1)  # NHW3
                c2 = torch.stack([zro, zro, c22], dim=-1)  # NHW3
            else:
                # Calculate A^T * A
                c00 = A_c[..., 0] ** 2 + A_c[..., 1] ** 2 + A_c[..., 2] ** 2  # NHW
                c01 = A_c[..., 1] * A_c[..., 3] + A_c[..., 2] * A_c[..., 4]
                c02 = A_c[..., 2] * A_c[..., 5]
                c11 = A_c[..., 3] ** 2 + A_c[..., 4] ** 2
                c12 = A_c[..., 4] * A_c[..., 5]
                c22 = A_c[..., 5] ** 2
                c0 = torch.stack([c00, c01, c02], dim=-1)  # NHW3
                c1 = torch.stack([c01, c11, c12], dim=-1)  # NHW3
                c2 = torch.stack([c02, c12, c22], dim=-1)  # NHW3
            sigma_x = torch.stack([c0, c1, c2], dim=-1)  # NHW33

        # Data on which noise parameter estimation is based.
        if noise_params == NoiseValue.UNKNOWN_CONSTANT:
            # Global constant over the entire dataset.
            noise_est_out = self.l_params[Denoiser.ESTIMATED_SIGMA]
        elif noise_params == NoiseValue.UNKNOWN_VARIABLE:
            # Separate analysis network.
            param_est_net_out = self.models[Denoiser.SIGMA_ESTIMATOR](noisy_in)
            param_est_net_out = torch.mean(param_est_net_out, dim=(2, 3), keepdim=True)
            noise_est_out = param_est_net_out  # .type(torch.float64)

        # Cast remaining data into float64.
        # noisy_in = noisy_in.type(torch.float64)
        # noise_params_in = noise_params_in.type(torch.float64)

        # Remap noise estimate to ensure it is always positive and starts near zero.
        if noise_params != NoiseValue.KNOWN:
            # default pytorch vals: beta=1, threshold=20
            softplus = torch.nn.Softplus()  # yes this line is necessary, don't ask
            noise_est_out = softplus(noise_est_out - 4.0) + 1e-3

        # Distill noise parameters from learned/known data.
        if noise_style.startswith("gauss"):
            if noise_params == NoiseValue.KNOWN:
                noise_std = torch.max(
                    noise_params_in, torch.tensor(1e-3)  # , dtype=torch.float64)
                )  # N111
            else:
                noise_std = noise_est_out
        elif noise_style.startswith(
            "poisson"
        ):  # Simple signal-dependent Poisson approximation [Hasinoff 2012].
            if noise_params == NoiseValue.KNOWN:
                noise_std = (
                    torch.maximum(mu_x, torch.tensor(1e-3))  # , dtype=torch.float64))
                    / noise_params_in
                ) ** 0.5  # NCHW
            else:
                noise_std = (
                    torch.maximum(mu_x, torch.tensor(1e-3))  # , dtype=torch.float64))
                    * noise_est_out
                ) ** 0.5  # NCHW

        # Casts and vars.
        # noise_std = noise_std.type(torch.float64)
        noise_std = noise_std.to(self.device)
        # I = tf.eye(num_channels, batch_shape=[1, 1, 1], dtype=tf.float64)
        I = torch.eye(num_channels, device=self.device)  # dtype=torch.float64
        I = I.reshape(
            1, 1, 1, num_channels, num_channels
        )  # Creates the same shape as the tensorflow thing did, wouldn't work for other batch shapes
        Ieps = I * 1e-6
        zero64 = torch.tensor(0.0, device=self.device)  # , dtype=torch.float64

        # Helpers.
        def batch_mvmul(m, v):  # Batched (M * v).
            return torch.sum(m * v[..., None, :], dim=-1)

        def batch_vtmv(v, m):  # Batched (v^T * M * v).
            return torch.sum(v[..., :, None] * v[..., None, :] * m, dim=[-2, -1])

        def batch_vvt(v):  # Batched (v * v^T).
            return v[..., :, None] * v[..., None, :]

        # Negative log-likelihood loss and posterior mean estimation.
        if noise_style.startswith("gauss") or noise_style.startswith("poisson"):
            if num_channels == 1:
                sigma_n = noise_std ** 2  # N111 / N1HW
                sigma_y = sigma_x + sigma_n  # N1HW. Total variance.
                loss_out = ((noisy_in - mu_x) ** 2) / sigma_y + torch.log(
                    sigma_y
                )  # N1HW
                pme_out = (noisy_in * sigma_x + mu_x * sigma_n) / (
                    sigma_x + sigma_n
                )  # N1HW
                net_std_out = (sigma_x ** 0.5)[:, 0, ...]  # NHW
                noise_std_out = noise_std[:, 0, ...]  # N11 / NHW
                if noise_params != NoiseValue.KNOWN:
                    loss_out = loss_out - 0.1 * noise_std  # Balance regularization.
            else:
                # Training loss.
                noise_std_sqr = noise_std ** 2
                sigma_n = (
                    noise_std_sqr.permute(0, 2, 3, 1)[..., None] * I
                )  # NHWC1 * NHWCC = NHWCC
                if debug:
                    print("sigma_n device:", sigma_n.device)
                if debug:
                    print("sigma_x device:", sigma_x.device)
                sigma_y = (
                    sigma_x + sigma_n
                )  # NHWCC, total covariance matrix. Cannot be singular because sigma_n is at least a small diagonal.
                if debug:
                    print("sigma_y device:", sigma_y.device)
                sigma_y_inv = torch.inverse(sigma_y)  # NHWCC
                mu_x2 = mu_x.permute(0, 2, 3, 1)  # NHWC
                noisy_in2 = noisy_in.permute(0, 2, 3, 1)  # NHWC
                diff = noisy_in2 - mu_x2  # NHWC
                diff = -0.5 * batch_vtmv(diff, sigma_y_inv)  # NHW
                dets = torch.det(sigma_y)  # NHW
                dets = torch.max(
                    zero64, dets
                )  # NHW. Avoid division by zero and negative square roots.
                loss_out = 0.5 * torch.log(dets) - diff  # NHW
                if noise_params != NoiseValue.KNOWN:
                    loss_out = loss_out - 0.1 * torch.mean(
                        noise_std, dim=1
                    )  # Balance regularization.

                # Posterior mean estimate.
                sigma_x_inv = torch.inverse(sigma_x + Ieps)  # NHWCC
                sigma_n_inv = torch.inverse(sigma_n + Ieps)  # NHWCC
                pme_c1 = torch.inverse(sigma_x_inv + sigma_n_inv + Ieps)  # NHWCC
                pme_c2 = batch_mvmul(sigma_x_inv, mu_x2)  # NHWCC * NHWC -> NHWC
                pme_c2 = pme_c2 + batch_mvmul(sigma_n_inv, noisy_in2)  # NHWC
                pme_out = batch_mvmul(pme_c1, pme_c2)  # NHWC
                pme_out = pme_out.permute(0, 3, 1, 2)  # NCHW

                # Summary statistics.
                net_std_out = torch.max(zero64, torch.det(sigma_x)) ** (
                    1.0 / 6.0
                )  # NHW
                noise_std_out = torch.max(zero64, torch.det(sigma_n)) ** (
                    1.0 / 6.0
                )  # N11 / NHW

        # mu_x = mean of x
        # pme_out = posterior mean estimate
        # loss_out = loss
        # net_std_out = std estimate from nn
        # noise_std_out = predicted noise std?
        # return mu_x, pme_out, loss_out, net_std_out, noise_std_out
        loss_out = loss_out.view(loss_out.shape[0], -1).mean(1, keepdim=True)
        return {
            PipelineOutput.INPUTS: data,
            PipelineOutput.IMG_MU: mu_x,
            # PipelineOutput.IMG_PME: pme_out,
            PipelineOutput.IMG_DENOISED: pme_out,
            PipelineOutput.LOSS: loss_out,
            PipelineOutput.NOISE_STD_DEV: noise_std_out,
            PipelineOutput.MODEL_STD_DEV: net_std_out,
        }

    def state_dict(self, params_only: bool = False) -> Dict:
        state_dict = state_dict = super().state_dict()
        if not params_only:
            state_dict["cfg"] = self.cfg
        return state_dict

    @staticmethod
    def from_state_dict(state_dict: Dict) -> Denoiser:
        denoiser = Denoiser(state_dict["cfg"])
        denoiser.load_state_dict(state_dict, strict=False)
        return denoiser

    def config_name(self) -> str:
        return ssdn.cfg.config_name(self.cfg)
