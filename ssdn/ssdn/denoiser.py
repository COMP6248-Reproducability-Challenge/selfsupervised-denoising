from __future__ import annotations


import torch
import torch.nn as nn

from torch import Tensor

from ssdn.params import (
    ConfigValue,
    NoiseAlgorithm,
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

        # Store the denoiser configuration. Use the algorithm to infer
        # pipeline and model settings if not already set
        self.cfg = cfg
        Denoiser.infer_cfg(cfg)

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
        # Create general denoising model
        self.add_model(
            Denoiser.MODEL, NoiseNetwork(blindspot=self.cfg[ConfigValue.BLINDSPOT])
        )
        # Create separate model for variable parameter estimation
        if (
            self.cfg[ConfigValue.PIPELINE] == Pipeline.SSDN
            and self.cfg[ConfigValue.NOISE_VALUE] == NoiseValue.UNKNOWN_VARIABLE
        ):
            self.add_model(
                Denoiser.SIGMA_ESTIMATOR,
                NoiseNetwork(out_channels=1, blindspot=False, zero_output_weights=True),
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

    def _ssdn_pipeline(self, data: List, **kwargs) -> Dict:
        return {
            PipelineOutput.INPUTS: data,
            PipelineOutput.LOSS: torch.tensor(1.23, requires_grad=True),
            PipelineOutput.IMG_MU: data[NoisyDataset.INPUT],
            PipelineOutput.IMG_DENOISED: data[NoisyDataset.INPUT],
            PipelineOutput.NOISE_STD_DEV: torch.tensor(0.252),
            PipelineOutput.MODEL_STD_DEV: torch.tensor(23.242),
        }
        # Equivalent of blindspot_pipeline
        raise NotImplementedError()

    def state_dict(self, params_only: bool = False) -> Dict:
        state_dict = state_dict = super().state_dict()
        if not params_only:
            state_dict["cfg"] = self.cfg
        return state_dict

    @staticmethod
    def from_state_dict(state_dict: str) -> Denoiser:
        denoiser = Denoiser(state_dict["cfg"])
        denoiser.load_state_dict(state_dict, strict=False)
        return denoiser

    def config_name(self) -> str:
        return Denoiser._config_name(self.cfg)

    @staticmethod
    def infer_cfg(cfg: Dict) -> Dict:
        if cfg.get(ConfigValue.PIPELINE, None) is None:
            cfg[ConfigValue.PIPELINE] = Denoiser.infer_pipeline(
                cfg[ConfigValue.ALGORITHM]
            )
        if cfg.get(ConfigValue.BLINDSPOT, None) is None:
            cfg[ConfigValue.BLINDSPOT] = Denoiser.infer_blindspot(
                cfg[ConfigValue.ALGORITHM]
            )
        return cfg

    @staticmethod
    def infer_pipeline(algorithm: NoiseAlgorithm) -> Pipeline:
        if algorithm in [NoiseAlgorithm.SELFSUPERVISED_DENOISING]:
            return Pipeline.SSDN
        elif algorithm in [
            NoiseAlgorithm.SELFSUPERVISED_DENOISING_MEAN_ONLY,
            NoiseAlgorithm.NOISE_TO_NOISE,
            NoiseAlgorithm.NOISE_TO_CLEAN,
        ]:
            return Pipeline.MSE
        else:
            raise NotImplementedError("Algorithm does not have a default pipeline.")

    @staticmethod
    def infer_blindspot(algorithm: NoiseAlgorithm):
        if algorithm in [
            NoiseAlgorithm.SELFSUPERVISED_DENOISING,
            NoiseAlgorithm.SELFSUPERVISED_DENOISING_MEAN_ONLY,
        ]:
            return True
        elif algorithm in [
            NoiseAlgorithm.NOISE_TO_NOISE,
            NoiseAlgorithm.NOISE_TO_CLEAN,
        ]:
            return False
        else:
            raise NotImplementedError("Not known if algorithm requires blindspot.")

    @staticmethod
    def _config_name(cfg: Dict) -> str:
        cfg = Denoiser.infer_cfg(cfg)
        config_lst = [cfg[ConfigValue.ALGORITHM].value]

        # Check if pipeline cannot be inferred
        inferred_pipeline = Denoiser.infer_pipeline(cfg[ConfigValue.ALGORITHM])
        if cfg[ConfigValue.PIPELINE] != inferred_pipeline:
            config_lst += [cfg[ConfigValue.PIPELINE].value + "_pipeline"]
        # Check if blindspot enable cannot be inferred
        inferred_blindspot = Denoiser.infer_blindspot(cfg[ConfigValue.ALGORITHM])
        if cfg[ConfigValue.BLINDSPOT] != inferred_blindspot:
            config_lst += [
                "blindspot" if cfg[ConfigValue.BLINDSPOT] else "blindspot_disabled"
            ]
        # Add noise information
        config_lst += [cfg[ConfigValue.NOISE_STYLE]]
        if cfg[ConfigValue.PIPELINE] in [Pipeline.SSDN]:
            config_lst += ["sigma_" + cfg[ConfigValue.NOISE_VALUE].value]

        if cfg[ConfigValue.IMAGE_CHANNELS] == 1:
            config_lst += ["mono"]
        if cfg[ConfigValue.PIPELINE] in [Pipeline.SSDN]:
            if cfg[ConfigValue.DIAGONAL_COVARIANCE]:
                config_lst += ["diag"]

        config_name = "-".join(config_lst)
        return config_name
