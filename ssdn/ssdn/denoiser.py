from __future__ import annotations


import torch
import torch.optim as optim
import torch.nn as nn
import itertools

import ssdn

from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import Tensor

from ssdn.params import (
    ConfigValue,
    StateValue,
    NoiseAlgorithm,
    PipelineOutput,
    Pipeline,
    NoiseValue,
)

from ssdn.models import NoiseNetwork
from ssdn.datasets import NoisyDataset

from typing import Dict
from tqdm import tqdm


class Denoiser:

    MODEL = "denoiser_model"
    PARAM_ESTIMATION = "param_estimation_model"
    ESTIMATED_PARAM = "estimated_param"

    def __init__(self, cfg: Dict, state: Dict = {}, device: str = None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.state = state

        # Models to use during training, these can be parallelised
        self.models = nn.ModuleDict()
        # References to models that are guaranteed to be device independent
        self._models = nn.ModuleDict()
        # Initialise networks using current configuration
        self.init_networks()

        # Learnable parameters
        self.parameters = nn.ParameterDict()
        self.init_parameters()

        # Refresh optimiser for parameters
        self.init_optimiser()

        # Initialise training state
        if StateValue.INITIALISED not in self.state:
            self.init_state()

            self.testloader = None
            self.trainloader = None

    def init_state(self):
        self.state[StateValue.INITIALISED] = True
        self.state[StateValue.ITERATION] = 0
        self.state[StateValue.EVAL_MODE] = False
        self.state[StateValue.HISTORY] = {}

    def init_networks(self):
        # Create general denoising model
        self.add_model(
            Denoiser.MODEL, NoiseNetwork(blindspot=self.cfg[ConfigValue.BLINDSPOT])
        )
        # Create separate model for parameter estimation
        if (
            self.cfg[ConfigValue.PIPELINE] == Pipeline.SSDN
            and self.cfg[ConfigValue.NOISE_VALUE_STATE] == NoiseValue.VARIABLE_UNKNOWN
        ):
            # TODO: Do we need to add a flag for zeroing last set of biases
            self.add_model(
                Denoiser.PARAM_ESTIMATION, NoiseNetwork(out_channels=1, blindspot=False)
            )

    def init_parameters(self):
        # if (
        #     self.cfg[ConfigValue.PIPELINE] == Pipeline.SSDN
        #     and self.cfg[ConfigValue.NOISE_VALUE_STATE] == NoiseValue.CONSTANT_UNKNOWN
        # ):
        init_value = torch.zeros((1, 1, 1, 1))
        self.parameters[Denoiser.ESTIMATED_PARAM] = nn.Parameter(init_value)

    def init_optimiser(self):
        parameters = itertools.chain(
            self.models.parameters(), self.parameters.parameters()
        )
        self._optimizer = optim.Adam(parameters, betas=[0.9, 0.99])

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

    def eval(self):
        self.models.eval()
        # for model in self.models.values():
        #     self.network.eval()

    def train(self, trainloader: DataLoader, testloader: DataLoader = None):
        self.cfg[ConfigValue.TRAIN_ITERATIONS] = len(trainloader)
        prog_bar = tqdm(
            total=self.cfg[ConfigValue.TRAIN_ITERATIONS],
            initial=self.state[StateValue.ITERATION],
        )
        data_itr = iter(trainloader)
        while (
            self.state[StateValue.ITERATION] < self.cfg[ConfigValue.TRAIN_ITERATIONS]
        ):
            data = next(data_itr)
            outputs = self.run_pipeline(data, True)
            prog_bar.update()
            self.state[StateValue.ITERATION] += 1  # actual batch size
            if (
                testloader is not None
                and (
                    self.state[StateValue.ITERATION]
                    % self.cfg[ConfigValue.EVAL_INTERVAL]
                )
                == 0
            ):
                self.evaluate(testloader)

    # def evaluate(self, testloader: DataLoader):
    #     for data, indexes in testloader:
    #         outputs = self.run_pipeline(data, False)
    #         (dirty, _), (reference, _) = outputs[PipelineOutput.INPUTS]

    #         cleaned = outputs[PipelineOutput.IMG_DENOISED].cpu()
    #         joined = torch.cat((dirty, cleaned, reference), axis=3)
    #         joined = joined.detach().numpy()
    #         for i in range(joined.shape[0]):
    #             single_joined = joined[i]
    #             # joined = np.clip(joined, 0, 1)
    #             # psnr = 20 * math.log10(1) - 10 * math.log10(mse)
    #             single_joined = single_joined.transpose(1, 2, 0)
    #             # print("{},{},{}".format(i, mse, psnr))
    #             im = Image.fromarray(np.uint8(single_joined * 255))
    #             im.save("results/{}.jpeg".format(indexes[i]))

    def run_pipeline(self, data: Tensor, train: bool, **kwargs):
        if self.cfg[ConfigValue.PIPELINE] == Pipeline.MSE:
            return self._mse_pipeline(data, train, **kwargs)
        elif self.cfg[ConfigValue.PIPELINE] == Pipeline.SSDN:
            return self._ssdn_pipeline(data, train, **kwargs)
        else:
            raise NotImplementedError("Unsupported processing pipeline")

    def _ssdn_pipeline(self, data: Tensor, train: bool, **kwargs) -> Dict:
        # Equivalent of blindspot_pipeline
        pass

    def _mse_pipeline(self, data: Tensor, train: bool, **kwargs) -> Dict:
        # Equivalent of simple_pipeline
        inp, ref = data[NoisyDataset.INPUT], data[NoisyDataset.REFERENCE]
        inp, ref = inp.to(self.device), ref.to(self.device)

        if train:
            optimizer = self.optimizer
            optimizer.zero_grad()
            inp.requires_grad = True
            ref.requires_grad = True

        cleaned = self.models[Denoiser.MODEL](inp)
        # Do MSE but preserve individual loss per batch
        loss = nn.MSELoss(reduction="none")(cleaned, ref)
        loss = loss.view(loss.shape[0], -1).mean(1, keepdim=True)

        if train:
            # Train using average loss across batch
            torch.mean(loss).backward()
            optimizer.step()

        return {
            PipelineOutput.INPUTS: data,
            PipelineOutput.IMG_DENOISED: cleaned,
            PipelineOutput.LOSS: loss,
        }

    @property
    def optimizer(self) -> Optimizer:
        learning_rate = ssdn.utils.compute_ramped_lrate(
            self.state[StateValue.ITERATION],
            self.cfg[ConfigValue.TRAIN_ITERATIONS],
            self.cfg[ConfigValue.LR_RAMPDOWN_FRACTION],
            self.cfg[ConfigValue.LR_RAMPUP_FRACTION],
            self.cfg[ConfigValue.LEARNING_RATE],
        )
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = learning_rate
        return self._optimizer

    def state_dict(self) -> Dict:
        state_dict = {}
        # Add model weights to state dict
        state_dict["models"] = {}
        for model_name, model in self._models.items():
            state_dict["models"][model_name] = model.state_dict()
        # Add learnable parameters to state dict
        state_dict["parameters"] = {}
        for parameter_name, parameter in self.parameters.items():
            state_dict["parameters"][parameter_name] = parameter
        # Store for maintaining training
        state_dict["cfg"] = self.cfg
        state_dict["state"] = self.state
        state_dict["optimizer"] = self.optimizer.state_dict()
        state_dict["rng"] = torch.get_rng_state()
        return state_dict

    @staticmethod
    def from_state_dict(state_dict: str) -> Denoiser:
        # Initialise RNG
        torch.set_rng_state(state_dict["rng"])
        denoiser = Denoiser(state_dict["cfg"], state_dict["state"])
        # Load model weights
        for model_name, model_state_dict in state_dict["models"].items():
            model = denoiser.get_model(model_name, parallelised=False)
            model.load_state_dict(model_state_dict)
        # Load learnable parameters
        for parameter_name, value in state_dict["parameters"].items():
            denoiser.parameters[parameter_name] = nn.Parameter(value)

        # Load optimiser
        denoiser.optimizer.load_state_dict(state_dict["optimizer"])
        return denoiser

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path: str) -> Denoiser:
        return Denoiser.from_state_dict(torch.load(path))

    def __call__(self, data: Tensor) -> Tensor:
        """Pass an input into the denoiser for inference. This will not train
        the network. Inference will be applied using current model state with
        the configured pipeline.

        Args:
            data (Tensor): Image or batch of images to denoise in BCHW format.

        Returns:
            Tensor: Denoised image or images.
        """
        model = self.models[Denoiser.MODEL]
        data = data.to(self.device)
        return model(data)
