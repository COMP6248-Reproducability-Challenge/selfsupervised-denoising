import torch
import torch.optim as optim
import torch.nn as nn

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
)
from ssdn.models.blindspot import NoiseNetwork
from ssdn.datasets import NoisyDataset

from typing import Dict
from tqdm import tqdm


class Denoiser:

    MODEL = "denoiser_model"
    PARAM_ESTIMATION = "param_estimation_model"

    def __init__(self, cfg: Dict, device: str = None):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.state = {}

        # References to models that are guaranteed to be device independent
        self._models = nn.ModuleDict()
        # Models to use during training, these can be parallelised
        self.models = nn.ModuleDict()
        self.init_networks()

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

    def init_networks(self):
        # Create general denoising model
        self.add_model(
            Denoiser.MODEL, NoiseNetwork(blindspot=self.cfg[ConfigValue.BLINDSPOT])
        )
        # Create separate model for parameter estimation
        if self.cfg[ConfigValue.PIPELINE] == Pipeline.SSDN:
            # TODO: Do we need to add a flag for zeroing last set of biases
            self.add_model(
                Denoiser.PARAM_ESTIMATION, NoiseNetwork(out_channels=1, blindspot=False)
            )
        # Refresh optimiser for new network parameters
        self.init_optimiser()

    def init_optimiser(self):
        self._optimizer = optim.Adam(self.models.parameters(), betas=[0.9, 0.99])

    def eval(self):
        self.models.eval()
        # for model in self.models.values():
        #     self.network.eval()

    def train(self, trainloader: DataLoader, testloader: DataLoader = None):
        if StateValue.ITERATION not in self.state:
            self.state[StateValue.ITERATION] = 0
        self.cfg[ConfigValue.TRAINING_ITERATIONS] = len(trainloader)
        prog_bar = tqdm(
            total=self.cfg[ConfigValue.TRAINING_ITERATIONS],
            initial=self.state[StateValue.ITERATION],
        )
        data_itr = iter(trainloader)
        while (
            self.state[StateValue.ITERATION] < self.cfg[ConfigValue.TRAINING_ITERATIONS]
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

    def _ssdn_pipeline(self, data: Tensor, train: bool, **kwargs):
        # Equivalent of blindspot_pipeline
        pass

    def _mse_pipeline(self, data: Tensor, train: bool, **kwargs):
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
            self.cfg[ConfigValue.TRAINING_ITERATIONS],
            self.cfg[ConfigValue.LR_RAMPDOWN_FRACTION],
            self.cfg[ConfigValue.LR_RAMPUP_FRACTION],
            self.cfg[ConfigValue.LEARNING_RATE],
        )
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = learning_rate
        return self._optimizer

    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    # def eval(self):
    #     raise NotImplementedError()

    # def train(self):
    #     raise NotImplementedError()
