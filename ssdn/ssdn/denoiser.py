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
    Pipeline
)
from ssdn.models.blindspot import NoiseNetwork
from ssdn.datasets import NoisyDataset

from typing import Dict
from tqdm import tqdm

class Denoiser:

    def __init__(self, cfg: Dict):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.cfg = cfg
        self.state = {}
        self.denoise_net = NoiseNetwork(blindspot=cfg[ConfigValue.BLINDSPOT])
        self.param_estimation_net = None
        # Dataloaders
        self.loss_fn = nn.MSELoss(reduction="mean")
        self._optimizer = optim.Adam(self.denoise_net.parameters(), betas=[0.9, 0.99])

        for network in [self.denoise_net, self.param_estimation_net]:
            if network is not None:
                network.to(self.device)

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
            self.optimizer.zero_grad()
            inp.requires_grad = True
            ref.requires_grad = True

        cleaned = self.denoise_net(inp)
        loss = self.loss_fn(cleaned, ref)
        # Reduce to batch losses if not already reduced
        if len(loss.shape) > 0:
            loss = loss.view(loss.shape[0], -1).mean(1, keepdim=True)

        if train:
            # Train using a single loss
            torch.mean(loss).backward()
            self.optimizer.step()

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
