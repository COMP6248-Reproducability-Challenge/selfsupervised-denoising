import os
import torchbearer
import torch
import ssdn
import numpy as np
import torch.optim as optim
from torch.optim import Optimizer
import torch.nn as nn
import glob
import time

from torchvision.transforms import Compose, RandomCrop, ToTensor
from ssdn.datasets import UnlabelledImageFolderDataset, FixedLengthSampler
from ssdn.models.noise2noise import Noise2Noise

from torch.utils.data import DataLoader

from torchbearer import Trial

import torch.functional as F
import math
from PIL import Image
from torchbearer.callbacks import Interval

from ssdn.params import ConfigValue, StateValue, NoiseAlgorithm, PipelineOutput
import ssdn.utils.noise as noise

from torch import Tensor
from typing import Tuple, Union
from numbers import Number

from tqdm import tqdm

DEFAULT_CFG = {
    ConfigValue.TRAINING_ITERATIONS: 1000,
    ConfigValue.MINIBATCH_SIZE: 1,
    ConfigValue.BLINDSPOT: False,
    ConfigValue.ALGORITHM: NoiseAlgorithm.NOISE_TO_CLEAN,
    ConfigValue.NOISE_STYLE: "gauss50",
    ConfigValue.TRAINING_PATCH_SIZE: 256,
    ConfigValue.LEARNING_RATE: 3e-4,
    ConfigValue.LR_RAMPDOWN_FRACTION: 0.1,
    ConfigValue.LR_RAMPUP_FRACTION: 0.3,
    ConfigValue.EVAL_INTERVAL: 50
}


def show_chw():
    pass


def calculate_psnr(img: Tensor, ref: Tensor, axis: int = None):
    a, b = [clip_img(x) for x in [img, ref]]
    a, b = [tf.cast(x, tf.float32) for x in [a, b]]
    x = tf.reduce_mean((a - b) ** 2, axis=axis)
    high_val = 1.0 if img.is_floating_point() else 255
    return 20 * torch.log10(high_val) - 10 * torch.log10(mse)


def mse2psnr(mse: Tensor, float_imgs: bool = True):
    high_val = torch.tensor(1.0) if float_imgs else torch.tensor(255)
    return 20 * torch.log10(high_val) - 10 * torch.log10(mse)


class Denoiser:
    def __init__(self, cfg=DEFAULT_CFG):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.cfg = cfg
        self.state = {}
        self.denoise_net = Noise2Noise().to(self.device)
        self.param_estimation_net = None
        # Dataloaders
        self.loss_fn = nn.MSELoss(reduction="mean")
        self._optimizer = optim.Adam(self.denoise_net.parameters(), betas=[0.9, 0.99])
        

    def train(self, trainloader: DataLoader, testloader: DataLoader = None):

        if StateValue.ITERATION not in self.state:
            self.state[StateValue.ITERATION] = 0
        self.cfg[ConfigValue.TRAINING_ITERATIONS] = len(trainloader)
        prog_bar = tqdm(
            total=self.cfg[ConfigValue.TRAINING_ITERATIONS],
            initial=self.state[StateValue.ITERATION],
        )
        while (
            self.state[StateValue.ITERATION] < self.cfg[ConfigValue.TRAINING_ITERATIONS]
        ):
            data, indexes = next(iter(trainloader))
            outputs = self.run_pipeline(data, True)
            prog_bar.update()
            self.state[StateValue.ITERATION] += 1 # actual batch size
            if (
                testloader is not None
                and (self.state[StateValue.ITERATION]
                % self.cfg[ConfigValue.EVAL_INTERVAL]) == 0
            ):
                self.evaluate(testloader)

    def evaluate(self, testloader: DataLoader):
        for data, indexes in testloader:
            outputs = self.run_pipeline(data, testloader)
            (dirty, _), (reference, _) = outputs[PipelineOutput.INPUTS]

            cleaned = outputs[PipelineOutput.IMG_DENOISED].cpu()
            joined = torch.cat((dirty, cleaned, reference), axis=3)
            joined = joined.detach().numpy()
            for i in range(joined.shape[0]):
                single_joined = joined[i]
                # joined = np.clip(joined, 0, 1)
                single_joined = single_joined.transpose(1, 2, 0)
                # psnr = 20 * math.log10(1) - 10 * math.log10(mse)
                # print("{},{},{}".format(i, mse, psnr))
                im = Image.fromarray(np.uint8(single_joined * 255))
                im.save("results/{}.jpeg".format(indexes[i]))

    def run_pipeline(self, data: Tensor, train: bool, **kwargs):
        if self.cfg[ConfigValue.BLINDSPOT]:
            return self._blindspot_pipeline(data, train, **kwargs)
        else:
            return self._simple_pipeline(data, train, **kwargs)

    def _blindspot_pipeline(self, data: Tensor, train: bool, **kwargs):
        pass

    def _simple_pipeline(self, data: Tensor, train: bool, **kwargs):
        inputs = self.prepare_input(data)
        (inp, _), (ref, _) = inputs
        inp = inp.to(self.device)
        ref = ref.to(self.device)
        if train:
            inp.requires_grad = True
            ref.requires_grad = True

        cleaned = self.denoise_net(inp)
        loss = self.loss_fn(inp, ref)
        # loss = loss.view(loss.shape[0], -1).mean(1, keepdim=True)
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            # torch.mean(loss).backward()
            self.optimizer.step()

        return {
            PipelineOutput.INPUTS: inputs,
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
        # for param_group in self._optimizer.param_groups:
        #     param_group["lr"] = learning_rate
        return self._optimizer

    def prepare_input(
        self, clean_img_batch: Tensor
    ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        # Helper function to fix coefficient shape to [n, 1, 1, 1] shape
        def broadcast_coeffs(imgs: Tensor, coeffs: Union[Tensor, Number]):
            return torch.zeros((imgs.shape[0], 1, 1, 1)) + coeffs

        noise_style = self.cfg[ConfigValue.NOISE_STYLE]
        # Create the noisy input images
        noisy_in, noisy_in_coeff = noise.add_style(clean_img_batch, noise_style)
        noisy_in_coeff = broadcast_coeffs(noisy_in, noisy_in_coeff)

        # N2C requires noisy input and clean reference images
        if self.cfg[ConfigValue.ALGORITHM] == NoiseAlgorithm.NOISE_TO_CLEAN:
            return ((noisy_in, noisy_in_coeff), (clean_img_batch, None))
        # N2N requires noisy input and noisy reference images
        if self.cfg[ConfigValue.ALGORITHM] == NoiseAlgorithm.NOISE_TO_NOISE:
            noisy_ref, noisy_ref_coeff = noise.add_style(clean_img_batch, noise_style)
            noisy_ref_coeff = broadcast_coeffs(noisy_in, noisy_in_coeff)
            return ((noisy_in, noisy_in_coeff), (noisy_ref, noisy_ref_coeff))
        # SSDN requires noisy input and no reference images
        if self.cfg[ConfigValue.ALGORITHM] == NoiseAlgorithm.SELFSUPERVISED_DENOISING:
            return ((noisy_in, noisy_in_coeff), None)

        raise NotImplementedError("Denoising algorithm not supported")


def train_noise2noise():

    transform = Compose([RandomCrop(256, pad_if_needed=True)])
    dataset_dir = "C:/Scratch/COMP6202-DL-Reproducibility-Challenge/BSDS300/images/"
    training_dataset = UnlabelledImageFolderDataset(
        os.path.join(dataset_dir, "train/"), transform=transform
    )
    test_dataset = UnlabelledImageFolderDataset(
        os.path.join(dataset_dir, "test/"), transform=transform
    )
    loader_params = {
        "batch_size": 4,
        "num_workers": 1,
        "pin_memory": True
    }
    sampler = FixedLengthSampler(training_dataset, num_samples=1000, shuffled=False)
    trainloader = DataLoader(training_dataset, sampler=sampler, **loader_params)
    sampler = FixedLengthSampler(test_dataset, num_samples=5, shuffled=False)
    testloader = DataLoader(test_dataset, sampler=sampler, **loader_params)

    # fix random seed for reproducibility
    seed = 7
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    denoiser = Denoiser()
    denoiser.train(trainloader, testloader=testloader)
    exit()
    for (data, indexes) in trainloader:
        (inp, inp_coeffs), (ref, ref_coeffs) = denoiser.prepare_input(data)
        print(inp_coeffs, ref_coeffs)
        exit()

    # build the model
    model = Noise2Noise()

    # define the loss function and the optimizer
    # TODO need to use signal-to-noise ratio somewhere?
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    checkpointer = Interval(
        os.path.join(model_dir, "{epoch:02d}_{t:010d}.weights"),
        period=1,
        on_batch=False,
    )  # t == BATCH
    models = glob.glob(os.path.join(model_dir, "*.weights"))
    # models = sorted(models, key=lambda x: os.path.basename(x).split(".weights")[0])
    trial = Trial(
        model,
        optimizer,
        loss_function,
        callbacks=[checkpointer],
        metrics=["loss"],
        verbose=2,
    ).to(device)
    if True and len(models) > 0:
        model_path = models[-1]
        state_dict = torch.load(model_path, map_location=device)
        trial.load_state_dict(state_dict)
        print("Loaded: ", model_path)
    trial.with_generators(trainloader, val_generator=testloader)
    trial.run(epochs=1)
    # results = trial.evaluate(data_key=torchbearer.TEST_DATA)
    mse_model = nn.MSELoss()
    os.makedirs("results", exist_ok=True)
    for i, (dirty, reference) in enumerate(testloader):
        cleaned = model.forward(dirty.to(device)).cpu()
        mse = mse_model(cleaned, reference)
        joined = torch.cat((dirty, cleaned, reference), axis=3)
        joined = joined.detach().numpy()[0]
        joined = np.clip(joined, 0, 1)
        joined = joined.transpose(1, 2, 0)
        psnr = 20 * math.log10(1) - 10 * math.log10(mse)
        print("{},{},{}".format(i, mse, psnr))
        im = Image.fromarray(np.uint8(joined * 255))
        im.save("results/{}.jpeg".format(i))


if __name__ == "__main__":
    train_noise2noise()
