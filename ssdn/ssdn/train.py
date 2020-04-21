from __future__ import annotations

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
from ssdn.datasets import UnlabelledImageFolderDataset, FixedLengthSampler, NoisyDataset

from ssdn.models.noise2noise import Noise2Noise

from torch.utils.data import Dataset, DataLoader

from torchbearer import Trial

import torch.functional as F
import math
from PIL import Image
from torchbearer.callbacks import Interval

from ssdn.params import (
    ConfigValue,
    StateValue,
    NoiseAlgorithm,
    PipelineOutput,
    Pipeline
)
import ssdn.utils.noise as noise

from torch import Tensor
from typing import Tuple, Union
from numbers import Number

from tqdm import tqdm
from ssdn.models.blindspot import NoiseNetwork

from ssdn.denoiser import Denoiser

from typing import Dict

DEFAULT_CFG = {
    ConfigValue.TRAINING_ITERATIONS: 25,
    ConfigValue.MINIBATCH_SIZE: 1,
    ConfigValue.BLINDSPOT: True,
    ConfigValue.PIPELINE: Pipeline.MSE,
    ConfigValue.ALGORITHM: NoiseAlgorithm.NOISE_TO_CLEAN,
    ConfigValue.NOISE_STYLE: "gauss30",
    ConfigValue.TRAINING_PATCH_SIZE: 256,
    ConfigValue.LEARNING_RATE: 3e-4,
    ConfigValue.LR_RAMPDOWN_FRACTION: 0.1,
    ConfigValue.LR_RAMPUP_FRACTION: 0.3,
    ConfigValue.EVAL_INTERVAL: 50,
}


def mse2psnr(mse: Tensor, float_imgs: bool = True):
    high_val = torch.tensor(1.0) if float_imgs else torch.tensor(255)
    return 20 * torch.log10(high_val) - 10 * torch.log10(mse)


def calculate_psnr(img: Tensor, ref: Tensor, axis: int = None):
    mse = F.reduce_mean((img - ref) ** 2, axis=axis)
    return mse2psnr(mse, img.is_floating_point())


def train_noise2noise():
    cfg = DEFAULT_CFG

    torch.backends.cudnn.deterministic = True

    transform = Compose(
        [RandomCrop(cfg[ConfigValue.TRAINING_PATCH_SIZE], pad_if_needed=True)]
    )
    dataset_dir = r"C:\dsj\deep_learning\coursework\git/BSDS300/images/"
    training_dataset = UnlabelledImageFolderDataset(
        os.path.join(dataset_dir, r"D:\Downloads\ILSVRC2012_img_val"),
        transform=transform,
    )
    test_dataset = UnlabelledImageFolderDataset(
        os.path.join(dataset_dir, "test/"), transform=transform
    )

    training_dataset = NoisyDataset(
        training_dataset, cfg[ConfigValue.NOISE_STYLE], cfg[ConfigValue.ALGORITHM]
    )
    test_dataset = NoisyDataset(
        test_dataset, cfg[ConfigValue.NOISE_STYLE], cfg[ConfigValue.ALGORITHM]
    )

    loader_params = {
        "batch_size": cfg[ConfigValue.MINIBATCH_SIZE],
        "num_workers": 4,
        "pin_memory": True,
    }

    sampler = FixedLengthSampler(
        training_dataset,
        num_samples=cfg[ConfigValue.TRAINING_ITERATIONS],
        shuffled=False,
    )
    trainloader = DataLoader(training_dataset, sampler=sampler, **loader_params)
    sampler = FixedLengthSampler(test_dataset, num_samples=1, shuffled=False)
    testloader = DataLoader(test_dataset, sampler=sampler, **loader_params)

    # fix random seed for reproducibility
    seed = 0
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    if True:
        denoiser = Denoiser(cfg)
        t1 = time.time()
        denoiser.train(trainloader)
        print("\n", time.time() - t1, "\n")

        model = denoiser.denoise_net
        data = next(iter(testloader))
        inp, ref = data[NoisyDataset.INPUT], data[NoisyDataset.REFERENCE]

        cleaned = model(inp)

        joined = torch.cat((inp, cleaned), axis=3)
        joined = joined.detach().numpy()[0]
        joined = np.clip(joined, 0, 1)
        joined = joined.transpose(1, 2, 0)

        im = Image.fromarray(np.uint8(joined * 255))
        im.show()

    # fix random seed for reproducibility
    seed = 0
    torch.manual_seed(seed)

    training_dataset.enable_metadata = False
    test_dataset.enable_metadata = False
    sampler = FixedLengthSampler(
        training_dataset,
        num_samples=cfg[ConfigValue.TRAINING_ITERATIONS],
        shuffled=False,
    )
    trainloader = DataLoader(training_dataset, sampler=sampler, **loader_params)
    sampler = FixedLengthSampler(test_dataset, num_samples=1, shuffled=False)
    testloader = DataLoader(test_dataset, sampler=sampler, **loader_params)

    # build the model
    model = NoiseNetwork(blindspot=False)
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

    trial = Trial(
        model,
        optimizer,
        loss_function,
        callbacks=[checkpointer],
        metrics=["loss"],
        verbose=2,
    ).to(device)

    # Load existing model
    models = glob.glob(os.path.join(model_dir, "*.weights"))
    if True and len(models) > 0:
        model_path = models[-1]
        state_dict = torch.load(model_path, map_location=device)
        trial.load_state_dict(state_dict)
        print("Loaded: ", model_path)
    trial.with_generators(trainloader)  # , val_generator=testloader

    t1 = time.time()
    trial.run(epochs=1)
    print("\n", time.time() - t1, "\n")

    # results = trial.evaluate(data_key=torchbearer.TEST_DATA)
    mse_model = nn.MSELoss()
    os.makedirs("results", exist_ok=True)
    for i, (dirty, reference) in enumerate(testloader):
        cleaned = model.forward(dirty.to(device)).cpu()
        joined = torch.cat((dirty, cleaned, reference), axis=3)
        joined = joined.detach().numpy()[0]
        joined = np.clip(joined, 0, 1)
        joined = joined.transpose(1, 2, 0)

        mse = mse_model(cleaned, reference)
        psnr = 20 * math.log10(1) - 10 * math.log10(mse)
        print("{},{},{}".format(i, mse, psnr))
        im = Image.fromarray(np.uint8(joined * 255))
        im.show()
        im.save("results/{}.jpeg".format(i))


if __name__ == "__main__":
    train_noise2noise()
