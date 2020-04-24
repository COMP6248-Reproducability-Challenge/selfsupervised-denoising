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
    Pipeline,
)
import ssdn.utils.noise as noise

from torch import Tensor
from typing import Tuple, Union
from numbers import Number

from tqdm import tqdm
from ssdn.models import NoiseNetwork

from ssdn.denoiser import Denoiser

from typing import Dict


DEFAULT_CFG = {
    ConfigValue.TRAINING_ITERATIONS: 25,
    ConfigValue.MINIBATCH_SIZE: 2,
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


def train_noise2noise():
    cfg = DEFAULT_CFG
    torch.backends.cudnn.deterministic = True

    # from torchsummary import summary

    # network()
    # network = NoiseNetwork(blindspot=True)
    # # print(network.device)
    # summary(network, (3, 256, 256))
    # exit()

    dataset_dir = r"C:\dsj\deep_learning\coursework\git/BSDS300/images/"

    print("Loading training dataset...")
    training_dataset = UnlabelledImageFolderDataset(
        os.path.join(dataset_dir, r"D:\Downloads\ILSVRC2012_img_val"),
        transform=RandomCrop(cfg[ConfigValue.TRAINING_PATCH_SIZE], pad_if_needed=True),
        recursive=True,
    )
    pad_args = {
        "pad_uniform": False,
        "pad_multiple": 32,
        "square": cfg[ConfigValue.BLINDSPOT],
    }
    training_dataset = NoisyDataset(
        training_dataset,
        cfg[ConfigValue.NOISE_STYLE],
        cfg[ConfigValue.ALGORITHM],
        **pad_args
    )
    data = training_dataset[0]

    print("Loaded training dataset.")

    print("Loading test dataset...")
    test_dataset = UnlabelledImageFolderDataset(os.path.join(dataset_dir, "test/"))
    pad_args = {
        "pad_uniform": True,
        "pad_multiple": 32,
        "square": cfg[ConfigValue.BLINDSPOT],
    }
    test_dataset = NoisyDataset(
        test_dataset,
        cfg[ConfigValue.NOISE_STYLE],
        cfg[ConfigValue.ALGORITHM],
        **pad_args
    )
    data = test_dataset[0]

    print("Loaded test dataset.")

    loader_params = {
        "batch_size": 2, #cfg[ConfigValue.MINIBATCH_SIZE],
        "num_workers": 1,
        "pin_memory": True,
    }

    sampler = FixedLengthSampler(
        training_dataset,
        num_samples=cfg[ConfigValue.TRAINING_ITERATIONS],
        shuffled=False,
    )
    trainloader = DataLoader(training_dataset, sampler=sampler, **loader_params)
    sampler = FixedLengthSampler(
        test_dataset, num_samples=len(test_dataset) * 3, shuffled=False
    )
    testloader = DataLoader(test_dataset, sampler=sampler, **loader_params)
    data = next(iter(testloader))

    image = data[0]
    metadata = data[2]
    input_shape = metadata[NoisyDataset.Metadata.INPUT_SHAPE]
    print(input_shape)
    print(input_shape.shape)

    #ssdn.utils.show_tensor_image(image[1])
    unpadded = test_dataset.unpad_input(image, metadata)
    #ssdn.utils.show_tensor_image(unpadded[1])
    from ssdn.models import Shift2d
    shift = Shift2d([-50, -50])
    shifted = shift(image)
    #ssdn.utils.show_tensor_image(shifted[1])




    # fix random seed for reproducibility
    seed = 0
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # if True:
    #     model = torch.load("model.pt", map_location="cpu")
    #     data = next(iter(testloader))
    #     inp, ref = data[NoisyDataset.INPUT], data[NoisyDataset.REFERENCE]

    #     cleaned = model(inp)

    #     joined = torch.cat((inp, cleaned), axis=3)
    #     joined = joined.detach().numpy()[0]
    #     joined = np.clip(joined, 0, 1)
    #     joined = joined.transpose(1, 2, 0)

    #     im = Image.fromarray(np.uint8(joined * 255))
    #     im.show()
    # exit()
    if True:
        denoiser = Denoiser(cfg)
        t1 = time.time()
        # denoiser.train(trainloader)
        print("\n", time.time() - t1, "\n")

        # model = denoiser.get_model(Denoiser.MODEL, parallelised=False)
        data = next(iter(testloader))
        inp, ref = data[NoisyDataset.INPUT], data[NoisyDataset.REFERENCE]
        cleaned = denoiser(inp)
        joined = torch.cat((inp, cleaned), axis=3)
        print(joined.shape)
        ssdn.utils.show_tensor_image(joined[0])


if __name__ == "__main__":
    train_noise2noise()
