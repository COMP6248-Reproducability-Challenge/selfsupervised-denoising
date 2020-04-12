import torchbearer
import torch
import ssdn

import torch.optim as optim
import torch.nn as nn

from torchvision.transforms import Compose, RandomCrop, ToTensor
from ssdn.datasets import UnlabelledImageFolderDataset, FixedLengthSampler
from ssdn.models.noise2noise import Noise2Noise

from torch.utils.data import DataLoader

from torchbearer import Trial

import torch.functional as F


def train_noise2noise():
    transform = Compose([RandomCrop(256, pad_if_needed=True)])
    transform
    training_dataset = UnlabelledImageFolderDataset(
        "C:/Scratch/COMP6202-DL-Reproducibility-Challenge/BSDS300/images/train/", transform=transform
    )
    test_dataset = UnlabelledImageFolderDataset(
        "C:/Scratch/COMP6202-DL-Reproducibility-Challenge/BSDS300/images/test/", transform=transform
    )
    loader_params = {
        "batch_size": 1,
        "num_workers": 1,
    }
    sampler = FixedLengthSampler(training_dataset, num_samples=None, shuffled=True)
    trainloader = DataLoader(training_dataset, sampler=sampler, **loader_params)
    sampler = FixedLengthSampler(test_dataset, num_samples=None, shuffled=True)
    testloader = DataLoader(test_dataset, sampler=sampler, **loader_params)

    # fix random seed for reproducibility
    seed = 7
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # build the model
    model = Noise2Noise()

    # define the loss function and the optimiser
    # TODO need to use signal-to-noise ratio somewhere?
    loss_function = nn.MSELoss()
    optimiser = optim.Adam(model.parameters())

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trial = Trial(model, optimiser, loss_function, metrics=['loss']).to(device)
    trial.with_generators(trainloader, test_generator=testloader)
    trial.run(epochs=1)
    results = trial.evaluate(data_key=torchbearer.TEST_DATA)
    print(results)


if __name__ == "__main__":
    train_noise2noise()
