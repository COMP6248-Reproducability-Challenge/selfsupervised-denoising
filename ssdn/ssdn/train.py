import os
import torchbearer
import torch
import ssdn
import numpy as np
import torch.optim as optim
import torch.nn as nn
import glob

from torchvision.transforms import Compose, RandomCrop, ToTensor
from ssdn.datasets import UnlabelledImageFolderDataset, FixedLengthSampler
from ssdn.models.noise2noise import Noise2Noise

from torch.utils.data import DataLoader

from torchbearer import Trial

import torch.functional as F
import math
from PIL import Image
from torchbearer.callbacks import Interval


def train_noise2noise():
    transform = Compose([RandomCrop(256, pad_if_needed=True)])
    transform
    training_dataset = UnlabelledImageFolderDataset(
        "C:/dsj/deep_learning/coursework/git/BSDS300/images/train/", transform=transform
    )
    test_dataset = UnlabelledImageFolderDataset(
        "C:/dsj/deep_learning/coursework/git/BSDS300/images/test/"
    )
    loader_params = {
        "batch_size": 1,
        "num_workers": 1,
    }
    sampler = FixedLengthSampler(training_dataset, num_samples=100, shuffled=False)
    trainloader = DataLoader(training_dataset, sampler=sampler, **loader_params)
    sampler = FixedLengthSampler(test_dataset, num_samples=5, shuffled=False)
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
        optimiser,
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
