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
from ssdn.datasets.folder import NoisyUnlabelledImageFolderDataset
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
from ssdn.models.blindspot import NoiseNetwork


DEFAULT_CFG = {
    ConfigValue.TRAINING_ITERATIONS: 1000,
    ConfigValue.MINIBATCH_SIZE: 4,
    ConfigValue.BLINDSPOT: False,
    ConfigValue.ALGORITHM: NoiseAlgorithm.NOISE_TO_CLEAN,
    ConfigValue.NOISE_STYLE: "gauss0",
    ConfigValue.TRAINING_PATCH_SIZE: 256,
    ConfigValue.LEARNING_RATE: 3e-4,
    ConfigValue.LR_RAMPDOWN_FRACTION: 0.1,
    ConfigValue.LR_RAMPUP_FRACTION: 0.3,
    ConfigValue.EVAL_INTERVAL: 50
}


def mse2psnr(mse: Tensor, float_imgs: bool = True):
    high_val = torch.tensor(1.0) if float_imgs else torch.tensor(255)
    return 20 * torch.log10(high_val) - 10 * torch.log10(mse)

def calculate_psnr(img: Tensor, ref: Tensor, axis: int = None):
    mse = F.reduce_mean((img - ref) ** 2, axis=axis)
    return mse2psnr(mse, img.is_floating_point())


class Denoiser:
    def __init__(self, cfg=DEFAULT_CFG):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.cfg = cfg
        self.state = {}
        self.denoise_net = NoiseNetwork(blindspot=True).to(self.device)

        self.param_estimation_net = None
        # Dataloaders
        self.loss_fn = nn.MSELoss(reduction="none")
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
            outputs = self.run_pipeline(data, False)
            (dirty, _), (reference, _) = outputs[PipelineOutput.INPUTS]

            cleaned = outputs[PipelineOutput.IMG_DENOISED].cpu()
            joined = torch.cat((dirty, cleaned, reference), axis=3)
            joined = joined.detach().numpy()
            for i in range(joined.shape[0]):
                single_joined = joined[i]
                # joined = np.clip(joined, 0, 1)
                # psnr = 20 * math.log10(1) - 10 * math.log10(mse)
                single_joined = single_joined.transpose(1, 2, 0)
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
        # inp = inp[0].detach().numpy().transpose(1, 2, 0)
        # ref = ref[0].detach().numpy().transpose(1, 2, 0)
        # cleaned = cleaned[0].detach().numpy().transpose(1, 2, 0)

        # im = Image.fromarray(np.uint8(inp * 255))
        # im.show()
        # im = Image.fromarray(np.uint8(ref * 255))
        # im.show()
        # im = Image.fromarray(np.uint8(cleaned * 255))
        # im.show()
        # exit()

        loss = self.loss_fn(cleaned, ref)
        loss = loss.view(loss.shape[0], -1).mean(1, keepdim=True)
        if train:
            self.optimizer.zero_grad()
            # loss.backward()
            torch.mean(loss).backward()
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
        if self.cfg[ConfigValue.ALGORITHM] == NoiseAlgorithm.SELFSUPERVISED_DENOISING_MEAN_ONLY:
            return ((noisy_in, noisy_in_coeff), (noisy_in, noisy_in_coeff))
        raise NotImplementedError("Denoising algorithm not supported")


def train_noise2noise():
    torch.backends.cudnn.deterministic = True

    transform = Compose([RandomCrop(256, pad_if_needed=True)])
    dataset_dir = r"C:\dsj\deep_learning\coursework\git/BSDS300/images/"
    training_dataset = UnlabelledImageFolderDataset(
        os.path.join(dataset_dir, "train/"), transform=transform
    )
    test_dataset = UnlabelledImageFolderDataset(
        os.path.join(dataset_dir, "test/"), transform=transform
    )
    loader_params = {
        "batch_size": 1,
        "num_workers": 4,
        "pin_memory": True
    }
    sampler = FixedLengthSampler(training_dataset, num_samples=8, shuffled=False)
    trainloader = DataLoader(training_dataset, sampler=sampler, **loader_params)
    sampler = FixedLengthSampler(test_dataset, num_samples=1, shuffled=False)
    testloader = DataLoader(test_dataset, sampler=sampler, **loader_params)

    # fix random seed for reproducibility
    seed = 0
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # model = NoiseNetwork(blindspot=True)
    # for (image, indexes) in trainloader:
    #     cleaned = model(image)
    #     cleaned = cleaned.detach().numpy()[0]
    #     cleaned = cleaned.transpose(1, 2, 0)
    #     im = Image.fromarray(np.uint8(cleaned * 255))
    #     im.show()
    #     exit()
    
    # dirty, reference = next(iter(testloader))
    # cleaned = model2(dirty)

    # joined = torch.cat((dirty, cleaned, reference), axis=3)
    # joined = joined.detach().numpy()[0]
    # joined = np.clip(joined, 0, 1)
    # joined = joined.transpose(1, 2, 0)

    # im = Image.fromarray(np.uint8(joined * 255))
    # im.show()
    # print(model.conv.weight)
    # exit()

    seed = 0
    torch.manual_seed(seed)

    # denoiser = Denoiser()
    # t1 = time.time()
    # denoiser.train(trainloader)
    # print("\n", time.time() - t1, "\n")
    # # print(weights_b - weights_a)
    # model = denoiser.denoise_net
    # dirty, _ = next(iter(testloader))
    # cleaned = model(dirty)

    # joined = torch.cat((dirty, cleaned), axis=3)
    # joined = joined.detach().numpy()[0]
    # joined = np.clip(joined, 0, 1)
    # joined = joined.transpose(1, 2, 0)

    # im = Image.fromarray(np.uint8(joined * 255))
    # im.show()



    training_dataset = NoisyUnlabelledImageFolderDataset(
        os.path.join(dataset_dir, "train/"), transform=transform
    )
    test_dataset = NoisyUnlabelledImageFolderDataset(
        os.path.join(dataset_dir, "test/"), transform=transform
    )
    sampler = FixedLengthSampler(training_dataset, num_samples=100, shuffled=True)
    trainloader = DataLoader(training_dataset, sampler=sampler, **loader_params)
    sampler = FixedLengthSampler(test_dataset, num_samples=1, shuffled=False)
    testloader = DataLoader(test_dataset, sampler=sampler, **loader_params)


    # denoiser.evaluate(testloader)
    # exit()

    # build the model
    model = NoiseNetwork(blindspot=False)
    # cleaned = model(next(iter(testloader))[0])
    # cleaned = cleaned[0].detach().numpy().transpose(1, 2, 0)
    # im = Image.fromarray(np.uint8(cleaned * 255))
    # im.show()

    # exit()

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

    trial = Trial(
        model,
        optimizer,
        loss_function,
        # callbacks=[checkpointer],
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
    trial.with_generators(trainloader) # , val_generator=testloader
    
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
