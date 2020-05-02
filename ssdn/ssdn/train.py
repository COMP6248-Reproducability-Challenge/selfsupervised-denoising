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
from ssdn.datasets import (
    HDF5Dataset,
    UnlabelledImageFolderDataset,
    FixedLengthSampler,
    NoisyDataset,
    SamplingOrder,
)

from torch.utils.data import Dataset, DataLoader, Sampler

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
    DatasetType,
    NoiseValue,
    HistoryValue,
)
import ssdn.utils.noise as noise

from torch import Tensor
from typing import Tuple, Union
from numbers import Number

from tqdm import tqdm
from ssdn.models import NoiseNetwork
from ssdn.utils.data_format import DataFormat
from ssdn.denoiser import Denoiser

from typing import Dict
import logging
import itertools

import ssdn.logging_helper

import copy
import logging

from torch.utils.tensorboard import SummaryWriter

from typing import List, Callable
import re

import ssdn.cfg

logger = logging.getLogger("ssdn.train")


DEFAULT_CFG = ssdn.cfg.base()
DEFAULT_CFG.update(
    {
        ConfigValue.ALGORITHM: NoiseAlgorithm.SELFSUPERVISED_DENOISING,
        ConfigValue.NOISE_STYLE: "gauss25",
        ConfigValue.NOISE_VALUE: NoiseValue.UNKNOWN_CONSTANT,
        ConfigValue.TRAIN_DATA_PATH: "C:/dsj/deep_learning/coursework/ilsvrc_val.h5",
        ConfigValue.TEST_DATA_PATH: "D:/Downloads/Datasets/BSDS300/images/test",
    }
)


class DenoiserTrainer:
    def __init__(
        self, cfg: Dict, state: Dict = {}, runs_dir: str = "runs", run_dir: str = None
    ):
        self.runs_dir = os.path.abspath(runs_dir)
        self._run_dir = run_dir
        self._writer = None

        self.cfg = cfg
        if self.cfg:
            Denoiser.infer_cfg(self.cfg)
        self.state = state
        self._denoiser: Denoiser = None
        self._train_iter = None

    @property
    def denoiser(self) -> Denoiser:
        return self._denoiser

    @denoiser.setter
    def denoiser(self, denoiser: Denoiser):
        self._denoiser = denoiser
        # Refresh optimiser for parameters
        self.init_optimiser()

    def init_optimiser(self):
        """Create a new Adam optimiser for the current denoiser parameters.
        Uses defaults except a reduced beta2. When optimiser is accessed via
        its property the learning rate will be updated appropriately for the
        current training state.
        """
        parameters = self.denoiser.parameters()
        self._optimizer = optim.Adam(parameters, betas=[0.9, 0.99])

    def new_target(self):
        """Create a new Denoiser to train. Resets training state.
        """
        self.denoiser = Denoiser(self.cfg)
        self.init_state()

    def init_state(self):
        self.state[StateValue.INITIALISED] = True
        self.state[StateValue.ITERATION] = 0
        self.state[StateValue.HISTORY] = {}
        history_state = self.state[StateValue.HISTORY]
        history_state[HistoryValue.TRAIN] = {}
        history_state[HistoryValue.EVAL] = {}
        history_state[HistoryValue.TIMINGS] = ssdn.utils.TrackedTime.defaultdict()

    def train(self):
        if self.denoiser is None:
            self.new_target()
        denoiser = self.denoiser
        # Ensure writer is initialised
        _ = self.writer
        ssdn.logging_helper.setup(self.run_dir_path, "log.txt")
        logger.info(separator())

        logger.info("Loading Training Dataset...")
        self.trainloader, self.trainset, self.train_sampler = self.train_data()
        logger.info("Loaded Training Dataset.")
        logger.info("Loading Test Dataset...")
        self.testloader, self.testset, self.test_sampler = self.test_data()
        logger.info("Loaded Test Dataset.")

        logger.info(separator())
        logger.info("TRAINING STARTED")
        logger.info(separator())

        # Initialise metrics
        self.reset_metrics()
        # Use history for metric tracking
        history = self.state[StateValue.HISTORY]
        train_history = self.state[StateValue.HISTORY][HistoryValue.TRAIN]

        # Run for trainloader, use internal loop break so that interval checks
        # can run at start of loop
        data_itr = iter(self.trainloader)
        while True:
            iteration = self.state[StateValue.ITERATION]
            history[HistoryValue.TIMINGS]["total"].update()

            # Run evaluation if interval elapsed
            if (
                iteration % self.cfg[ConfigValue.EVAL_INTERVAL] == 0
            ) and self.testloader is not None:
                output_index = min(len(self.testloader), len(self.testset)) - 1
                self.evaluate(
                    self.testloader,
                    reset_metrics=False,
                    output_callback=self.validation_output_callback(output_index),
                )
            # Print and reset trackers if interval elapsed
            if iteration % self.cfg[ConfigValue.PRINT_INTERVAL] == 0:
                self.accumulate_metrics()
                history[HistoryValue.TIMINGS]["last_print"].update()
                logger.info(self.state_str(eval_prefix="VALID"))
                history[HistoryValue.TIMINGS]["last_print"].total = 0
                self.write_metrics(eval_prefix="valid")
                self.reset_metrics()
            # Save snapshots of model and training if interval elapsed
            if iteration % self.cfg[ConfigValue.SNAPSHOT_INTERVAL] == 0:
                self.snapshot()
                self.snapshot(model_only=True)
            # --- INTERNAL LOOP BREAK --- #
            if iteration >= self.cfg[ConfigValue.TRAIN_ITERATIONS]:
                break

            # Fetch next data
            data = next(data_itr)
            image_count = data[NoisyDataset.INPUT].shape[0]

            # Run pipeline calculating gradient from loss
            self.denoiser.train()
            optimizer = self.optimizer
            optimizer.zero_grad()
            outputs = denoiser.run_pipeline(data)
            torch.mean(outputs[PipelineOutput.LOSS]).backward()
            optimizer.step()

            # Increment metrics to be recorded at end of print interval
            with torch.no_grad():
                train_history["n"] += image_count
                train_history["loss"] += torch.sum(outputs[PipelineOutput.LOSS])
                # Calculate true PSNR losses for outputs using clean references
                # Known to be patches so no need to unpad
                for key, name in self.img_outputs(prefix="psnr").items():
                    train_history[name] += torch.sum(
                        self.calculate_psnr(outputs, key, unpad=False)
                    )
                # Track extras like estimated standard deviations
                for key, name in self.tracked_outputs().items():
                    train_history[name] += torch.sum(outputs[key])

            # Progress
            self.state[StateValue.ITERATION] += image_count
        logger.info(separator())
        logger.info("TRAINING FINISHED")
        logger.info(separator())

        # Save final output weights
        self.snapshot(
            output_name="final-{}.wt".format(self.denoiser.config_name()),
            subdir="",
            model_only=True,
        )

    def evaluate(
        self,
        dataloader: DataLoader,
        reset_metrics: bool = True,
        output_callback: Callable[int, Tensor, None] = None,
    ):
        self.reset_metrics(eval=reset_metrics, train=False)
        with torch.no_grad():
            return self._evaluate(dataloader, output_callback)

    # def valid_save():
    #     pass

    # def save_img(name, img: Tensor):
    #     path = os.path.join(img_dir, 'img-%07d-%s.%s' % (k, name, ext))
    #     save_image(img, , [0.0, 1.0])

    def _evaluate(
        self, dataloader: DataLoader, output_callback: Callable[int, Dict, None]
    ):
        self.denoiser.eval()

        eval_history = self.state[StateValue.HISTORY][HistoryValue.EVAL]
        idx = 0
        for data in dataloader:
            image_count = data[NoisyDataset.INPUT].shape[0]
            outputs = self.denoiser.run_pipeline(data)

            eval_history["n"] += image_count
            eval_history["loss"] += torch.sum(outputs[PipelineOutput.LOSS])
            # Calculate true PSNR losses for outputs using clean references
            for key, name in self.img_outputs(prefix="psnr").items():
                eval_history[name] += torch.sum(
                    self.calculate_psnr(outputs, key, unpad=True)
                )
            if output_callback:
                output_callback(idx, outputs)
            idx += image_count

            # if (eval_mode and (j < original_validation_image_count)) or ((not eval_mode) and (j == len(validation_images) - 1)): # Export last image, or all if evaluating.
            #     k, ext = (j, 'png') if eval_mode else (n, 'jpg')
            #     save_img('a_nsy',  crop_noisy)      # Noisy input
            #     save_img('b_out',  crop_mu_x)       # Predicted mean
            #     save_img('b_out2', crop_pme)        # Posterior mean estimate (actual output)
            #     save_img('b_std',  crop_net_std)    # Predicted std. dev
            #     save_img('c_cln',  crop_val_input)  # Clean reference image

    def evaluation_output_callback(self, max_output_index: int):
        def callback(output_0_index: int, outputs: Dict):
            output_dir = os.path.join(self.run_dir_path, "imgs")
            os.makedirs(output_dir, exist_ok=True)

            metadata = outputs[PipelineOutput.INPUTS][NoisyDataset.METADATA]
            clean = metadata[NoisyDataset.Metadata.CLEAN]
            output_count = clean.shape[0]
            for bi in range(output_count):
                if output_0_index + bi > max_output_index:
                    return

        return callback

    def validation_output_callback(self, output_index: int):
        def callback(output_0_index: int, outputs: Dict):
            metadata = outputs[PipelineOutput.INPUTS][NoisyDataset.METADATA]
            clean = metadata[NoisyDataset.Metadata.CLEAN]
            output_count = clean.shape[0]
            # If the required image is in this batch export it
            bi = output_index - output_0_index
            if bi >= 0 and bi < output_count:
                output_dir = os.path.join(self.run_dir_path, "imgs")
                os.makedirs(output_dir, exist_ok=True)
    
                out_clean = NoisyDataset.unpad(clean, metadata, batch_index=bi)
                path = os.path.join(
                    output_dir,
                    "{:08d}_cln.png".format(self.state[StateValue.ITERATION]),
                )
                ssdn.utils.save_tensor_image(out_clean, path)

                out_clean = NoisyDataset.unpad(
                    outputs[PipelineOutput.INPUTS][NoisyDataset.INPUT],
                    metadata,
                    batch_index=bi,
                )
                path = os.path.join(
                    output_dir,
                    "{:08d}_nsy.png".format(self.state[StateValue.ITERATION]),
                )
                ssdn.utils.save_tensor_image(out_clean, path)

                out_clean = NoisyDataset.unpad(
                    outputs[PipelineOutput.IMG_DENOISED], metadata, batch_index=bi
                )
                path = os.path.join(
                    output_dir,
                    "{:08d}_out.png".format(self.state[StateValue.ITERATION]),
                )
                ssdn.utils.save_tensor_image(out_clean, path)

                if PipelineOutput.IMG_MU in outputs:
                    out_clean = NoisyDataset.unpad(
                        outputs[PipelineOutput.IMG_DENOISED], metadata, batch_index=bi
                    )
                    path = os.path.join(
                        output_dir,
                        "{:08d}_out_mu.png".format(self.state[StateValue.ITERATION]),
                    )
                    ssdn.utils.save_tensor_image(out_clean, path)

                if PipelineOutput.MODEL_STD_DEV in outputs:
                    print(outputs[PipelineOutput.MODEL_STD_DEV].shape)
                    out_clean = NoisyDataset.unpad(
                        outputs[PipelineOutput.MODEL_STD_DEV], metadata, batch_index=bi
                    )
                    out_clean /= 10.0 / 255
                    path = os.path.join(
                        output_dir,
                        "{:08d}_out_std_dev.png".format(self.state[StateValue.ITERATION]),
                    )
                    ssdn.utils.save_tensor_image(out_clean, path)

            # output_0_index =

            # if output_index == index:
            #     pass

            # for key, name in self.img_outputs(prefix="img").items():
            #     print("Saving " + key)
            #     eval_history[name] += torch.sum(
            #         self.calculate_psnr(outputs, key, unpad=True)
            #     )

        return callback

    @property
    def optimizer(self) -> Optimizer:
        """Fetch optimizer whilst applying cosine ramped learning rate. Aim to only
        call this once per iteration to avoid extra computation.

        Returns:
            Optimizer: Adam optimizer with learning rate set automatically.
        """
        learning_rate = self.learning_rate
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = learning_rate
        return self._optimizer

    @property
    def learning_rate(self):
        return ssdn.utils.compute_ramped_lrate(
            self.state[StateValue.ITERATION],
            self.cfg[ConfigValue.TRAIN_ITERATIONS],
            self.cfg[ConfigValue.LR_RAMPDOWN_FRACTION],
            self.cfg[ConfigValue.LR_RAMPUP_FRACTION],
            self.cfg[ConfigValue.LEARNING_RATE],
        )

    def snapshot(
        self, output_name: str = None, subdir: str = None, model_only: bool = False
    ):
        if subdir is None:
            subdir = "models" if model_only else "training"
        output_dir = os.path.join(self.run_dir_path, subdir)
        os.makedirs(output_dir, exist_ok=True)
        if model_only:
            if output_name is None:
                iteration = self.state[StateValue.ITERATION]
                output_name = "model_{:08d}.wt".format(iteration)
            torch.save(
                self.denoiser.state_dict(), os.path.join(output_dir, output_name)
            )
        else:
            if output_name is None:
                iteration = self.state[StateValue.ITERATION]
                output_name = "model_{:08d}.training".format(iteration)
            torch.save(self.state_dict(), os.path.join(output_dir, output_name))

    def write_metrics(self, eval_prefix: str = "eval"):
        if self.state[StateValue.HISTORY][HistoryValue.TRAIN]["n"] > 0:
            self.write_train_metrics()
        if self.state[StateValue.HISTORY][HistoryValue.EVAL]["n"] > 0:
            self.write_eval_metrics(eval_prefix)
        self.writer.add_scalar(
            "train/learning_rate", self.learning_rate, self.state[StateValue.ITERATION],
        )

    def write_train_metrics(self):
        for metric in self.train_metrics():
            metric_value = self.state[StateValue.HISTORY][HistoryValue.TRAIN][metric]
            self.writer.add_scalar(
                "train/" + metric, metric_value, self.state[StateValue.ITERATION]
            )

    def write_eval_metrics(self, eval_prefix: str = "eval"):
        for metric in self.eval_metrics():
            metric_value = self.state[StateValue.HISTORY][HistoryValue.EVAL][metric]
            self.writer.add_scalar(
                eval_prefix + "/" + metric,
                metric_value,
                self.state[StateValue.ITERATION],
            )

    def accumulate_metrics(self, eval: bool = True, train: bool = True):
        def accumulate_metric_dict(metric_dict: Dict, metrics: List[str]):
            if metric_dict["n"] > 0:
                for metric in metrics:
                    metric_dict[metric] /= metric_dict["n"]

        if train:
            accumulate_metric_dict(
                self.state[StateValue.HISTORY][HistoryValue.TRAIN], self.train_metrics()
            )
        if eval:
            accumulate_metric_dict(
                self.state[StateValue.HISTORY][HistoryValue.EVAL], self.eval_metrics()
            )

    def reset_metrics(self, eval: bool = True, train: bool = True):
        def reset_metric_dict(metric_dict: Dict, metrics: List[str]):
            metric_dict["n"] = 0
            for metric in metrics:
                metric_dict[metric] = 0

        if train:
            history = self.state[StateValue.HISTORY][HistoryValue.TRAIN]
            reset_metric_dict(history, self.train_metrics())
        if eval:
            history = self.state[StateValue.HISTORY][HistoryValue.EVAL]
            reset_metric_dict(history, self.eval_metrics())

    def train_metrics(self) -> List[str]:
        psnr_keys = self.img_outputs(prefix="psnr").values()
        tracked = self.tracked_outputs().values()
        return ["loss"] + list(psnr_keys) + list(tracked)

    def eval_metrics(self) -> List[str]:
        psnr_keys = self.img_outputs(prefix="psnr").values()
        return ["loss"] + list(psnr_keys)

    def state_str(self, eval_prefix: str = "EVAL") -> str:
        state_str = self.train_state_str()
        if self.state[StateValue.HISTORY][HistoryValue.EVAL]["n"] > 0:
            state_str = os.linesep.join(
                [state_str, self.eval_state_str(prefix=eval_prefix)]
            )
        return state_str

    def train_state_str(self) -> str:
        def calc_eta(elapsed: float):
            t = elapsed / self.cfg[ConfigValue.PRINT_INTERVAL]
            r = (
                self.cfg[ConfigValue.TRAIN_ITERATIONS]
                - self.state[StateValue.ITERATION]
            )
            return t * r

        def eta_str(elapsed: float):
            eta = calc_eta(elapsed)
            if eta < 1:
                return "<1s"
            else:
                return ssdn.utils.seconds_to_dhms(eta)

        prefix = "TRAIN"
        summary = "[{:08d}] {:>5} | ".format(self.state[StateValue.ITERATION], prefix)
        metric_str_list = []
        history = self.state[StateValue.HISTORY][HistoryValue.TRAIN]
        for metric in self.train_metrics():
            metric_str_list += ["{}={:8.2f}".format(metric, history[metric])]
        summary += ", ".join(metric_str_list)
        # Add time with ETA
        total_train = self.state[StateValue.HISTORY][HistoryValue.TIMINGS]["total"]
        last_print = self.state[StateValue.HISTORY][HistoryValue.TIMINGS]["last_print"]
        summary += " | [{} ~ ETA: {}]".format(
            ssdn.utils.seconds_to_dhms(total_train.total, trim=False),
            eta_str(last_print.total) if last_print.total > 0 else "???",
        )
        return summary

    def eval_state_str(self, prefix: str = "EVAL"):
        summary = "{:10} {:>5} | ".format("", prefix)
        metric_str_list = []
        history = self.state[StateValue.HISTORY][HistoryValue.EVAL]
        for metric in self.eval_metrics():
            metric_str_list += ["{}={:8.2f}".format(metric, history[metric])]
        summary += ", ".join(metric_str_list)

        return summary

    def img_outputs(self, prefix: str = None):
        outputs = {PipelineOutput.IMG_DENOISED: "out"}
        if self.cfg[ConfigValue.PIPELINE] == Pipeline.SSDN:
            outputs.update({PipelineOutput.IMG_MU: "mu_out"})
        if prefix:
            for output, key in outputs.items():
                outputs[output] = "_".join((prefix, key))
        return outputs

    def tracked_outputs(self) -> Dict:
        if self.cfg[ConfigValue.PIPELINE] == Pipeline.SSDN:
            outputs = {
                PipelineOutput.NOISE_STD_DEV: "noise_std",
                PipelineOutput.MODEL_STD_DEV: "model_std",
            }
        else:
            outputs = {}

        return outputs

    def calculate_psnr(
        self, outputs: Dict, output: PipelineOutput, unpad: bool = True
    ) -> Dict:
        # Get clean reference
        metadata = outputs[PipelineOutput.INPUTS][NoisyDataset.METADATA]
        clean = metadata[NoisyDataset.Metadata.CLEAN]
        if unpad:
            clean = NoisyDataset.unpad(clean, metadata)
            cleaned = NoisyDataset.unpad(outputs[output], metadata)
            zipped = zip(cleaned, clean)
            psnrs = list(
                map(
                    lambda x: ssdn.utils.calculate_psnr(*x, data_format=DataFormat.CHW),
                    zipped,
                )
            )
            psnr = torch.stack(psnrs)
        else:
            psnr = ssdn.utils.calculate_psnr(outputs[output], clean)
        return psnr

    @property
    def writer(self) -> SummaryWriter:
        os.makedirs(self.run_dir_path, exist_ok=True)
        if self._writer is None:
            start_iteration = self.state[StateValue.ITERATION]
            self._writer = SummaryWriter(
                log_dir=self.run_dir_path, purge_step=start_iteration
            )
        return self._writer

    @property
    def run_dir_path(self) -> str:
        return os.path.join(self.runs_dir, self.run_dir)

    def train_config_name(self) -> str:
        def iter_str() -> str:
            iterations = self.cfg[ConfigValue.TRAIN_ITERATIONS]
            if iterations >= 1000000:
                return "iter%dm" % (iterations // 1000000)
            elif iterations >= 1000:
                return "iter%dk" % (iterations // 1000)
            else:
                return "iter%d" % iterations

        config_name_lst = ["train", Denoiser._config_name(self.cfg), iter_str()]
        config_name = "-".join(config_name_lst)
        return config_name

    @property
    def run_dir(self) -> str:

        if self._run_dir is None:
            # Find existing runs
            run_ids = []
            if os.path.exists(self.runs_dir):
                for run_dir_path, _, _ in os.walk(self.runs_dir):
                    run_dir = run_dir_path.split(os.sep)[-1]
                    try:
                        run_ids += [int(run_dir.split("-")[0])]
                    except Exception:
                        continue
            # Calculate the next run name for the current configuration
            next_run_id = max(run_ids) + 1 if len(run_ids) > 0 else 0
            config_name = self.train_config_name()
            run_dir_name = "{:05d}-{}".format(next_run_id, config_name)
            self._run_dir = run_dir_name

        return self._run_dir

    def state_dict(self) -> Dict:
        """A copy of the target denoiser state as well as all data required to continue
        training from the current point.

        Returns:
            Dict: Resumeable state dictionary.
        """
        state_dict = {}
        state_dict["denoiser"] = self.denoiser.state_dict()
        state_dict["state"] = self.state
        state_dict["train_order_iter"] = self.train_sampler.last_iter().state_dict()
        # Reset train order iterator to match the amount of data actually processed
        # not just the amount of data loaded
        state_dict["train_order_iter"]["index"] = self.state[StateValue.ITERATION]
        state_dict["optimizer"] = self.optimizer.state_dict()
        state_dict["rng"] = torch.get_rng_state()
        return state_dict

    def load_state_dict(self, state_dict: Union[Dict, str]):
        """Load the contents of a state dictionary into the current instance such that
        training can be resumed when `train()` is called.

        Args:
            state_dict (str): Either a state dictionary or a path to a state dictionary.
                If a string is provided this will be used to load the state dictionary
                from disk.
        """
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict)
        self.denoiser = Denoiser.from_state_dict(state_dict["denoiser"])
        self.cfg = self.denoiser.cfg
        self.state = state_dict["state"]
        self._train_iter = SamplingOrder.from_state_dict(state_dict["train_order_iter"])
        self._optimizer.load_state_dict(state_dict["optimizer"])
        torch.set_rng_state(state_dict["rng"])

    def train_data(self) -> Tuple[DataLoader, NoisyDataset, Sampler]:
        """Configure the training set using the current configuration.

        Returns:
            Tuple[Dataset, DataLoader, Sampler]: Returns a NoisyDataset object
                wrapping either a folder or HDF5 dataset, a DataLoader for that
                dataset that uses a FixedLengthSampler (also returned).
        """
        cfg = self.cfg
        ssdn.cfg.infer_datasets(cfg)
        transform = RandomCrop(
            cfg[ConfigValue.TRAIN_PATCH_SIZE],
            pad_if_needed=True,
            padding_mode="reflect",
        )
        # Load dataset
        if cfg[ConfigValue.TRAIN_DATASET_TYPE] == DatasetType.FOLDER:
            dataset = UnlabelledImageFolderDataset(
                cfg[ConfigValue.TRAIN_DATA_PATH],
                channels=cfg[ConfigValue.IMAGE_CHANNELS],
                # minimum_size=(ConfigValue.TRAIN_PATCH_SIZE, ConfigValue.TRAIN_PATCH_SIZE),
                transform=transform,
                recursive=True,
            )
        elif cfg[ConfigValue.TRAIN_DATASET_TYPE] == DatasetType.HDF5:
            # It is assumed that the created dataset does not violate the minimum patch size
            dataset = HDF5Dataset(
                cfg[ConfigValue.TRAIN_DATA_PATH],
                transform=transform,
                channels=cfg[ConfigValue.IMAGE_CHANNELS],
            )
        else:
            raise NotImplementedError("Dataset type not implemented")
        # Wrap dataset for creating noisy samples
        dataset = NoisyDataset(
            dataset,
            cfg[ConfigValue.NOISE_STYLE],
            cfg[ConfigValue.ALGORITHM],
            pad_uniform=False,
            pad_multiple=NoiseNetwork.input_wh_mul(),
            square=cfg[ConfigValue.BLINDSPOT],
        )
        # Ensure dataset initialised by loading first bit of data
        _ = dataset[0]
        # Create a dataloader that will sample from this dataset for a fixed number of samples
        sampler = FixedLengthSampler(
            dataset, num_samples=cfg[ConfigValue.TRAIN_ITERATIONS], shuffled=False,
        )
        # Resume train sampler
        if self._train_iter is not None:
            sampler.for_next_iter(self._train_iter)
            self._train_iter = None

        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=cfg[ConfigValue.TRAIN_MINIBATCH_SIZE],
            num_workers=cfg[ConfigValue.DATALOADER_WORKERS],
            pin_memory=cfg[ConfigValue.PIN_DATA_MEMORY],
        )
        return dataloader, dataset, sampler

    def test_data(self) -> Tuple[DataLoader, NoisyDataset, Sampler]:
        """Configure the test set using the current configuration.

        Returns:
            Tuple[Dataset, DataLoader, Sampler]: Returns a NoisyDataset object
                wrapping either a folder or HDF5 dataset, a DataLoader for that
                dataset that uses a FixedLengthSampler (also returned).
        """
        cfg = self.cfg
        ssdn.cfg.infer_datasets(cfg)
        # Load dataset
        if cfg[ConfigValue.TEST_DATASET_TYPE] == DatasetType.FOLDER:
            dataset = UnlabelledImageFolderDataset(
                cfg[ConfigValue.TEST_DATA_PATH],
                recursive=True,
                channels=cfg[ConfigValue.IMAGE_CHANNELS],
            )
        elif cfg[ConfigValue.TEST_DATASET_TYPE] == DatasetType.HDF5:
            dataset = HDF5Dataset(
                cfg[ConfigValue.TEST_DATA_PATH],
                channels=cfg[ConfigValue.IMAGE_CHANNELS],
            )
        else:
            raise NotImplementedError("Dataset type not implemented")
        # Wrap dataset for creating noisy samples
        dataset = NoisyDataset(
            dataset,
            cfg[ConfigValue.NOISE_STYLE],
            cfg[ConfigValue.ALGORITHM],
            pad_uniform=True,
            pad_multiple=NoiseNetwork.input_wh_mul(),
            square=cfg[ConfigValue.BLINDSPOT],
        )
        # Ensure dataset initialised by loading first bit of data
        _ = dataset[0]
        # Create a dataloader that will sample from this dataset for a fixed number of samples
        sampler = FixedLengthSampler(
            dataset,
            num_samples=4,
            # num_samples=ssdn.cfg.test_length(cfg[ConfigValue.TEST_DATASET_NAME], len(dataset)),
            shuffled=False,
        )
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=cfg[ConfigValue.TEST_MINIBATCH_SIZE],
            num_workers=cfg[ConfigValue.DATALOADER_WORKERS],
            pin_memory=cfg[ConfigValue.PIN_DATA_MEMORY],
        )
        return dataloader, dataset, sampler


def separator(cols=100) -> str:
    return "#" * cols


def resume_run(run_dir: str, iteration: int = None) -> DenoiserTrainer:
    """Resume training of a Denoiser model from a previous run.

    Args:
        run_dir (str): The root directory of execution. Resumable training states
            are expected to be in {run_dir}/training.
        iteration (int, optional): The iteration to resume from, if not provided
            the last iteration found will be used.

    Returns:
        DenoiserTrainer: Fully initialised trainer which will update existing
            run directory.
    """
    run_dir = os.path.abspath(run_dir)
    runs_dir = os.path.abspath(os.path.join(run_dir, ".."))
    iterations = {}
    for path in glob.glob(os.path.join(run_dir, "training", "*.training")):
        try:
            iterations[int(re.findall(r"\d+", os.path.basename(path))[0])] = path
        except Exception:
            continue
    if iteration is None:
        if len(iterations) == 0:
            raise ValueError("Run directory contains no training files.")
        iteration = max(iterations.keys())

    load_file_path = iterations[iteration]
    logger.info("Loading from '{}'...".format(load_file_path))
    trainer = DenoiserTrainer(
        None, runs_dir=runs_dir, run_dir=os.path.basename(run_dir)
    )
    trainer.load_state_dict(load_file_path)
    logger.info("Loaded training state.")
    # Cannot trust old absolute times so discard
    for timing in trainer.state[StateValue.HISTORY][HistoryValue.TIMINGS].values():
        timing.forget()
    return trainer


if __name__ == "__main__":

    # values = [ssdn.utils.compute_ramped_lrate(
    #     i,
    #     25000,
    #     DEFAULT_CFG[ConfigValue.LR_RAMPDOWN_FRACTION],
    #     DEFAULT_CFG[ConfigValue.LR_RAMPUP_FRACTION],
    #     DEFAULT_CFG[ConfigValue.LEARNING_RATE],
    # ) for i in range(0, 25000)]
    # import matplotlib.pyplot as plt
    # plt.plot(values)
    # plt.show()
    # exit()

    ssdn.logging_helper.setup()
    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # if True:
    #     trainer = resume_run(
    #         r"C:\dsj\deep_learning\coursework\git\runs\00145-n2c-gauss25"
    #     )
    #     trainer.train()
    #     exit()

    trainer = DenoiserTrainer(DEFAULT_CFG)
    trainer.train()
    Denoiser._config_name(DEFAULT_CFG)
    # Initialise RNG
