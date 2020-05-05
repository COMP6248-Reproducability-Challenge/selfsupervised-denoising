from __future__ import annotations

import os
import math
import torch
import torch.optim as optim
import re
import glob
import logging

import ssdn


from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.transforms import RandomCrop
from ssdn.datasets import (
    HDF5Dataset,
    UnlabelledImageFolderDataset,
    FixedLengthSampler,
    NoisyDataset,
    SamplingOrder,
)

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

from ssdn.models import NoiseNetwork
from ssdn.denoiser import Denoiser
from ssdn.utils import TrackedTime, Metric, separator
from ssdn.utils.data_format import DataFormat
from ssdn.cfg import DEFAULT_RUN_DIR
from typing import Dict, Tuple, Union, Callable
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict, defaultdict

logger = logging.getLogger("ssdn.train")


DEFAULT_CFG = ssdn.cfg.base()
DEFAULT_CFG.update(
    {
        ConfigValue.ALGORITHM: NoiseAlgorithm.NOISE_TO_CLEAN,
        ConfigValue.NOISE_STYLE: "gauss25",
        ConfigValue.NOISE_VALUE: NoiseValue.KNOWN,
        ConfigValue.TRAIN_DATA_PATH: "C:/dsj/deep_learning/coursework/ilsvrc_val.h5",
        ConfigValue.TEST_DATA_PATH: "D:/Downloads/Datasets/BSDS300/images/test",
    }
)


class OrderedDefaultDict(OrderedDict):
    # TODO: Make generic and move
    def __missing__(self, key):
        self[key] = value = Metric()
        return value


class DenoiserTrainer:
    def __init__(
        self,
        cfg: Dict,
        state: Dict = {},
        runs_dir: str = DEFAULT_RUN_DIR,
        run_dir: str = None,
    ):
        self.runs_dir = os.path.abspath(runs_dir)
        self._run_dir = run_dir
        self._writer = None

        self.cfg = cfg
        if self.cfg:
            ssdn.cfg.infer(self.cfg)
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
        """Initialise the state for a fresh training.
        """
        self.state[StateValue.INITIALISED] = True
        self.state[StateValue.ITERATION] = 0
        # History stores events that may happen each iteration
        self.state[StateValue.HISTORY] = {}
        history_state = self.state[StateValue.HISTORY]
        history_state[HistoryValue.TRAIN] = OrderedDefaultDict()
        history_state[HistoryValue.EVAL] = OrderedDefaultDict()
        history_state[HistoryValue.TIMINGS] = defaultdict(TrackedTime)
        self.reset_metrics()

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

        # Use history for metric tracking
        history = self.state[StateValue.HISTORY]
        train_history = self.state[StateValue.HISTORY][HistoryValue.TRAIN]

        # Run for trainloader, use internal loop break so that interval checks
        # can run at start of loop
        data_itr = iter(self.trainloader)
        while True:
            iteration = self.state[StateValue.ITERATION]

            # Run evaluation if interval elapsed
            if (
                iteration % self.cfg[ConfigValue.EVAL_INTERVAL] == 0
            ) and self.testloader is not None:
                self._evaluate(
                    self.testloader, output_callback=self.validation_output_callback(0),
                )

            # Print and reset trackers if interval elapsed
            if iteration % self.cfg[ConfigValue.PRINT_INTERVAL] == 0:
                history[HistoryValue.TIMINGS]["total"].update()
                last_print = history[HistoryValue.TIMINGS]["last_print"]
                last_print.update()
                # Update ETA with metrics captured between prints
                samples = (
                    history[HistoryValue.EVAL]["n"] + history[HistoryValue.TRAIN]["n"]
                )
                self.update_eta(samples, last_print.total)
                # Write to console/file and Tensorboard
                logger.info(self.state_str(eval_prefix="VALID"))
                self.write_metrics(eval_prefix="valid")
                # Reset
                last_print.total = 0
                self.reset_metrics()

            # Save snapshots of model and training if interval elapsed
            if iteration % self.cfg[ConfigValue.SNAPSHOT_INTERVAL] == 0:
                self.snapshot()

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
                train_history["loss"] += outputs[PipelineOutput.LOSS]
                # Calculate true PSNR losses for outputs using clean references
                # Known to be patches so no need to unpad
                for key, name in self.img_outputs(prefix="psnr").items():
                    train_history[name] += self.calculate_psnr(outputs, key, False)
                # Track extra metrics if available
                if PipelineOutput.NOISE_STD_DEV in outputs:
                    output = outputs[PipelineOutput.NOISE_STD_DEV]
                    train_history[PipelineOutput.NOISE_STD_DEV.value] += output * 255
                if PipelineOutput.MODEL_STD_DEV in outputs:
                    output = outputs[PipelineOutput.MODEL_STD_DEV]
                    train_history[PipelineOutput.MODEL_STD_DEV.value] += output * 255

            # Progress
            self.state[StateValue.ITERATION] += image_count

        logger.info(separator())
        logger.info("TRAINING FINISHED")
        logger.info(separator())

        # Save final output weights
        self.snapshot()
        self.snapshot(
            output_name="final-{}.wt".format(self.denoiser.config_name()),
            subdir="",
            model_only=True,
        )

    def evaluate(
        self,
        dataloader: DataLoader,
        output_callback: Callable[int, Tensor, None] = None,
    ):
        self.reset_metrics(train=False)
        return self._evaluate(dataloader, output_callback)

    def _evaluate(
        self, dataloader: DataLoader, output_callback: Callable[int, Dict, None],
    ):
        self.denoiser.eval()
        with torch.no_grad():
            eval_history = self.state[StateValue.HISTORY][HistoryValue.EVAL]
            idx = 0
            for data in dataloader:
                image_count = data[NoisyDataset.INPUT].shape[0]
                outputs = self.denoiser.run_pipeline(data)
                eval_history["n"] += image_count
                eval_history["loss"] += outputs[PipelineOutput.LOSS]
                # Calculate true PSNR losses for outputs using clean references
                for key, name in self.img_outputs(prefix="psnr").items():
                    eval_history[name] += self.calculate_psnr(outputs, key, unpad=True)
                if output_callback:
                    output_callback(idx, outputs)
                idx += image_count

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

    def validation_output_callback(
        self, output_index: int
    ) -> Callable[int, Dict, None]:
        """[summary]

        Args:
            output_index (int): [description]

        Returns:
            Callable[int, Dict, None]: Callback function for `evaluation()`.
        """

        def callback(output_0_index: int, outputs: Dict):
            inp = outputs[PipelineOutput.INPUTS][NoisyDataset.INPUT]
            output_count = inp.shape[0]
            # If the required image is in this batch export it
            bi = output_index - output_0_index
            if bi >= 0 and bi < output_count:
                output_dir = os.path.join(self.run_dir_path, "val_imgs")
                fileformat = "{iter:08}_{desc}.png"
                self._save_image_outputs(outputs, output_dir, fileformat, bi)

        return callback

    def save_image_outputs(
        self,
        outputs: Dict,
        output_dir: str,
        fileformat: str,
        batch_indexes: int = None,
    ):
        """Save all images present in the outputs. If the data is batched all
        values will saved separately.

        Args:
            outputs (Dict): Outputs to extract images from.
            output_dir (str): Output directory for images.
            fileformat (str): Template format for filenames. This can use keyword
                string arguments: (`iter`, TRAIN_ITERATION), (`index`, IMG INDEX),
                (`desc`, IMG DESCRIPTION).
            batch_indexes (int, optional): Batch indexes to save. Defaults to
                None; indicating all.
        """
        if batch_indexes is None:
            metadata = outputs[PipelineOutput.INPUTS][NoisyDataset.METADATA]
            clean = metadata[NoisyDataset.Metadata.CLEAN]
            batch_indexes = range(clean.shape[0])
        for bi in batch_indexes:
            self._save_image_outputs(outputs, output_dir, fileformat, bi)

    def _save_image_outputs(
        self, outputs: Dict, output_dir: str, fileformat: str, batch_index: int
    ):
        """Save all images present in the outputs for a single item in the batch.

        Args:
            outputs (Dict): Outputs to extract images from.
            output_dir (str): Output directory for images.
            fileformat (str): Template format for filenames. This can use keyword
                string arguments: (`iter`, TRAIN_ITERATION), (`index`, IMG INDEX),
                (`desc`, IMG DESCRIPTION).
            batch_index (int, optional): Item index in batch to save.
        """
        os.makedirs(output_dir, exist_ok=True)
        metadata = outputs[PipelineOutput.INPUTS][NoisyDataset.METADATA]

        def make_path(desc: str) -> str:
            filename_args = {
                "iter": self.state[StateValue.ITERATION],
                "index": metadata[NoisyDataset.Metadata.INDEXES][batch_index],
                "desc": desc,
            }
            filename = fileformat.format(**filename_args)
            return os.path.join(output_dir, filename)

        def unpad_save(img: Tensor, desc: str):
            out = NoisyDataset.unpad(img, metadata, batch_index)
            ssdn.utils.save_tensor_image(out, make_path(desc))

        # Save all present outputs
        if NoisyDataset.Metadata.CLEAN in metadata:
            unpad_save(metadata[NoisyDataset.Metadata.CLEAN], "cln")
        if PipelineOutput.INPUTS in outputs:
            unpad_save(outputs[PipelineOutput.INPUTS][NoisyDataset.INPUT], "nsy")
        if PipelineOutput.IMG_DENOISED in outputs:
            unpad_save(outputs[PipelineOutput.IMG_DENOISED], "out")
        if PipelineOutput.IMG_MU in outputs:
            unpad_save(outputs[PipelineOutput.IMG_MU], "out-mu")
        if PipelineOutput.MODEL_STD_DEV in outputs:
            # N.B Scales and adds channel dimension
            img = outputs[PipelineOutput.MODEL_STD_DEV][:, None, ...]
            img /= 10.0 / 255
            unpad_save(img, "out-std")

    def snapshot(
        self, output_name: str = None, subdir: str = None, model_only: bool = False
    ):
        """Save the current Denoiser object with or without all training info.

        Args:
            output_name (str, optional): Fixed name for the output file, will default
                to model_{itertion}.{ext}. {ext} = wt for model weights, and training
                for full training configuration.
            subdir (str, optional): Subdirectory of run directory to store models in.
                Will default to 'models' if only saving the model, 'training'
                otherwise.
            model_only (bool, optional): Whether to only save the model state
                dictionary. Defaults to False.
        """
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
        """Writes the accumulated (mean) of each metric in the evaluation and train
        history dictionaries to the current tensorboard writer. Also tracks the
        learning rate.

        Args:
            eval_prefix (str, optional): Subsection for evaluation metrics to be
                written to in the case there are validation and test sets.
                Defaults to "eval".
        """

        def write_metric_dict(metric_dict: Dict, prefix: str):
            for key, metric in metric_dict.items():
                if isinstance(metric, Metric) and not metric.empty():
                    self.writer.add_scalar(
                        prefix + "/" + key,
                        metric.accumulated(),
                        self.state[StateValue.ITERATION],
                    )

        # Training metrics
        metric_dict = self.state[StateValue.HISTORY][HistoryValue.TRAIN]
        write_metric_dict(metric_dict, "train")
        self.writer.add_scalar(
            "train/learning_rate", self.learning_rate, self.state[StateValue.ITERATION],
        )
        # Eval metrics
        metric_dict = self.state[StateValue.HISTORY][HistoryValue.EVAL]
        write_metric_dict(metric_dict, eval_prefix)

    def state_str(self, eval_prefix: str = "EVAL") -> str:
        """[summary]

        Args:
            eval_prefix (str, optional): String to put before evaluation lines.
                Defaults to "EVAL".

        Returns:
            str: String containing all available metrics.
        """
        state_str = self.train_state_str()
        if self.state[StateValue.HISTORY][HistoryValue.EVAL]["n"] > 0:
            state_str = os.linesep.join(
                [state_str, self.eval_state_str(prefix=eval_prefix)]
            )
        return state_str

    def train_state_str(self) -> str:
        def eta_str():
            timings = self.state[StateValue.HISTORY][HistoryValue.TIMINGS]
            if "eta" in timings:
                eta = timings["eta"]
                if eta < 1:
                    return "<1s"
                else:
                    return ssdn.utils.seconds_to_dhms(eta)
            else:
                return "???"

        history = self.state[StateValue.HISTORY]
        prefix = "TRAIN"
        summary = "[{:08d}] {:>5} | ".format(self.state[StateValue.ITERATION], prefix)
        metric_str_list = []
        train_metrics = history[HistoryValue.TRAIN]
        for key, metric in train_metrics.items():
            if isinstance(metric, Metric) and not metric.empty():
                metric_str_list += ["{}={:8.2f}".format(key, metric.accumulated())]
        summary += ", ".join(metric_str_list)
        # Add time with ETA
        total_train = history[HistoryValue.TIMINGS]["total"]
        if len(metric_str_list) > 0:
            summary += " | "
        summary += "[{} ~ ETA: {}]".format(
            ssdn.utils.seconds_to_dhms(total_train.total, trim=False), eta_str(),
        )
        return summary

    def eval_state_str(self, prefix: str = "EVAL") -> str:
        """[summary]

        Args:
            eval_prefix (str, optional): String to put at start of state string.
                Defaults to "EVAL".

        Returns:
            str: [description]
        """
        summary = "{:10} {:>5} | ".format("", prefix)
        metric_str_list = []
        eval_metrics = self.state[StateValue.HISTORY][HistoryValue.EVAL]
        for key, metric in eval_metrics.items():
            if isinstance(metric, Metric) and not metric.empty():
                metric_str_list += ["{}={:8.2f}".format(key, metric.accumulated())]
        summary += ", ".join(metric_str_list)

        return summary

    def reset_metrics(self, eval: bool = True, train: bool = True):
        """Clear any metric value and reset the data count for both the evaluation
        and training metric histories.

        Args:
            eval (bool, optional): Reset the eval metrics. Defaults to True.
            train (bool, optional): Reset the train metrics. Defaults to True.
        """

        def reset_metric_dict(metric_dict: Dict):
            metric_dict["n"] = 0
            for key, metric in metric_dict.items():
                if isinstance(metric, Metric):
                    metric.reset()

        if train:
            reset_metric_dict(self.state[StateValue.HISTORY][HistoryValue.TRAIN])
        if eval:
            reset_metric_dict(self.state[StateValue.HISTORY][HistoryValue.EVAL])

    def img_outputs(self, prefix: str = None) -> Dict:
        """[summary]

        Args:
            prefix (str, optional): [description]. Defaults to None.

        Returns:
            Dict: [description]
        """        
        outputs = {PipelineOutput.IMG_DENOISED: "out"}
        if self.cfg[ConfigValue.PIPELINE] == Pipeline.SSDN:
            outputs.update({PipelineOutput.IMG_MU: "mu_out"})
        if prefix:
            for output, key in outputs.items():
                outputs[output] = "_".join((prefix, key))
        return outputs

    def calculate_psnrs():
        pass

    @staticmethod
    def calculate_psnr(
        outputs: Dict, output: PipelineOutput, unpad: bool = True
    ) -> Tensor:
        # Get clean reference
        metadata = outputs[PipelineOutput.INPUTS][NoisyDataset.METADATA]
        clean = metadata[NoisyDataset.Metadata.CLEAN]
        if unpad:
            clean = NoisyDataset.unpad(clean, metadata)
            clean = [c.to(outputs[output].device) for c in clean]
            cleaned = NoisyDataset.unpad(outputs[output], metadata)
            zipped = zip(cleaned, clean)
            psnrs = map(
                lambda x: ssdn.utils.calculate_psnr(*x, data_format=DataFormat.CHW),
                zipped,
            )
            psnr = torch.stack(list(psnrs))
        else:
            psnr = ssdn.utils.calculate_psnr(outputs[output], clean)
        return psnr

    @property
    def writer(self) -> SummaryWriter:
        """The Tensorboard Summary Writer. When this method is first called a new
        SummaryWriter will be created targetting the run directory. Any data present
        in the Tensorboard for the current run (i.e. if resuming) will be removed
        from the current iteration onwards.

        Returns:
            SummaryWriter: Initialised Tensorboard SummaryWriter.
        """
        os.makedirs(self.run_dir_path, exist_ok=True)
        if self._writer is None:
            start_iteration = self.state[StateValue.ITERATION]
            self._writer = SummaryWriter(
                log_dir=self.run_dir_path, purge_step=start_iteration
            )
        return self._writer

    @property
    def run_dir_path(self) -> str:
        """
        Returns:
            str: Full path to run directory (`run_dir`) inside runs directory.
        """
        return os.path.join(self.runs_dir, self.run_dir)

    @property
    def run_dir(self) -> str:
        """The run path to use for this run. When this method is first called
        a new directory name will be generated using the next run ID and current
        configuration.

        Returns:
            str: Run directory name, note this is not a full path.
        """
        if self._run_dir is None:
            config_name = self.train_config_name()
            next_run_id = self.next_run_id()
            run_dir_name = "train-{:05d}-{}".format(next_run_id, config_name)
            self._run_dir = run_dir_name

        return self._run_dir

    def next_run_id(self) -> int:
        """For the run directory, search for any previous runs and return an
        ID that is one greater.

        Returns:
            int: Run ID Number
        """
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
        return next_run_id

    def update_eta(
        self, samples: int, elapsed: float, smoothing_factor: int = 0.95
    ) -> float:
        """Update the tracked ETA based on how many samples have been processed
        over a given time period. Estimated time treats evaluation samples as
        taking the same amount of time as training samples.

        Args:
            samples (int): The number of samples in the processing window.
            elapsed (float): The amount of time elapsed in the processing window.
            smoothing_factor (int, optional): The proportion to use of this value
                against the previous ETA. For frequent calls to `update_eta` use
                a low value, for infrequent use a high value. Defaults to 0.95.

        Returns:
            float: The estimated remaining time in seconds.
        """
        timings = self.state[StateValue.HISTORY][HistoryValue.TIMINGS]
        if samples <= 0:
            return timings["eta"]
        # Time per number of processed samples
        t = elapsed / samples
        # Remaining iterations
        r = self.cfg[ConfigValue.TRAIN_ITERATIONS] - self.state[StateValue.ITERATION]
        # Add on eval iterations
        r += len(self.testloader) * math.ceil(r / self.cfg[ConfigValue.EVAL_INTERVAL])
        new_eta = t * r
        if "eta" not in timings:
            timings["eta"] = new_eta
        else:
            sf = smoothing_factor
            timings["eta"] = sf * new_eta + (1 - sf) * timings["eta"]
        return timings["eta"]

    def train_config_name(self) -> str:
        def iter_str() -> str:
            iterations = self.cfg[ConfigValue.TRAIN_ITERATIONS]
            if iterations >= 1000000:
                return "iter%dm" % (iterations // 1000000)
            elif iterations >= 1000:
                return "iter%dk" % (iterations // 1000)
            else:
                return "iter%d" % iterations

        config_name_lst = ["train", ssdn.cfg.config_name(self.cfg), iter_str()]
        config_name = "-".join(config_name_lst)
        return config_name

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
            dataset, num_samples=cfg[ConfigValue.TRAIN_ITERATIONS], shuffled=True,
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
            num_samples=ssdn.cfg.test_length(cfg[ConfigValue.TEST_DATASET_NAME]),
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
    # torch.set_default_dtype(torch.float64)

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
    # Initialise RNG
